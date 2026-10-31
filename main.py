#!/usr/bin/env python3
"""3AM Tales — the orchestrator.

Runs the agent team end to end, one part of an illustrated horror series per day:

  Analyst → Researcher (new series or next part) → Writer ⇄ Critic → Voice cast → Illustrator
  (character portraits → shots → camera moves) → Composer (original score) → Editor → Caption
  → Publisher → Memory (series progress + character portraits)

Usage:
  python main.py              # full run: create and post today's part
  python main.py --dry-run    # create it but don't post (review it in out/)
  python main.py --preview    # quick ~30 s sample to check characters, art and voices
  python main.py --mock       # offline test: no API keys needed, placeholder art and tones
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone

from agents import analyst, animator, caption, composer, cover, critic, editor, gpu, illustrator, publisher, qa, researcher, sound, upscaler, voice, writer, youtube
from agents.common import ROOT, LLM, Context, LLMBlocked, LLMError, load_config, log
from agents.memory import Memory


def produce_script(ctx: Context, memory: Memory) -> tuple[dict, dict, dict, dict]:
    s = ctx.cfg["story"]
    rejected: list[str] = []
    for _ in range(s["max_ideas_per_run"]):
        series, fmt = researcher.todays_series(ctx, memory, avoid_extra=rejected)
        try:
            script = writer.write(ctx, series, fmt)
            review = critic.review(ctx, script, series, fmt)
        except LLMBlocked as e:
            log("Orchestrator", f"{e}; trying a different story")
            rejected.append(series["title"])
            continue
        best = (review["score"], script, review)
        rev = 0
        while not review["approved"] and rev < s["max_revisions"]:
            rev += 1
            notes = (review.get("feedback", "") + " " + review.get("safety_issues", "")).strip()
            try:
                script = writer.revise(ctx, script, notes, series, fmt)
                review = critic.review(ctx, script, series, fmt)
            except LLMError as e:
                # a blocked or failed revision must not crash the run: keep the best draft so far
                log("Orchestrator", f"revision stopped ({str(e).splitlines()[0][:120]}); keeping the best draft")
                score, script, review = best
                break
            if review.get("safety_ok") and review["score"] > best[0]:
                best = (review["score"], script, review)
        if review["approved"]:
            return series, fmt, script, review
        score, script, review = best
        if review.get("safety_ok") and score >= s["min_critic_score"] - 10:
            log("Orchestrator", f"using the best draft ({score}/100), close enough to the bar")
            return series, fmt, script, review
        if series.get("characters"):
            # an ongoing series must continue; post the safest best draft rather than abandoning it
            if review.get("safety_ok"):
                log("Orchestrator", f"continuing the series with the best draft ({score}/100)")
                return series, fmt, script, review
        rejected.append(series["title"])
        log("Orchestrator", "story rejected; trying a new one")
    raise RuntimeError("No script passed the Critic today — nothing will be posted.")


IST = timezone(timedelta(hours=5, minutes=30))


def _post_target(cfg: dict, now_utc: datetime) -> datetime:
    """Today's posting time (India time) as an aware datetime."""
    hh, mm = (int(x) for x in str(cfg.get("schedule", {}).get("post_time_ist", "21:17")).split(":"))
    return now_utc.astimezone(IST).replace(hour=hh, minute=mm, second=0, microsecond=0)


def _hours_since_last_post(memory: Memory, now_utc: datetime) -> float | None:
    for p in reversed(memory.posts):
        if not p.get("media_id"):
            continue
        try:
            when = datetime.fromisoformat(p.get("posted_at", ""))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return (now_utc - when).total_seconds() / 3600
    return None


def schedule_check(cfg: dict, memory: Memory, mode: str, now_utc: datetime) -> tuple[bool, str]:
    """Should this run make a reel? (auto = the daily schedule; post = a manual 'post now')"""
    sc = cfg.get("schedule", {})
    since = _hours_since_last_post(memory, now_utc)
    if mode == "post":
        gap = float(sc.get("post_mode_min_gap_hours", 3))
        if since is not None and since < gap:
            return False, (f"a reel went out only {since * 60:.0f} min ago; not posting another one so soon "
                           f"(wait {gap:.0f} h between posts, or use mode 'dry_run' to just make one)")
        return True, "mode 'post': making and posting a reel now"
    target = _post_target(cfg, now_utc)
    window = float(sc.get("start_window_hours", 5))
    now = now_utc.astimezone(IST)
    slot_start = target - timedelta(hours=window)
    if since is not None and now - timedelta(hours=since) >= slot_start:
        return False, f"today's reel already went out {since:.1f} h ago; nothing to do"
    if now < target - timedelta(hours=window):
        return False, (f"too early ({now:%H:%M} IST): a later scheduled run will make today's reel and post it at "
                       f"{target:%H:%M} IST")
    if now <= target:
        return True, f"making today's reel now; it will be posted at {target:%H:%M} IST"
    return True, f"GitHub started this run late ({now:%H:%M} IST): making the reel and posting it as soon as it's ready"


def wait_for_post_time(cfg: dict, mode: str) -> None:
    """auto mode: the reel is ready early; hold it until the posting time."""
    if mode != "auto":
        return
    target = _post_target(cfg, datetime.now(timezone.utc))
    left = (target - datetime.now(IST)).total_seconds()
    if left <= 0:
        return
    log("Orchestrator", f"reel ready; waiting {left / 60:.0f} min to post at {target:%H:%M} IST")
    while True:
        left = (target - datetime.now(IST)).total_seconds()
        if left <= 0:
            break
        time.sleep(min(left, 1800))
        if left > 1800:
            log("Orchestrator", f"… {max(0.0, left - 1800) / 60:.0f} min to go")
    log("Orchestrator", f"it's {datetime.now(IST):%H:%M} IST — posting")


def _trim_for_preview(script: dict, cuts: int = 12) -> dict:
    """Preview mode: keep the first cuts only (about 25-30 s) to check look and sound cheaply."""
    shots = script["shots"][:cuts]
    keep = {sh["image_id"] for sh in shots}
    return {**script, "shots": shots, "images": [im for im in script["images"] if im["id"] in keep]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build the reel but do not post it")
    ap.add_argument("--preview", action="store_true", help="quick ~30 s sample (characters, look, voices); not posted")
    ap.add_argument("--mock", action="store_true", help="offline test with placeholder art and tones")
    args = ap.parse_args()
    mode = (os.getenv("RUN_MODE") or "").lower()
    preview = args.preview or mode == "preview"
    dry = args.dry_run or args.mock or preview or mode == "dry_run" or \
        os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")

    cfg = load_config()
    if preview:
        cfg["qa"]["min_seconds"] = 10
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work = ROOT / "work" / stamp
    out_dir = ROOT / "out"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    ctx = Context(cfg=cfg, workdir=work, mock=args.mock, dry_run=dry)
    ctx.llm = LLM(cfg, mock=args.mock)
    if mode == "gpu_setup":
        log("Orchestrator", "GPU SETUP + BENCHMARK · code v7.1.13 (nothing is posted)")
        mem = Memory()
        gpu.setup_and_benchmark(ctx, mem.data, out_dir)
        mem.save()
        shutil.rmtree(work, ignore_errors=True)
        return 0
    memory = Memory()
    if not args.mock:
        import atexit
        atexit.register(memory.save)     # even if the run crashes, remember what it used today
    if mode in ("auto", "post") and not args.mock:
        go, why = schedule_check(cfg, memory, mode, datetime.now(timezone.utc))
        log("Orchestrator", why)
        if not go:
            shutil.rmtree(work, ignore_errors=True)
            return 0
    t0 = time.time()
    label = "MOCK" if args.mock else "PREVIEW" if preview else "DRY RUN" if dry else "LIVE"
    log("Orchestrator", f"run {stamp} · {label} · code v7.1.13")

    if not args.mock:
        missing = [k for k in ("GEMINI_API_KEY",) if not os.getenv(k)]
        if not dry:
            missing += [k for k in ("IG_ACCESS_TOKEN", "IG_USER_ID") if not os.getenv(k)]
        if missing:
            log("Orchestrator", f"missing secrets: {', '.join(missing)} — see README.md")
            return 2
        if not (os.getenv("CF_ACCOUNT_ID") and os.getenv("CF_API_TOKEN")):
            log("Orchestrator", "note: CF_ACCOUNT_ID / CF_API_TOKEN not set — using the backup image "
                                "generator, so characters will be less consistent (see README.md)")

    if not dry and not args.mock:
        ok, msg = publisher.check_token(ctx)
        log("Orchestrator", msg)
        # Instagram tokens expire after 60 days: remember when this token was first used and warn in time
        import hashlib
        th = hashlib.sha256(os.getenv("IG_ACCESS_TOKEN", "").encode()).hexdigest()[:12]
        tk = memory.data.get("ig_token") or {}
        if tk.get("hash") != th:
            tk = {"hash": th, "first_seen": datetime.now(timezone.utc).strftime("%Y-%m-%d")}
            memory.data["ig_token"] = tk
        age = (datetime.now(timezone.utc) - datetime.strptime(tk["first_seen"], "%Y-%m-%d").replace(
            tzinfo=timezone.utc)).days
        if age >= 50:
            log("Orchestrator", f"⚠️ WARNING: the Instagram token is about {age} days old and stops working at 60 days. "
                                "Create a new one now (README step 8) and update the IG_ACCESS_TOKEN secret.")
        if not ok:
            log("Orchestrator", "STOPPED before making the reel: refresh IG_ACCESS_TOKEN (README step 8) and run again.")
            return 4

    # 1. learn from past posts
    analyst.run(ctx, memory)

    # 2-4. today's part → script → critique loop
    series, fmt, script, review = produce_script(ctx, memory)
    final = series["part"] >= series["total_parts"]
    script["voice_style"] = series.get("voice_style", "dialogue")
    script["series_voice_engine"] = series.get("voice_engine")
    script["narrator_gender"] = series.get("narrator_gender", "male")
    if preview:
        script = _trim_for_preview(script)
        log("Orchestrator", f"preview: first {len(script['shots'])} cuts only")

    # 5. the voice cast performs it
    v = cfg["video"]
    voice_wav, timeline, shot_starts, voice_len = voice.perform(ctx, script)
    total = round(v["lead_in_seconds"] + voice_len + v["tail_seconds"], 3)
    reel_cap = float(cfg["story"].get("max_reel_seconds", 0) or 0)
    if reel_cap and total > reel_cap:
        # a gentle, pitch-preserving speed-up (max 8%) keeps the reel a YouTube Short (≤ 3:00)
        room = reel_cap - v["lead_in_seconds"] - v["tail_seconds"]
        speed = min(1.08, voice_len / room)
        fast = work / "voice_fast.wav"
        import subprocess as _sp
        _sp.run(["ffmpeg", "-y", "-v", "error", "-i", str(voice_wav), "-af", f"atempo={speed:.4f}",
                 "-c:a", "pcm_s24le", str(fast)], check=True)
        voice_wav, voice_len = fast, voice_len / speed
        shot_starts = [s / speed for s in shot_starts]
        timeline = [{**t, "start": round(t["start"] / speed, 3), "end": round(t["end"] / speed, 3)} for t in timeline]
        total = round(v["lead_in_seconds"] + voice_len + v["tail_seconds"], 3)
        log("Orchestrator", f"voice sped up {100 * (speed - 1):.1f}% so the reel is {total:.0f}s "
                            f"({'a YouTube Short' if total <= 180 else 'still over 3:00 — YouTube will treat it as a normal video'})")
    # no picture longer than ~6 s: long lines get extra cuts at the storyteller's breaths
    shot_starts = editor.split_long_shots(ctx, script, shot_starts, voice_wav, voice_len)
    if total > 600:
        raise RuntimeError(f"Reel came out at {total:.0f}s, which is far too long; not posting it.")

    # 6. illustrations: character portraits (kept for the series) → drawings → inspection → camera moves
    budget = float(cfg.get("billing", {}).get("monthly_budget_usd", 9.0))
    spent = memory.month_spend()
    image_budget = max(0.0, budget - spent - 0.15)   # keep a little for this run's voices and writing
    if not args.mock and image_budget < 0.05 and cfg["visuals"].get("gemini_hero_shots", 0):
        log("Orchestrator", f"this month's Gemini budget is used (${spent:.2f} of ${budget:.2f}); drawing with free providers")
    engine = gpu.Engine(ctx, memory.data)
    studio = illustrator.Studio(ctx, gemini_budget_usd=image_budget, first=engine.picture_provider())
    durations = editor.shot_durations(shot_starts, total, v["lead_in_seconds"])
    saved = series.get("ref_files") or {}
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")      # Cloudflare's allowance resets 00:00 UTC (5:30 AM IST)
    ledger = memory.data.setdefault("cf_day", {"date": day, "used": 0.0})
    if ledger.get("date") != day:
        ledger.update(date=day, used=0.0)
    affordable = studio.plan(len(script["images"]), sum(1 for c in script["characters"] if c["id"] not in saved),
                             ledger["used"])
    if affordable < len(script["images"]):
        if affordable < 6 and not args.mock:
            log("Orchestrator", f"STOPPED: today's free Cloudflare picture allowance is almost used up (earlier runs "
                                f"today used ~{ledger['used']:.0f} of 10,000). It resets at 5:30 AM IST — run again after "
                                "that. Tip: avoid test runs on the day of a post.")
            memory.save()
            shutil.rmtree(work, ignore_errors=True)
            return 5
        merged = illustrator.limit_images(script, affordable)
        log("Orchestrator", f"today's allowance pays for {affordable} pictures: {merged} pictures merged into "
                            "earlier ones (reused, like the reference reels)")
    used_before = float(ledger["used"])

    def _record_cloudflare_use() -> None:
        ledger["used"] = 10000.0 if studio.quota_hit else round(used_before + studio.neurons)
    if not args.mock:
        import atexit
        atexit.register(_record_cloudflare_use)   # runs before the memory save, even after a crash
    refs = illustrator.character_refs(ctx, studio, series, script)
    heroes = illustrator.hero_images(ctx, script, durations)
    try:
        image_paths = illustrator.illustrate(ctx, studio, script, refs, heroes)
    except illustrator.ImageError as e:
        _record_cloudflare_use()
        log("Orchestrator", f"STOPPED: no pictures could be drawn ({str(e)[:160]}). "
                            + ("Today's free Cloudflare picture allowance is used up; it resets at 5:30 AM IST. "
                               if studio.quota_hit else "") + "Nothing was posted.")
        memory.save()
        shutil.rmtree(work, ignore_errors=True)
        return 7
    illustrator.quality_check(ctx, studio, script, refs, image_paths)
    _record_cloudflare_use()
    image_paths = upscaler.upscale(ctx, image_paths)
    motion = engine.motion_clips(script["shots"], image_paths, durations)
    clips = animator.animate(ctx, image_paths, script["shots"], durations, motion)
    engine.finish()

    # 7. original score
    score = None
    if cfg["music"].get("enabled", True):
        score = composer.compose(ctx, script["shots"], shot_starts, voice_len, v["lead_in_seconds"], total)
        try:
            score = sound.add_effects(ctx, score, script, shot_starts, v["lead_in_seconds"], total)
        except Exception as e:  # noqa: BLE001 — sound effects are a bonus, never a reason to fail
            log("Orchestrator", f"sound effects skipped ({str(e)[:120]})")

    # 8. edit
    end_card = ("The End", "Follow for more kahaniyan") if final else \
        ("To Be Continued", f"Follow For Part {series['part'] + 1}")
    overlay = work / "overlay.ass"
    speakers = ["N"] + [c["id"] for c in script["characters"]]
    editor.build_overlay(ctx, timeline, speakers, v["lead_in_seconds"], total, end_card, overlay,
                         hook_text=script.get("hook_text", ""))
    reel = out_dir / f"reel-{stamp}.mp4"
    editor.render(ctx, clips, durations, voice_wav, score, overlay, total, reel)

    # 9. final quality gate
    report = qa.check(ctx, reel, script, total)

    # paid Gemini usage estimate for this run (images + voices + writing), counted against the monthly budget
    run_cost = (studio.gemini_images * illustrator.NanoBanana.PRICE
                + (voice_len * 0.000225 if ctx.notes.get("voice_engine") == "gemini" else 0.0)
                + ctx.llm.calls * 0.01)
    if not args.mock:
        memory.add_spend(run_cost)
    report["gemini_images"] = studio.gemini_images
    report["gpu_motion_clips"] = len(motion)
    report["modal_usd"] = round(engine.run_cost, 3)
    report["cloudflare_neurons"] = round(studio.neurons)
    report["hq_images"] = studio.hq_count
    log("Orchestrator", f"Cloudflare ≈ {studio.neurons:.0f} of 10,000 free neurons today "
                        f"({studio.hq_count} high-quality drawings)")
    report["est_cost_usd"] = round(run_cost, 3)
    report["month_spend_usd"] = round(memory.month_spend(), 2)
    log("Orchestrator", f"Gemini usage this run ≈ ${run_cost:.2f} ({studio.gemini_images} Nano Banana images); "
                        f"this month ≈ ${memory.month_spend():.2f} of ${budget:.2f}")

    # 10. caption
    cap = caption.write_caption(ctx, script, series, fmt)
    cover_path = None
    try:
        cover_path = cover.make(ctx, script, series, image_paths, out_dir / f"cover-{stamp}.jpg")
        _, cut = cover.pick_image(script, image_paths)
        ctx.notes["cover_offset_ms"] = int((v["lead_in_seconds"] + shot_starts[cut] + 0.4) * 1000)
    except Exception as e:  # noqa: BLE001 — a cover problem must never stop the post
        log("Orchestrator", f"cover skipped ({str(e)[:120]})")

    post = {"title": f"{series['title']} — Part {series['part']}", "series_title": series["title"],
            "series_id": series["id"], "part": series["part"], "total_parts": series["total_parts"],
            "format": fmt["id"], "hook": script.get("hook", ""), "critic_score": review["score"],
            "voice_engine": ctx.notes.get("voice_engine", ""), "images": ctx.notes.get("image_providers", []),
            "duration_s": total, "file": reel.name, "caption": cap, "qa": report}
    (out_dir / f"post-{stamp}.json").write_text(
        json.dumps({**post, "script": script, "review": review}, indent=2, ensure_ascii=False), encoding="utf-8")
    pics = out_dir / f"pictures-{stamp}"
    pics.mkdir(exist_ok=True)
    for cid, p in refs.items():
        shutil.copy(p, pics / f"character-{cid}.png")
    for iid, p in image_paths.items():
        shutil.copy(p, pics / f"{iid}.png")

    # 11. publish + remember
    if dry:
        log("Orchestrator", f"{label.lower()} — reel saved to out/{reel.name} (not posted; series not advanced)")
    elif not report["ok"]:
        log("Orchestrator", "NOT POSTED because the quality check failed: " + "; ".join(report["errors"]))
        memory.save()
        shutil.rmtree(work, ignore_errors=True)
        return 3
    else:
        wait_for_post_time(cfg, mode)
        if cfg.get("youtube", {}).get("enabled", True) and youtube.ready():
            try:
                post.update(youtube.upload(ctx, reel, series, cap))
            except Exception as e:  # noqa: BLE001 — YouTube problems never stop the Instagram post
                log("YouTube", f"not uploaded ({str(e)[:220]})")
        try:
            result = publisher.publish_reel(ctx, reel, cap, cover_path)
        except Exception as e:  # noqa: BLE001 — never lose the finished reel
            log("Orchestrator", f"NOT POSTED: the upload to Instagram failed ({str(e)[:200]}). The finished reel and its "
                                f"cover are in this run's Artifacts (out/{reel.name}) — you can post them from the app.")
            memory.save()
            return 6
        post.update(result)
        memory.add_post(post)
        series["voice_engine"] = series.get("voice_engine") or ctx.notes.get("voice_engine")
        memory.advance_series(series, script, refs)

    if not args.mock:
        memory.save()
    shutil.rmtree(work, ignore_errors=True)
    log("Orchestrator", f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
