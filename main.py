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
from datetime import datetime

from agents import analyst, animator, caption, composer, critic, editor, illustrator, publisher, qa, researcher, upscaler, voice, writer
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
    if mode == "style_test":
        from agents import style_test
        log("Orchestrator", "STYLE TEST — drawing the same scenes in every art style (nothing is posted)")
        style_test.run(ctx, out_dir)
        shutil.rmtree(work, ignore_errors=True)
        return 0
    memory = Memory()
    t0 = time.time()
    label = "MOCK" if args.mock else "PREVIEW" if preview else "DRY RUN" if dry else "LIVE"
    log("Orchestrator", f"run {stamp} · {label}")

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
    if total > 600:
        raise RuntimeError(f"Reel came out at {total:.0f}s, which is far too long; not posting it.")

    # 6. illustrations: character portraits (kept for the series) → drawings → inspection → camera moves
    budget = float(cfg.get("billing", {}).get("monthly_budget_usd", 9.0))
    spent = memory.month_spend()
    image_budget = max(0.0, budget - spent - 0.15)   # keep a little for this run's voices and writing
    if not args.mock and image_budget < 0.05 and cfg["visuals"].get("gemini_hero_shots", 0):
        log("Orchestrator", f"this month's Gemini budget is used (${spent:.2f} of ${budget:.2f}); drawing with free providers")
    studio = illustrator.Studio(ctx, gemini_budget_usd=image_budget)
    durations = editor.shot_durations(shot_starts, total, v["lead_in_seconds"])
    saved = series.get("ref_files") or {}
    studio.plan(len(script["images"]), sum(1 for c in script["characters"] if c["id"] not in saved))
    refs = illustrator.character_refs(ctx, studio, series, script)
    heroes = illustrator.hero_images(ctx, script, durations)
    image_paths = illustrator.illustrate(ctx, studio, script, refs, heroes)
    illustrator.quality_check(ctx, studio, script, refs, image_paths)
    image_paths = upscaler.upscale(ctx, image_paths)
    clips = animator.animate(ctx, image_paths, script["shots"], durations)

    # 7. original score
    score = None
    if cfg["music"].get("enabled", True):
        score = composer.compose(ctx, script["shots"], shot_starts, voice_len, v["lead_in_seconds"], total)

    # 8. edit
    end_card = ("The End", "Follow for more kahaniyan") if final else \
        ("To Be Continued", f"Follow For Part {series['part'] + 1}")
    overlay = work / "overlay.ass"
    speakers = ["N"] + [c["id"] for c in script["characters"]]
    editor.build_overlay(ctx, timeline, speakers, v["lead_in_seconds"], total, end_card, overlay)
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
        result = publisher.publish_reel(ctx, reel, cap)
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
