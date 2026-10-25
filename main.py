#!/usr/bin/env python3
"""Night Stories — the orchestrator.

Runs the agent team end to end:

  Analyst → Researcher → Writer ⇄ Critic → Voice → Visual Director → Editor → Caption → Publisher → Memory

Usage:
  python main.py              # full run: create and post today's Reel
  python main.py --dry-run    # create the Reel but don't post it (review it in out/)
  python main.py --mock       # offline test: no API keys needed, fake story and footage
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from agents import analyst, caption, critic, editor, publisher, researcher, visual_director, voice, writer
from agents.common import ROOT, LLM, Context, load_config, log
from agents.memory import Memory


def produce_script(ctx: Context, memory: Memory) -> tuple[dict, dict, dict, dict]:
    s = ctx.cfg["story"]
    fmt = researcher.choose_format(ctx, memory)
    rejected: list[str] = []
    for attempt in range(s["max_ideas_per_run"]):
        idea = researcher.find_idea(ctx, memory, fmt, avoid_extra=rejected)
        script = writer.write(ctx, idea, fmt)
        review = critic.review(ctx, script, fmt)
        rev = 0
        while not review["approved"] and rev < s["max_revisions"]:
            rev += 1
            script = writer.revise(ctx, script, review.get("feedback", "") + " " + review.get("safety_issues", ""), fmt)
            review = critic.review(ctx, script, fmt)
        if review["approved"]:
            return fmt, idea, script, review
        rejected.append(idea["title"])
        log("Orchestrator", f"idea rejected after {rev} revision(s); trying a new one")
    raise RuntimeError("No script passed the Critic today — nothing will be posted.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="build the Reel but do not post it")
    ap.add_argument("--mock", action="store_true", help="offline test with fake story and footage")
    args = ap.parse_args()
    dry = args.dry_run or args.mock or os.getenv("DRY_RUN", "").lower() in ("1", "true", "yes")

    cfg = load_config()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work = ROOT / "work" / stamp
    out_dir = ROOT / "out"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    ctx = Context(cfg=cfg, workdir=work, mock=args.mock, dry_run=dry)
    ctx.llm = LLM(cfg, mock=args.mock)
    memory = Memory()
    t0 = time.time()
    log("Orchestrator", f"run {stamp} · {'MOCK' if args.mock else 'DRY RUN' if dry else 'LIVE'}")

    if not args.mock:
        missing = [k for k in ("GEMINI_API_KEY",) if not os.getenv(k)]
        if not visual_director.active_provider():
            missing.append("PIXABAY_API_KEY")
        if not dry:
            missing += [k for k in ("IG_ACCESS_TOKEN", "IG_USER_ID") if not os.getenv(k)]
        if missing:
            log("Orchestrator", f"missing secrets: {', '.join(missing)} — see README.md")
            return 2

    # 1. learn from past posts
    analyst.run(ctx, memory)

    # 2-4. idea → script → critique loop
    fmt, idea, script, review = produce_script(ctx, memory)
    text = writer.narration_text(script)

    # 5. narration
    vlead, vtail = cfg["video"]["lead_in_seconds"], cfg["video"]["tail_seconds"]
    voice_wav, words, narr_len = voice.narrate(ctx, text)
    total = round(vlead + narr_len + vtail, 3)
    if total > 89:
        log("Orchestrator", f"warning: reel is {total:.0f}s; trimming to 89s")
        total = 89.0

    # 6. footage, cut to the narration
    durations = editor.scene_boundaries(script["scenes"], words, total, vlead)
    clips, video_ids = visual_director.gather(ctx, script["scenes"], durations, memory.used_video_ids())

    # 7. edit
    subs = work / "subs.ass"
    editor.build_subtitles(ctx, words, script["hook"], vlead, subs, script_text=text)
    reel = out_dir / f"reel-{stamp}.mp4"
    editor.render(ctx, clips, durations, voice_wav, subs, total, reel)

    # 8. caption
    cap = caption.write_caption(ctx, script, idea, fmt)

    post = {"title": idea["title"], "format": fmt["id"], "hook": script["hook"],
            "critic_score": review["score"], "video_ids": video_ids,
            "duration_s": total, "file": reel.name, "caption": cap}
    (out_dir / f"post-{stamp}.json").write_text(
        json.dumps({**post, "script": script, "idea": idea, "review": review}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    # 9. publish
    if dry:
        log("Orchestrator", f"dry run — reel saved to out/{reel.name} (not posted)")
    else:
        result = publisher.publish_reel(ctx, reel, cap)
        post.update(result)
        memory.add_post(post)
        memory.remember_videos(video_ids)

    # 10. remember (analyst updates are saved even on dry runs)
    if not args.mock:
        memory.save()
    shutil.rmtree(work, ignore_errors=True)
    log("Orchestrator", f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
