"""Critic agent: an independent editor that scores the script before it ships.

It checks hook strength, tension, twist, pacing, and safety. Scripts below the
threshold go back to the Writer with specific feedback.
"""
from __future__ import annotations

import json

from .common import SAFETY_RULES, Context, log
from .writer import word_count

NAME = "Critic"


def review(ctx: Context, script: dict, fmt: dict) -> dict:
    s = ctx.cfg["story"]
    words = word_count(script)
    prompt = f"""You are a ruthless editor for a horror Reels account aimed at Gen Z.
You did NOT write this script. Judge it as a scroller deciding in 2 seconds whether to keep watching.

Format: {fmt['label']} — {fmt['brief']}
Script ({words} words; target {s['min_words']}-{s['max_words']}):
{json.dumps(script, indent=2, ensure_ascii=False)}

Score each 0-10:
- hook: would a Gen Z viewer stop scrolling in the first line?
- tension: does every scene escalate? any dead lines?
- twist: does the ending land and recontextualise the story? is it set up fairly?
- spoken_flow: natural when read aloud by a narrator? short sentences?
- originality: fresh, not a tired creepypasta?
- visuals: are the stock footage searches realistic and atmospheric?

Safety check against these rules:
{SAFETY_RULES}
For the 'real mystery' format also check factual accuracy: flag anything invented or overstated.

Return JSON:
{{"scores": {{"hook":0,"tension":0,"twist":0,"spoken_flow":0,"originality":0,"visuals":0}},
  "safety_ok": true, "safety_issues": "", "facts_ok": true,
  "feedback": "specific, actionable notes for the writer (which line, what to change)"}}"""

    result = ctx.llm.json(prompt, temperature=0.3, mock={
        "scores": {"hook": 9, "tension": 8, "twist": 9, "spoken_flow": 8, "originality": 8, "visuals": 8},
        "safety_ok": True, "safety_issues": "", "facts_ok": True,
        "feedback": "Strong. Consider tightening scene 4.",
    })

    scores = result.get("scores", {})
    total = round(sum(scores.values()) / max(1, len(scores)) * 10) if scores else 0
    # Length outside the target range costs points (TTS length matters for Reels).
    if words < s["min_words"] - 15 or words > s["max_words"] + 20:
        total -= 15
        result["feedback"] = (result.get("feedback", "") +
                              f" LENGTH: script is {words} words; it must be {s['min_words']}-{s['max_words']}.").strip()
    result["score"] = total
    result["approved"] = (total >= s["min_critic_score"] and result.get("safety_ok", False)
                          and result.get("facts_ok", True))
    log(NAME, f"score {total}/100 · safety {'ok' if result.get('safety_ok') else 'FAIL'}"
              f" · {'APPROVED' if result['approved'] else 'needs revision'}")
    return result
