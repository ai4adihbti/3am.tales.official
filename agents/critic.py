"""Critic agent: an independent editor that scores the script before it ships.

Checks the hook, tension, twist, how natural the Hinglish sounds, whether the two characters
sound different, length, and safety. Scripts below the threshold go back to the Writer.
"""
from __future__ import annotations

import json

from .common import HINGLISH_GUIDE, SAFETY_RULES, Context, log
from .writer import estimate_seconds, line_count

NAME = "Critic"


def review(ctx: Context, script: dict, fmt: dict) -> dict:
    s = ctx.cfg["story"]
    secs, lines = estimate_seconds(script), line_count(script)
    prompt = f"""You are a ruthless editor for a Hindi horror reels account aimed at Indian Gen Z.
You did NOT write this script. Judge it as a scroller deciding in 2 seconds whether to keep watching.

Format: {fmt['label']} — {fmt['brief']}
Script ({lines} lines, about {secs:.0f} seconds; target {s['min_lines']}-{s['max_lines']} lines,
{s['min_seconds']}-{s['max_seconds']} seconds):
{json.dumps(script, indent=2, ensure_ascii=False)}

The language should be: {HINGLISH_GUIDE}

Score each 0-10:
- hook: would an Indian Gen Z viewer stop scrolling at the first line?
- tension: does every scene escalate? any dead lines?
- twist: does the ending land and recontextualise the story? is it set up fairly?
- hinglish: does it sound like real young Indians talking (Hindi-heavy, natural English mix), not textbook or filmy?
  Do "hi" and "roman" say the same thing? Is "hi" in Devanagari with English words in English letters?
- characters: do the two people sound clearly different? is it pure dialogue with no narrator?
- desi_feel: is the setting and detail authentically Indian and fresh?
- visuals: are the stock footage searches realistic and atmospheric?

Safety check against these rules:
{SAFETY_RULES}

Return JSON:
{{"scores": {{"hook":0,"tension":0,"twist":0,"hinglish":0,"characters":0,"desi_feel":0,"visuals":0}},
  "safety_ok": true, "safety_issues": "",
  "feedback": "specific, actionable notes for the writer (which line, what to change)"}}"""

    result = ctx.llm.json(prompt, temperature=0.3, mock={
        "scores": {"hook": 9, "tension": 8, "twist": 9, "hinglish": 8, "characters": 8, "desi_feel": 9, "visuals": 8},
        "safety_ok": True, "safety_issues": "",
        "feedback": "Strong. Consider one more beat of tension before the reveal.",
    })

    scores = result.get("scores", {})
    total = round(sum(scores.values()) / max(1, len(scores)) * 10) if scores else 0
    notes = []
    if not ctx.mock:
        if secs < s["min_seconds"] - 10 or secs > s["max_seconds"] + 15:
            total -= 15
            notes.append(f"LENGTH: about {secs:.0f}s; it must be {s['min_seconds']}-{s['max_seconds']}s "
                         f"(roughly {int(s['min_seconds'] * 2.3)}-{int(s['max_seconds'] * 2.3)} words).")
        if lines < s["min_lines"] - 3:
            total -= 10
            notes.append(f"LINES: only {lines}; write {s['min_lines']}-{s['max_lines']} dialogue lines.")
    if len(script.get("characters", [])) != 2:
        total -= 20
        notes.append("CHARACTERS: there must be exactly two characters, A and B.")
    if notes:
        result["feedback"] = (result.get("feedback", "") + " " + " ".join(notes)).strip()
    result["score"] = total
    result["approved"] = total >= s["min_critic_score"] and result.get("safety_ok", False)
    log(NAME, f"score {total}/100 · safety {'ok' if result.get('safety_ok') else 'FAIL'}"
              f" · {'APPROVED' if result['approved'] else 'needs revision'}")
    return result
