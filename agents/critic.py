"""Critic agent: an independent editor that scores each part before it's produced.

Checks hook, emotional pull, tension, cliffhanger/twist, natural Hindi, distinct characters, visual
storytelling (fast cuts, shot/reverse-shot), length and safety. Weak scripts go back to the Writer.
"""
from __future__ import annotations

import json

from .common import HINGLISH_GUIDE, SAFETY_RULES, Context, log
from .writer import all_lines, craft, estimate_seconds, has_cold_open, to_screenplay, voice_rule

NAME = "Critic"


def review(ctx: Context, script: dict, series: dict, fmt: dict) -> dict:
    s = ctx.cfg["story"]
    secs = estimate_seconds(ctx, script)
    cuts = len(script["shots"])
    images = len(script["images"])
    lines = all_lines(script)
    narr = sum(1 for ln in lines if ln["speaker"] == "N")
    final = series["part"] >= series["total_parts"]
    prompt = f"""You are a ruthless editor for an illustrated Hindi story series on Instagram (Indian audience, 16-30).
You did NOT write this. Judge it as a viewer deciding in 2 seconds whether to keep watching.

Series "{series['title']}", part {series['part']} of {series['total_parts']} ({'final part' if final else 'must end on a cliffhanger'}).
Format: {fmt['label']} — {fmt['brief']}
Screenplay: {cuts} cuts, {images} images, {len(lines)} lines ({narr} narrator), about {secs:.0f} seconds
(target {s['min_cuts']}-{s['max_cuts']} cuts, {s['min_seconds']}-{s['max_seconds']} seconds):
{json.dumps(to_screenplay(script), indent=1, ensure_ascii=False)}

{craft(series)}

The language should be: {HINGLISH_GUIDE}

Score each 0-10:
- hook: would a viewer stop scrolling in the first 3 seconds? (cut 1 must be a scary close-up flash-forward with a
  short shocking line, cut 2 a turn back in time; a calm scenic opening scores at most 4)
- emotion: do we care about these people? is there a real bond/feeling at the centre?
- tension: does it escalate cut by cut through small, concrete details? any dead lines?
- ending: {'does the twist land emotionally and pay off earlier parts?' if final else 'is the cliffhanger strong enough to make people follow for the next part?'}
- dialogue: natural spoken Hindi for each character's age? short one-breath lines? distinct voices?
  does it follow this voice style: {voice_rule(series)} "hi" Devanagari + English words in English letters?
- visuals: drawable, varied framing, vivid image descriptions (light, expression, props), a recurring motif shot,
  emotional close-ups, clue inserts, no text in images?
- desi_feel: authentically Indian, fresh setting and details?

Safety check against these rules:
{SAFETY_RULES}

Return JSON:
{{"scores": {{"hook":0,"emotion":0,"tension":0,"ending":0,"dialogue":0,"visuals":0,"desi_feel":0}},
  "safety_ok": true, "safety_issues": "",
  "feedback": "specific, actionable notes for the writer (which cut, what to change)"}}"""

    result = ctx.llm.json(prompt, temperature=0.3, mock={
        "scores": {"hook": 9, "emotion": 9, "tension": 8, "ending": 9, "dialogue": 8, "visuals": 9, "desi_feel": 9},
        "safety_ok": True, "safety_issues": "", "feedback": "Strong."})

    scores = result.get("scores", {})
    total = round(sum(scores.values()) / max(1, len(scores)) * 10) if scores else 0
    notes = []
    if not ctx.mock:
        if secs < s["min_seconds"] - 15 or secs > s["max_seconds"] + 10:
            total -= 12
            notes.append(f"LENGTH: about {secs:.0f}s; it must be {s['min_seconds']}-{s['max_seconds']}s "
                         f"(roughly {int(s['min_seconds'] * 2.0)}-{int(s['max_seconds'] * 2.0)} spoken words).")
        if cuts < s["min_cuts"] - 6:
            total -= 10
            notes.append(f"CUTS: only {cuts}; write {s['min_cuts']}-{s['max_cuts']} cuts of ~2-3 s each, one line per cut, "
                         "cutting between speaker and listener close-ups.")
        shots = script["shots"]
        if not has_cold_open(shots):
            total -= 5
            notes.append("COLD OPEN: cut 1 must be a close-up flash-forward of the scariest moment with ONE gripping "
                         "line of at most 10 words; the calm motif shot comes only after cut 2.")
        longest = max((len(ln["roman"].split()) for ln in lines), default=0)
        if longest > 24:
            # feedback only (no penalty): the editor splits long lines anyway, and a missed post is worse
            notes.append(f"LONG LINES: a line has {longest} words; keep every line under 22 words (split it).")
        uses: dict[str, int] = {}
        for sh in shots:
            uses[sh["image_id"]] = uses.get(sh["image_id"], 0) + 1
        repeats = sum(1 for a, b in zip(shots, shots[1:]) if a["image_id"] == b["image_id"])
        if max(uses.values(), default=0) > 4 or repeats > 2:
            # feedback only (no penalty): the editor splits long lines anyway, and a missed post is worse
            notes.append("REPEATS: an image is reused too often or twice in a row; use new inserts "
                         "(hands, eyes, lantern, a clue) instead.")
        ratio = narr / len(lines) if lines else 0
        if series.get("voice_style") == "narrator":
            if lines and not 0.78 <= ratio <= 0.98:
                total -= 8
                notes.append("STYLE: this series is told by a storyteller: make 85-95% of lines narration, "
                             "with only 2-5 short key character lines.")
        elif lines and ratio > 0.25:
            total -= 8
            notes.append("NARRATOR: too much narration; turn it into character dialogue.")
    if not script["shots"]:
        total = 0
        notes.append("EMPTY: no usable cuts were returned.")
    if notes:
        result["feedback"] = (result.get("feedback", "") + " " + " ".join(notes)).strip()
    result["score"] = total
    result["approved"] = total >= s["min_critic_score"] and result.get("safety_ok", False)
    log(NAME, f"score {total}/100 · safety {'ok' if result.get('safety_ok') else 'FAIL'}"
              f" · {'APPROVED' if result['approved'] else 'needs revision'}")
    return result
