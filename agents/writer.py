"""Writer agent: turns an idea into a timed Reel script, and revises on feedback.

The script is split into scenes; each scene has its narration line(s) and the
stock-footage search terms the Visual Director will use.
"""
from __future__ import annotations

import json

from .common import SAFETY_RULES, Context, log

NAME = "Writer"

FORMAT_SPEC = """Return JSON:
{{
  "hook": "the opening line, max 12 words, must create instant curiosity or dread",
  "scenes": [
    {{"narration": "1-3 short spoken sentences", "visual": "2-4 word stock video search, e.g. 'night bus interior'"}}
  ],
  "ending": "the final line — the twist or chilling sting, max 14 words"
}}
Rules:
- 4 to 7 scenes. The FIRST scene's narration must start with the hook (verbatim).
  The LAST scene's narration must end with the ending line (verbatim).
- Total narration (all scenes) {min_w}-{max_w} words.
- Written to be SPOKEN: short sentences, present tense or close first person, no stage directions,
  no emojis, no hashtags, no 'Part 1', no 'Follow for more' (the caption handles that).
- Visual searches must be things that exist as stock footage (places, objects, weather, light) —
  never specific people's faces or impossible creatures."""


def _mock_script() -> dict:
    return {
        "hook": "There were eleven people on my bus. Then there were twelve.",
        "scenes": [
            {"narration": "There were eleven people on my bus. Then there were twelve. I know, because I count. I always count on the last bus home.",
             "visual": "night bus interior"},
            {"narration": "At the next stop the doors opened, nobody got on, and the number went up again. Thirteen.",
             "visual": "bus doors opening night"},
            {"narration": "I looked at the driver's mirror. It showed every seat. And every seat was full. Even the one next to me.",
             "visual": "empty bus seats"},
            {"narration": "My stop came. I stood up. The doors stayed shut. The driver just looked at me in the mirror and said, quietly,",
             "visual": "city lights rain window"},
            {"narration": "you can't get off twice. You got off last night. Remember?",
             "visual": "dark road night"},
        ],
        "ending": "You got off last night. Remember?",
    }


def write(ctx: Context, idea: dict, fmt: dict) -> dict:
    s = ctx.cfg["story"]
    prompt = f"""You write viral horror Reels scripts for "{ctx.cfg['channel']['name']}" (audience: {ctx.cfg['channel']['audience']}).
Format: {fmt['label']} — {fmt['brief']}

Idea:
{json.dumps(idea, indent=2, ensure_ascii=False)}

Craft notes: the first 2 seconds decide everything, so open mid-moment. Escalate every scene.
Plant a detail early that pays off in the twist. End on the twist with no explanation afterwards.

{SAFETY_RULES}

{FORMAT_SPEC.format(min_w=s['min_words'], max_w=s['max_words'])}"""
    script = ctx.llm.json(prompt, temperature=0.95, mock=_mock_script())
    log(NAME, f"draft written ({word_count(script)} words, {len(script['scenes'])} scenes)")
    return script


def revise(ctx: Context, script: dict, feedback: str, fmt: dict) -> dict:
    s = ctx.cfg["story"]
    prompt = f"""Revise this horror Reel script using the editor's feedback. Keep what works.

Current script:
{json.dumps(script, indent=2, ensure_ascii=False)}

Editor feedback:
{feedback}

Format: {fmt['label']} — {fmt['brief']}

{SAFETY_RULES}

{FORMAT_SPEC.format(min_w=s['min_words'], max_w=s['max_words'])}"""
    new = ctx.llm.json(prompt, temperature=0.8, mock=script)
    log(NAME, f"revision written ({word_count(new)} words)")
    return new


def narration_text(script: dict) -> str:
    return " ".join(sc["narration"].strip() for sc in script["scenes"])


def word_count(script: dict) -> int:
    return len(narration_text(script).split())
