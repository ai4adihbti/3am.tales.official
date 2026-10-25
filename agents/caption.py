"""Caption agent: writes the Instagram caption and hashtags."""
from __future__ import annotations

import json

from .common import Context, log

NAME = "Caption"


def write_caption(ctx: Context, script: dict, idea: dict, fmt: dict) -> str:
    c = ctx.cfg["caption"]
    ch = ctx.cfg["channel"]
    prompt = f"""Write the Instagram caption for this horror Reel on "{ch['name']}" (audience: {ch['audience']}).

Format: {fmt['label']}
Script:
{json.dumps(script, indent=2, ensure_ascii=False)}

Caption rules:
- Line 1: a curiosity line that is NOT the same as the hook and does NOT spoil the twist (max 90 chars).
- Line 2: one question that makes people comment (e.g. "Would you have stayed on the bus?").
- Line 3: exactly this call to action: {ch['cta']}
- {"Add '(fiction)' at the end of line 1." if fmt['id'] != 'unsettling_fact' else "Do not invent facts."}
- 0-2 emojis total, lowercase-friendly Gen Z tone, no cringe slang.
- Then {c['max_hashtags'] - len(c['core_hashtags'])} niche hashtags specific to this story's setting/theme.

Return JSON: {{"line1": "...", "line2": "...", "line3": "...", "hashtags": ["#..."]}}"""

    r = ctx.llm.json(prompt, temperature=0.8, mock={
        "line1": "the last bus home always has one more passenger (fiction)",
        "line2": "would you have counted? 👀",
        "line3": ch["cta"],
        "hashtags": ["#nightbus", "#creepypasta", "#twistending", "#horrortok"],
    })

    tags: list[str] = []
    for t in list(c["core_hashtags"]) + list(r.get("hashtags", [])):
        t = "#" + t.lstrip("#").replace(" ", "").lower()
        if len(t) > 1 and t not in tags:
            tags.append(t)
    tags = tags[: c["max_hashtags"]]

    caption = f"{r['line1'].strip()}\n\n{r['line2'].strip()}\n\n{r.get('line3') or ch['cta']}\n.\n.\n{' '.join(tags)}"
    caption = caption[:2200]
    log(NAME, f"caption ready ({len(tags)} hashtags)")
    return caption
