"""Caption agent: writes the Hinglish Instagram caption and hashtags."""
from __future__ import annotations

import json

from .common import Context, log

NAME = "Caption"


def write_caption(ctx: Context, script: dict, idea: dict, fmt: dict) -> str:
    c = ctx.cfg["caption"]
    ch = ctx.cfg["channel"]
    brief = {"title_card": script.get("title_card"), "hook": script.get("hook"),
             "characters": script.get("characters"), "idea": idea}
    prompt = f"""Write the Instagram caption for this Hindi horror reel on "{ch['name']}" (audience: {ch['audience']}).

Format: {fmt['label']}
Story brief (do NOT reveal the twist):
{json.dumps(brief, indent=2, ensure_ascii=False)}

Caption rules (Roman Hinglish, the way Indian Gen Z writes):
- Line 1: a curiosity line that is NOT the hook and does NOT spoil the twist (max 90 chars). End it with "(fiction)".
- Line 2: one question that makes people comment, e.g. "Tum hote toh phone kaat dete ya nahi? 👀".
- Line 3: exactly this call to action: {ch['cta']}
- 0-2 emojis total, no cringe.
- Then {c['max_hashtags'] - len(c['core_hashtags'])} niche hashtags for this story (mix Hindi-English, e.g. #pgstories #trainhorror).

Return JSON: {{"line1": "...", "line2": "...", "line3": "...", "hashtags": ["#..."]}}"""

    r = ctx.llm.json(prompt, temperature=0.8, mock={
        "line1": "PG ka woh teesra bed jispe koi nahi sota… (fiction)",
        "line2": "Tum hote toh light aane ka wait karte ya bhaag jaate? 👀",
        "line3": ch["cta"],
        "hashtags": ["#pgstories", "#punehorror", "#raatkikahani", "#desihorror"],
    })

    tags: list[str] = []
    for t in list(c["core_hashtags"]) + list(r.get("hashtags", [])):
        t = "#" + t.lstrip("#").replace(" ", "").lower()
        if len(t) > 1 and t not in tags:
            tags.append(t)
    tags = tags[: c["max_hashtags"]]

    caption = f"{r['line1'].strip()}\n\n{r['line2'].strip()}\n\n{r.get('line3') or ch['cta']}\n.\n.\n{' '.join(tags)}"
    log(NAME, f"caption ready ({len(tags)} hashtags)")
    return caption[:2200]
