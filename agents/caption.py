"""Caption agent: writes the Hinglish Instagram caption and hashtags for each part."""
from __future__ import annotations

import json

from .common import Context, log

NAME = "Caption"


def write_caption(ctx: Context, script: dict, series: dict, fmt: dict) -> str:
    c = ctx.cfg["caption"]
    ch = ctx.cfg["channel"]
    final = series["part"] >= series["total_parts"]
    part_tag = f"Part {series['part']}/{series['total_parts']}"
    brief = {"series": series["title"], "part": part_tag, "summary": script.get("summary"),
             "characters": [x.get("name") for x in script.get("characters", [])]}
    prompt = f"""Write the Instagram caption for an illustrated Hindi horror story reel on "{ch['name']}"
(audience: {ch['audience']}).

Format: {fmt['label']}
Story brief (do NOT reveal the {'twist' if final else 'cliffhanger'}):
{json.dumps(brief, indent=1, ensure_ascii=False)}

Caption rules (Roman Hinglish, the way Indian Gen Z writes):
- Line 1: "{series['title']} | {part_tag}" followed by a short curiosity hook (max 90 chars total). End with "(fiction)".
- Line 2: one question that makes people comment, e.g. "Tum hote toh qabristan jaate ya nahi? 👀".
- Line 3: {"'Poori kahani khatam… agli kahani kal raat 🕯️ Follow karo'" if final else f"exactly this call to action: {ch['cta']}"}
- 0-2 emojis total, no cringe.
- Then {c['max_hashtags'] - len(c['core_hashtags'])} niche hashtags for this story (mix Hindi-English, e.g. #qabristan #horrorseries).

Return JSON: {{"line1": "...", "line2": "...", "line3": "...", "hashtags": ["#..."]}}"""

    r = ctx.llm.json(prompt, temperature=0.8, mock={
        "line1": f"{series['title']} | {part_tag} — qabristan ka raaz (fiction)",
        "line2": "Tum hote toh andar jaate ya wapas bhaagte? 👀",
        "line3": ch["cta"],
        "hashtags": ["#qabristan", "#horrorseries", "#raatkikahani", "#desihorror"],
    })

    tags: list[str] = []
    for t in list(c["core_hashtags"]) + list(r.get("hashtags", [])):
        t = "#" + t.lstrip("#").replace(" ", "").lower()
        if len(t) > 1 and t not in tags:
            tags.append(t)
    tags = tags[: c["max_hashtags"]]
    caption = f"{r['line1'].strip()}\n\n{r['line2'].strip()}\n\n{(r.get('line3') or ch['cta']).strip()}\n.\n.\n{' '.join(tags)}"
    log(NAME, f"caption ready ({len(tags)} hashtags)")
    return caption[:2200]
