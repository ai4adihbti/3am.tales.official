"""Caption agent: writes the Hinglish Instagram caption and hashtags for each part."""
from __future__ import annotations

import json

from .common import Context, log

NAME = "Caption"


def write_caption(ctx: Context, script: dict, series: dict, fmt: dict) -> str:
    c = ctx.cfg["caption"]
    ch = ctx.cfg["channel"]
    final = series["part"] >= series["total_parts"]
    multi = int(series.get("total_parts") or 1) > 1
    part_tag = f"Part {series['part']}/{series['total_parts']}" if multi else ""
    title_part = f"{series['title']} | {part_tag}" if multi else series["title"]
    share = ch.get("share_line", "Ye reel us dost ko bhejo jo kehta hai bhoot nahi hote 👀")
    follow = ("Agla part jaldi 🕯️ Follow karo" if not final else ch.get("follow_line", "Roz 1 PM aur 9 PM nayi darawni kahani 🕯️ Follow karo"))
    brief = {"story": series["title"], "part": part_tag or "complete story", "summary": script.get("summary"),
             "characters": [x.get("name") for x in script.get("characters", [])]}
    prompt = f"""Write the Instagram caption for an illustrated Hindi horror story reel on "{ch['name']}"
(audience: {ch['audience']}).

Format: {fmt['label']}
Story brief (do NOT reveal the {'twist' if final else 'cliffhanger'}):
{json.dumps(brief, indent=1, ensure_ascii=False)}

Caption rules (Roman Hinglish, the way Indian Gen Z writes):
- Line 1 (people SEARCH these words on Instagram): a short curiosity hook about this story, then
  " | Bhoot ki kahani | Hindi horror story". Max 110 characters. Do not write "Part 1/1" or "(fiction)".
- Line 2: one question that makes people comment, about a choice the viewer would make
  (e.g. "Tum hote toh darwaza kholte ya nahi? 👀").
- 0-2 emojis total, no cringe.
- Then {c['max_hashtags'] - len(c['core_hashtags'])} niche hashtags for this story (mix Hindi-English, e.g. #qabristan #bhootiyahaveli).

Return JSON: {{"line1": "...", "line2": "...", "hashtags": ["#..."]}}"""

    plain = {"line1": f"{title_part} | Bhoot ki kahani | Hindi horror story",
             "line2": "Tum hote toh kya karte? 👀",
             "hashtags": ["#bhootkikahani", "#desihorror", "#raatkikahani", "#darawnikahani"]}
    try:
        r = _ask(ctx, prompt, ch, series, part_tag or "complete story")
    except Exception as e:  # noqa: BLE001 — the reel is finished; a plain caption beats not posting
        log(NAME, f"the caption writer is unavailable ({str(e).splitlines()[0][:100]}); using a simple caption")
        r = plain
    if not isinstance(r, dict) or not str(r.get("line1") or "").strip():
        r = plain
    if not isinstance(r.get("hashtags"), list):
        r["hashtags"] = plain["hashtags"]

    tags: list[str] = []
    for t in list(c["core_hashtags"]) + [str(x) for x in r.get("hashtags", [])]:
        t = "#" + t.lstrip("#").replace(" ", "").lower()
        if len(t) > 1 and t not in tags and t != "#aistory":
            tags.append(t)
    tags = tags[: c["max_hashtags"]]
    line1 = str(r["line1"]).strip().replace("(fiction)", "").replace("Part 1/1", "").strip(" |")
    if multi and part_tag not in line1:
        line1 = f"{part_tag} · {line1}"
    caption = (f"{line1}\n\n{str(r.get('line2') or plain['line2']).strip()}\n\n{share}\n{follow}\n\n"
               f"(kalpanik kahani)\n.\n.\n{' '.join(tags)}")
    log(NAME, f"caption ready ({len(tags)} hashtags)")
    return caption[:2200]


def _ask(ctx: Context, prompt: str, ch: dict, series: dict, part_tag: str) -> dict:
    return ctx.llm.json(prompt, temperature=0.8, mock={
        "line1": f"{series['title']} | {part_tag} — qabristan ka raaz (fiction)",
        "line2": "Tum hote toh andar jaate ya wapas bhaagte? 👀",
        "line3": ch["cta"],
        "hashtags": ["#qabristan", "#horrorseries", "#raatkikahani", "#desihorror"],
    })
