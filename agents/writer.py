"""Writer agent: turns an idea into a two-character Hinglish dialogue script, and revises it.

Each line carries:
- `hi`    : the line written for the voice actor — Devanagari Hindi with English words left in
            English letters (how Hindi TTS voices pronounce Hinglish best)
- `roman` : the same line in Roman Hinglish, used for on-screen subtitles
- `style` : a short delivery note for the voice (whispering, panicked, flat and cold …)
Scenes group lines and carry the stock-footage search for that moment.
"""
from __future__ import annotations

import json
import re

from .common import HINGLISH_GUIDE, SAFETY_RULES, Context, log

NAME = "Writer"

VOICE_TYPES = ["young_male", "adult_male", "old_male", "young_female", "adult_female", "old_female"]

FORMAT_SPEC = """Return JSON:
{{
  "title_card": "2-6 word on-screen setting shown in the first seconds, Roman script, e.g. 'Pune PG · 3:07 AM'",
  "characters": [
    {{"id": "A", "name": "Indian first name", "voice_type": one of {types},
      "is_entity": false, "description": "age, vibe, how they talk"}},
    {{"id": "B", ...}}
  ],
  "scenes": [
    {{"visual": "2-4 word stock video search in English, e.g. 'dark hostel corridor'",
      "lines": [
        {{"speaker": "A", "hi": "line for the voice actor", "roman": "same line in Roman Hinglish",
          "style": "2-5 word delivery note in English"}}
      ]}}
  ],
  "hook": "copy of the very first line's roman text",
  "ending": "copy of the final line's roman text"
}}
Rules:
- EXACTLY two characters, ids "A" and "B". DIALOGUE ONLY — no narrator, no stage directions in lines.
  If one of them turns out to be a spirit/possessed, set "is_entity": true for them (the voice gets an eerie effect
  ONLY if the twist is revealed from the start; otherwise keep it false so the twist isn't spoiled).
- {min_lines}-{max_lines} lines in total across 6-10 scenes; 2-6 lines per scene. Scenes change where the
  location, the tension or what we'd see changes.
- Total spoken length {min_s}-{max_s} seconds: roughly {min_w}-{max_w} words in total.
- "hi": Devanagari for Hindi words, English words in English letters (e.g. "यार, मेरा phone की battery
  dead होने वाली है"). Numbers as words. You may add at most one vocal tag per line, in English angle
  brackets, only from: <whisper>, <sigh>, <gasp>, <short pause>, <laugh>, <long pause>.
- "roman": the same line in natural Roman Hinglish (no tags), e.g. "Yaar, mera phone ki battery dead hone wali hai".
- The FIRST line must hook in under 3 seconds (a strange question, a sound, a panic). The LAST line is the twist,
  with nothing after it.
- "visual": must exist as real stock footage: places, objects, light, weather — e.g. 'india village night',
  'train window night', 'old haveli corridor', 'dark staircase', 'phone screen dark room'. Never a specific
  person's face or a creature."""


def _fmt(ctx: Context) -> dict:
    s = ctx.cfg["story"]
    return dict(types=" | ".join(f'"{t}"' for t in VOICE_TYPES), min_lines=s["min_lines"],
                max_lines=s["max_lines"], min_s=s["min_seconds"], max_s=s["max_seconds"],
                min_w=int(s["min_seconds"] * 2.3), max_w=int(s["max_seconds"] * 2.3))


def _mock_script() -> dict:
    A, B = "A", "B"
    scenes = [
        ("dark room night", [
            (A, "तन्वी… तू जाग रही है? <whisper>", "Tanvi… tu jaag rahi hai?", "whispering, scared"),
            (B, "हम्म… क्या हुआ? तीन बज रहे हैं यार।", "Hmm… kya hua? Teen baj rahe hain yaar.", "sleepy, annoyed"),
            (A, "Light गई हुई है, और मुझे लग रहा है कोई तीसरे bed पे साँस ले रहा है।",
             "Light gayi hui hai, aur mujhe lag raha hai koi teesre bed pe saans le raha hai.", "tense, hushed"),
        ]),
        ("empty bed dark", [
            (B, "पागल है क्या? वो bed तो खाली है, एक साल से।", "Pagal hai kya? Woh bed toh khaali hai, ek saal se.", "dismissive"),
            (A, "मुझे पता है… इसीलिए तो call किया।", "Mujhe pata hai… isiliye toh call kiya.", "shaky"),
        ]),
        ("phone screen dark room", [
            (B, "अच्छा सुन, तूने वो नीली hoodie पहनी है ना?", "Achha sun, tune woh neeli hoodie pehni hai na?", "calm, slow"),
            (A, "हाँ… पर तुझे कैसे पता? यहाँ तो घुप अंधेरा है।", "Haan… par tujhe kaise pata? Yahaan toh ghup andhera hai.", "confused, rising fear"),
        ]),
        ("dark corridor", [
            (B, "<short pause> क्योंकि मैं तुझे देख रही हूँ।", "Kyunki main tujhe dekh rahi hoon.", "flat, cold"),
            (A, "तू… तू तो पिछले हफ्ते घर चली गई थी ना?", "Tu… tu toh pichhle hafte ghar chali gayi thi na?", "terrified realisation"),
            (B, "हाँ। <whisper> तो फिर तीसरे bed पे कौन लेटा है?", "Haan. Toh phir teesre bed pe kaun leta hai?", "whispering, chilling"),
        ]),
    ]
    return {
        "title_card": "Pune PG · 3:07 AM",
        "characters": [
            {"id": "A", "name": "Ishita", "voice_type": "young_female", "is_entity": False, "description": "20, sleepy, scared"},
            {"id": "B", "name": "Tanvi", "voice_type": "young_female", "is_entity": False, "description": "21, calm, too calm"},
        ],
        "scenes": [{"visual": v, "lines": [{"speaker": s, "hi": h, "roman": r, "style": st} for s, h, r, st in ls]}
                   for v, ls in scenes],
        "hook": "Tanvi… tu jaag rahi hai?",
        "ending": "Haan. Toh phir teesre bed pe kaun leta hai?",
    }


def _clean(script: dict) -> dict:
    """Normalise whatever the model returned so later agents can rely on the shape."""
    chars = script.get("characters") or []
    ids = [c.get("id") for c in chars]
    if ids[:2] != ["A", "B"]:
        for c, i in zip(chars, ["A", "B"]):
            c["id"] = i
    for c in chars:
        if c.get("voice_type") not in VOICE_TYPES:
            c["voice_type"] = "young_male"
        c["is_entity"] = bool(c.get("is_entity"))
    scenes = []
    for sc in script.get("scenes", []):
        lines = [ln for ln in sc.get("lines", []) if (ln.get("hi") or "").strip() and ln.get("speaker") in ("A", "B")]
        for ln in lines:
            ln["roman"] = (ln.get("roman") or "").strip() or strip_tags(ln["hi"])
            ln["style"] = (ln.get("style") or "").strip()[:60]
        if lines:
            scenes.append({"visual": (sc.get("visual") or "dark room night").strip(), "lines": lines})
    script["scenes"] = scenes
    script["characters"] = chars[:2]
    if scenes:
        script["hook"] = scenes[0]["lines"][0]["roman"]
        script["ending"] = scenes[-1]["lines"][-1]["roman"]
    script.setdefault("title_card", "")
    return script


def write(ctx: Context, idea: dict, fmt: dict) -> dict:
    prompt = f"""You write viral Hindi horror reels for "{ctx.cfg['channel']['name']}" (audience: {ctx.cfg['channel']['audience']}).
Format: {fmt['label']} — {fmt['brief']}

Idea:
{json.dumps(idea, indent=2, ensure_ascii=False)}

{HINGLISH_GUIDE}

Craft notes: open mid-moment. Every scene raises the stakes. The two characters must sound different
(one panics, one is too calm; one jokes to cope, one goes quiet). Plant a small detail early that pays off
in the twist. Let silence and reactions do work ("…", "kya?", "ruk"). End on the twist, no explanation.

{SAFETY_RULES}

{FORMAT_SPEC.format(**_fmt(ctx))}"""
    script = _clean(ctx.llm.json(prompt, temperature=0.95, mock=_mock_script()))
    log(NAME, f"draft: {line_count(script)} lines, {len(script['scenes'])} scenes, ~{estimate_seconds(script):.0f}s")
    return script


def revise(ctx: Context, script: dict, feedback: str, fmt: dict) -> dict:
    prompt = f"""Revise this Hindi horror dialogue reel using the editor's feedback. Keep what works.

Current script:
{json.dumps(script, indent=2, ensure_ascii=False)}

Editor feedback:
{feedback}

Format: {fmt['label']} — {fmt['brief']}

{HINGLISH_GUIDE}

{SAFETY_RULES}

{FORMAT_SPEC.format(**_fmt(ctx))}"""
    new = _clean(ctx.llm.json(prompt, temperature=0.8, mock=script))
    log(NAME, f"revision: {line_count(new)} lines, ~{estimate_seconds(new):.0f}s")
    return new


# ── helpers used by other agents ────────────────────────────────────────
def strip_tags(text: str) -> str:
    return re.sub(r"\s*<[^>]+>\s*", " ", text).strip()


def all_lines(script: dict) -> list[dict]:
    return [ln for sc in script["scenes"] for ln in sc["lines"]]


def line_count(script: dict) -> int:
    return len(all_lines(script))


def word_count(script: dict) -> int:
    return sum(len(ln["roman"].split()) for ln in all_lines(script))


def estimate_seconds(script: dict) -> float:
    # Hindi dialogue with pauses runs at roughly 2.3 words per second.
    return word_count(script) / 2.3 + 0.35 * line_count(script)


def narration_text(script: dict) -> str:
    return " ".join(ln["roman"] for ln in all_lines(script))
