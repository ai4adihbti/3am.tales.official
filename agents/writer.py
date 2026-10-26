"""Writer agent: writes one PART of an illustrated Hindi story series as a fast-cut screenplay.

The script has two lists:
- "images": the unique illustrations to draw (each one a framing of a moment: a close-up of one
  character, a two-shot, a wide establishing shot, an object insert …)
- "cuts":   what's on screen, in order, ~2-3 s each. Every cut points to an image and carries at most
  one spoken line. Close-ups are reused for back-and-forth dialogue (shot / reverse shot), exactly like
  the reference reels, so a 2-minute part needs ~30 drawings for ~50 cuts.

Speakers: "N" = narrator (sparingly), "C1".."C4" = the characters. Each line has:
- `hi`    : for the voice actor: Devanagari Hindi with English words in English letters
- `roman` : the same line in Roman Hinglish (for subtitles if enabled, and timing)
- `style` : a short delivery note (whispering, crying softly, flat and cold …)

Internally the cuts are normalised into `shots` (one per cut) that carry the image's description,
so the rest of the pipeline treats every cut the same way.
"""
from __future__ import annotations

import json
import re

from .common import HINGLISH_GUIDE, SAFETY_RULES, Context, log

NAME = "Writer"

VOICE_TYPES = ["child_boy", "child_girl", "young_male", "adult_male", "old_male",
               "young_female", "adult_female", "old_female"]
FRAMINGS = ["establishing", "wide", "two_shot", "medium", "closeup", "extreme_closeup", "object", "over_shoulder"]
MOTIONS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down", "push_fast", "hold"]
MOODS = ["calm", "uneasy", "tense", "scare", "sad"]

CRAFT = """What makes these reels work (study this):
- EMOTION FIRST. The audience must care about the people within 10 seconds: a tired mother protecting her child,
  a father who promised to come home, friends who tease each other. Horror hits harder when we love them.
- MYSTERY THROUGH SMALL DETAILS. Strange things are noticed in dialogue ("Maa, woh aunty ke pair ulte kyun the?"),
  never explained by a narrator. Viewers piece it together.
- SHORT SPOKEN LINES. Each line is one breath (4-14 words). Real reactions: "Kya?", "Ruk…", "Maa…", sighs, pauses.
- SHOT / REVERSE SHOT. Dialogue cuts between close-ups of the speaker and the listener's reaction.
  Open on an establishing shot, use wide/two-shots to show where they are, inserts for clues.
- A TWIST WITH A FEELING. The reveal should make people go "rula diya" or "rongte khade ho gaye", ideally both."""

CHARACTER_SPEC = """"characters": [
    {{"id": "C1", "name": "Indian first name", "gender": "male|female", "age": 30,
      "voice_type": one of {types},
      "look": "FIXED design used in every illustration: build, face shape, skin tone, hairstyle, facial hair,
               jewellery (bindi, mangalsutra, kada…) and one exact outfit with colours
               (e.g. 'maroon cotton saree with beige blouse, black hair in a low bun, small red bindi, glass bangles')",
      "personality": "how they talk and react",
      "is_entity": false}}
  ],"""

BODY_SPEC = """"images": [
    {{"id": "I1", "framing": one of {framings},
      "characters": ["C1"],
      "image": "what this illustration shows: setting, lighting, pose, expression, props. Refer to characters by NAME
                (their designs are added automatically). No readable text, signs or writing in the picture.",
      "night": true}}
  ],
  "cuts": [
    {{"image": "I1", "motion": one of {motions}, "mood": one of {moods},
      "line": {{"speaker": "C1" or "N", "hi": "…", "roman": "…", "style": "2-5 word delivery note"}} or null}}
  ],"""

RULES = """Rules:
- {min_cuts}-{max_cuts} cuts, at most {max_images} images. A cut lasts as long as its line (~2-3 s) or ~1.8 s if
  silent (use 3-6 silent cuts for suspense: a reaction, an empty road, a clue). Reuse close-up images for
  back-and-forth dialogue; every image must be used at least once.
- Total spoken length {min_s}-{max_s} seconds: roughly {min_w}-{max_w} words in total.
- {voice_rule}
- Images: mostly single-character close-ups and medium shots (they draw best), plus establishing, two-shots and
  object inserts. "characters" lists only who is VISIBLE (max 3). Children are drawn as clearly young kids.
- "hi": Devanagari for Hindi words, English words in English letters (e.g. "माँ, phone में network नहीं है").
  Numbers as words. At most one vocal tag per line, only from: <whisper>, <sigh>, <gasp>, <short pause>,
  <laugh>, <long pause>, <crying>, in English angle brackets.
- "roman": the same line in natural Roman Hinglish, no tags.
- The first 2 cuts must hook in under 4 seconds. {ending_rule}"""


DIALOGUE_RULE = ("At least 80% of lines are CHARACTER dialogue. The narrator \"N\" only opens the part (one or two "
                 "lines), bridges a time jump, or delivers the final sting.")
NARRATOR_RULE = ("STORYTELLER STYLE: one {g} narrator \"N\" tells the story like a gripping late-night Hindi horror "
                 "kahani — intimate, vivid, in short spoken sentences, describing what the characters feel, see and hear. "
                 "70-85% of lines are narration (one sentence per cut). Characters speak only short, key lines at the "
                 "most dramatic moments (a scream, a warning, a whispered question). Keep cutting between close-ups, "
                 "reactions and clue inserts so the picture changes every ~2-3 s.")


def voice_rule(series: dict) -> str:
    if series.get("voice_style") == "narrator":
        g = "female" if series.get("narrator_gender") == "female" else "male"
        return NARRATOR_RULE.format(g=g)
    return DIALOGUE_RULE


def _fmt(ctx: Context, ending_rule: str = "", series: dict | None = None) -> dict:
    s = ctx.cfg["story"]
    return dict(types=" | ".join(f'"{t}"' for t in VOICE_TYPES),
                framings=" | ".join(f'"{f}"' for f in FRAMINGS),
                motions=" | ".join(f'"{m}"' for m in MOTIONS),
                moods=" | ".join(f'"{m}"' for m in MOODS),
                min_cuts=s["min_cuts"], max_cuts=s["max_cuts"], max_images=s["max_images"],
                min_s=s["min_seconds"], max_s=s["max_seconds"],
                min_w=int(s["min_seconds"] * 2.0), max_w=int(s["max_seconds"] * 2.0),
                ending_rule=ending_rule, voice_rule=voice_rule(series or {}))


def _ending_rule(series: dict) -> str:
    if series["part"] < series["total_parts"]:
        return ("This is NOT the last part: end on a sharp CLIFFHANGER (a reveal, a figure, a voice, a terrible "
                "realisation) with nothing resolved. 'To Be Continued' is added automatically, don't write it.")
    return ("This is the FINAL part: resolve the mystery with a twist that recontextualises earlier parts and lands "
            "emotionally, then one last line that lingers. 'The End' is added automatically, don't write it.")


def _series_block(series: dict) -> str:
    out = [f'Series: "{series["title"]}" — this is PART {series["part"]} of {series["total_parts"]}.']
    if series.get("concept"):
        out.append("Concept: " + json.dumps(series["concept"], ensure_ascii=False))
    if series.get("outline"):
        out.append("Series outline:\n" + "\n".join(f"- Part {i + 1}: {p}" for i, p in enumerate(series["outline"])))
    if series.get("previously"):
        out.append("What happened in earlier parts:\n" + "\n".join(f"- Part {i + 1}: {p}"
                                                                  for i, p in enumerate(series["previously"])))
        out.append("Open with the narrator giving a one-line recap ('Pichhle part mein…'), then continue straight "
                   "from the cliffhanger.")
    return "\n".join(out)


# ── mock (offline test) ─────────────────────────────────────────────────
def _mock_script(series: dict) -> dict:
    chars = series.get("characters") or [
        {"id": "C1", "name": "Savitri", "gender": "female", "age": 30, "voice_type": "adult_female",
         "look": "slim Indian woman, wheatish skin, black hair in a low bun, small red bindi, mangalsutra, "
                 "maroon cotton saree with beige blouse, glass bangles", "personality": "tired, protective", "is_entity": False},
        {"id": "C2", "name": "Chhotu", "gender": "male", "age": 6, "voice_type": "child_boy",
         "look": "small 6-year-old Indian boy, short black hair, light blue half-sleeve school shirt, brown shorts",
         "personality": "calm, knows too much", "is_entity": False},
    ]
    images = [
        {"id": "I1", "framing": "establishing", "characters": ["C1", "C2"], "night": True,
         "image": "Savitri carrying Chhotu on an empty cracked highway at night, fields on both sides, full moon"},
        {"id": "I2", "framing": "closeup", "characters": ["C1"], "night": True,
         "image": "Savitri worried, sweat on her forehead, looking down the dark road"},
        {"id": "I3", "framing": "closeup", "characters": ["C2"], "night": True,
         "image": "Chhotu looking up at his mother, calm and serious"},
        {"id": "I4", "framing": "object", "characters": [], "night": True,
         "image": "a phone screen in a woman's hand showing no signal bars, dark road behind"},
        {"id": "I5", "framing": "wide", "characters": ["C2"], "night": True,
         "image": "Chhotu pointing at an old stone by the roadside marked with a faded red cross"},
    ]
    L = lambda sp, hi, ro, st: {"speaker": sp, "hi": hi, "roman": ro, "style": st}
    cuts = [
        {"image": "I1", "motion": "zoom_in", "mood": "uneasy",
         "line": L("N", "उस रात हाईवे पर एक माँ अपने बेटे को लेकर अकेली चल रही थी।",
                   "Us raat highway par ek maa apne bete ko lekar akeli chal rahi thi.", "deep, slow storyteller")},
        {"image": "I2", "motion": "push_fast", "mood": "uneasy",
         "line": L("C1", "बस थोड़ी दूर और बेटा… अस्पताल पास ही है।", "Bas thodi door aur beta… aspataal paas hi hai.", "breathless, reassuring")},
        {"image": "I3", "motion": "zoom_in", "mood": "uneasy",
         "line": L("C2", "माँ… हम कल भी इसी रास्ते पे चले थे ना?", "Maa… hum kal bhi isi raaste pe chale the na?", "soft, innocent")},
        {"image": "I2", "motion": "hold", "mood": "tense",
         "line": L("C1", "क्या बोल रहा है? <short pause> कल तो हम घर पे थे।", "Kya bol raha hai? Kal toh hum ghar pe the.", "uneasy laugh")},
        {"image": "I4", "motion": "push_fast", "mood": "tense", "line": None},
        {"image": "I5", "motion": "pan_right", "mood": "tense",
         "line": L("C2", "वो पत्थर देखो माँ… कल भी वहीं था।", "Woh patthar dekho maa… kal bhi wahin tha.", "calm, certain")},
        {"image": "I2", "motion": "push_fast", "mood": "scare",
         "line": L("C1", "<gasp> छोटू… तुझे कैसे पता?", "Chhotu… tujhe kaise pata?", "frightened whisper")},
    ]
    return {"characters": chars, "images": images, "cuts": cuts,
            "summary": "Savitri carries Chhotu down an empty highway to a hospital; Chhotu insists they walked this road yesterday and recognises a marked stone.",
            "next_part_hook": "Why does Chhotu remember a road they've never walked?"}


# ── normalisation ───────────────────────────────────────────────────────
def _norm_line(ln, ids: set) -> dict | None:
    if not isinstance(ln, dict):
        return None
    sp = str(ln.get("speaker", "")).upper()
    hi = (ln.get("hi") or "").strip()
    if (sp not in ids and sp != "N") or not hi:
        return None
    return {"speaker": sp, "hi": hi, "roman": (ln.get("roman") or "").strip() or strip_tags(hi),
            "style": (ln.get("style") or "").strip()[:60]}


def _clean(script: dict, series: dict, max_images: int = 40) -> dict:
    chars = series.get("characters") or script.get("characters") or []
    seen = []
    for i, c in enumerate(chars[:4]):
        cid = str(c.get("id", ""))
        c["id"] = cid if re.fullmatch(r"C[1-4]", cid) and cid not in seen else f"C{i + 1}"
        seen.append(c["id"])
        if c.get("voice_type") not in VOICE_TYPES:
            fem = str(c.get("gender", "")).lower().startswith("f")
            age = int(c.get("age") or 25) if str(c.get("age", "")).isdigit() else 25
            c["voice_type"] = ("child_girl" if fem else "child_boy") if age < 13 else \
                ("young_female" if fem else "young_male") if age < 35 else \
                ("adult_female" if fem else "adult_male") if age < 60 else ("old_female" if fem else "old_male")
        c["is_entity"] = bool(c.get("is_entity"))
        c.setdefault("look", "")
        c.setdefault("name", c["id"])
    chars = chars[:4]
    ids = {c["id"] for c in chars}

    images = {}
    for k, im in enumerate(script.get("images") or []):
        iid = str(im.get("id") or f"I{k + 1}")
        desc = (im.get("image") or "").strip()
        if not desc or iid in images:
            continue
        images[iid] = {"id": iid, "image": desc,
                       "framing": im.get("framing") if im.get("framing") in FRAMINGS else "medium",
                       "characters": [c for c in (im.get("characters") or []) if c in ids][:3],
                       "night": bool(im.get("night", True))}

    shots = []
    if script.get("cuts"):
        for cut in script["cuts"]:
            im = images.get(str(cut.get("image", "")))
            if not im:
                continue
            ln = _norm_line(cut.get("line"), ids)
            shots.append({"image_id": im["id"], "image": im["image"], "framing": im["framing"],
                          "characters": im["characters"], "night": im["night"],
                          "motion": cut.get("motion") if cut.get("motion") in MOTIONS else "zoom_in",
                          "mood": cut.get("mood") if cut.get("mood") in MOODS else "uneasy",
                          "lines": [ln] if ln else []})
    elif script.get("shots"):  # already normalised (e.g. a revision of our own output)
        for sh in script["shots"]:
            if sh.get("image_id") and sh.get("image"):
                sh["lines"] = [x for x in (_norm_line(l, ids) for l in sh.get("lines") or []) if x]
                shots.append(sh)
                images.setdefault(sh["image_id"], {k: sh[k] for k in ("image", "framing", "characters", "night")}
                                  | {"id": sh["image_id"]})

    # keep the number of drawings inside the daily budget: extra images fall back to the nearest earlier one
    used_order = list(dict.fromkeys(sh["image_id"] for sh in shots))
    allowed = set(used_order[:max_images])
    last_ok = None
    for sh in shots:
        if sh["image_id"] in allowed:
            last_ok = sh
        elif last_ok:
            for k in ("image_id", "image", "framing", "characters", "night"):
                sh[k] = last_ok[k]
    script["characters"] = chars
    script["images"] = [images[i] for i in used_order if i in allowed]
    script["shots"] = shots
    script.pop("cuts", None)
    first = next((ln for sh in shots for ln in sh["lines"]), None)
    last = next((ln for sh in reversed(shots) for ln in reversed(sh["lines"])), None)
    script["hook"] = first["roman"] if first else ""
    script["ending"] = last["roman"] if last else ""
    script.setdefault("summary", "")
    script.setdefault("next_part_hook", "")
    return script


def to_screenplay(script: dict) -> dict:
    """The compact images + cuts form, used when sending a script back to the model."""
    cuts = [{"image": sh["image_id"], "motion": sh["motion"], "mood": sh["mood"],
             "line": sh["lines"][0] if sh["lines"] else None} for sh in script["shots"]]
    return {"characters": script["characters"], "images": script["images"], "cuts": cuts,
            "summary": script.get("summary", ""), "next_part_hook": script.get("next_part_hook", "")}


def write(ctx: Context, series: dict, fmt: dict) -> dict:
    chars_part = (f"The characters are FIXED from earlier parts; reuse them exactly (same ids, names, looks):\n"
                  f"{json.dumps(series['characters'], ensure_ascii=False, indent=1)}\n"
                  if series.get("characters") else
                  "Create 2-4 characters with fresh Indian names (not Rahul/Priya). Give each one precise, fixed look.\n")
    spec = ("Return JSON:\n{{\n  " + ("" if series.get("characters") else CHARACTER_SPEC + "\n  ") + BODY_SPEC
            + '\n  "summary": "2-3 sentences: what happened in this part",'
            + '\n  "next_part_hook": "1 sentence: what the next part is about (empty if final)"\n}}')
    prompt = f"""You write viral illustrated Hindi story reels for "{ctx.cfg['channel']['name']}"
(audience: {ctx.cfg['channel']['audience']}). Each reel is a fast-cut sequence of illustrations with voiced dialogue,
like an animated short.
Format: {fmt['label']} — {fmt['brief']}

{_series_block(series)}

{chars_part}
{CRAFT}

{HINGLISH_GUIDE}

{SAFETY_RULES}

{spec.format(**_fmt(ctx))}
{RULES.format(**_fmt(ctx, _ending_rule(series), series))}"""
    raw = ctx.llm.json(prompt, temperature=0.9, mock=_mock_script(series))
    script = _clean(raw, series, ctx.cfg["story"]["max_images"])
    log(NAME, f"part {series['part']}/{series['total_parts']}: {len(script['shots'])} cuts, "
              f"{len(script['images'])} images, {line_count(script)} lines, ~{estimate_seconds(ctx, script):.0f}s")
    return script


def revise(ctx: Context, script: dict, feedback: str, series: dict, fmt: dict) -> dict:
    prompt = f"""Revise this illustrated Hindi story reel using the editor's feedback. Keep what works.
Keep the same characters (ids, names, looks).

{_series_block(series)}

Current screenplay:
{json.dumps(to_screenplay(script), indent=1, ensure_ascii=False)}

Editor feedback:
{feedback}

{CRAFT}

{HINGLISH_GUIDE}

{SAFETY_RULES}

Return the full revised screenplay as JSON with the same keys ("characters", "images", "cuts", "summary",
"next_part_hook").
{RULES.format(**_fmt(ctx, _ending_rule(series), series))}"""
    fixed = {**series, "characters": series.get("characters") or script.get("characters")}
    new = _clean(ctx.llm.json(prompt, temperature=0.8, mock=to_screenplay(script)), fixed,
                 ctx.cfg["story"]["max_images"])
    if not new["shots"]:
        log(NAME, "revision came back empty; keeping the previous draft")
        return script
    log(NAME, f"revision: {len(new['shots'])} cuts, {len(new['images'])} images, ~{estimate_seconds(ctx, new):.0f}s")
    return new


# ── helpers used by other agents ────────────────────────────────────────
def strip_tags(text: str) -> str:
    return re.sub(r"\s*<[^>]+>\s*", " ", text).strip()


def all_lines(script: dict) -> list[dict]:
    return [ln for sh in script["shots"] for ln in sh["lines"]]


def line_count(script: dict) -> int:
    return len(all_lines(script))


def word_count(script: dict) -> int:
    return sum(len(ln["roman"].split()) for ln in all_lines(script))


def estimate_seconds(ctx: Context, script: dict) -> float:
    silent = sum(1 for sh in script["shots"] if not sh["lines"])
    return (word_count(script) / 2.3 + 0.3 * line_count(script)
            + silent * ctx.cfg["voice"]["silent_shot_seconds"])
