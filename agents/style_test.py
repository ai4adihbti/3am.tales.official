"""Style test: draws the SAME characters and scenes in every art-style preset, on the real image model,
so the channel's look can be chosen by eye (and with an independent Gemini art-director rating).

Run it from GitHub Actions with mode "style_test" (or locally: RUN_MODE=style_test python main.py).
Output in out/: style-test.jpg (all styles side by side), style-test.json (ratings), and every picture.
Cost: ~900 free Cloudflare neurons per style (~4,500 for all five), nothing is posted.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

from .common import Context, apply_style, log
from . import illustrator

NAME = "StyleTest"

CHARACTERS = [
    {"id": "C1", "name": "Meera", "gender": "female", "age": 17, "is_entity": False,
     "look": "slim village girl, wheatish skin, long straight black hair, small black bindi, small gold jhumkas, "
             "mustard-yellow cotton salwar kameez with small green prints and a dark green dupatta"},
    {"id": "C2", "name": "Dadi", "gender": "female", "age": 72, "is_entity": False,
     "look": "thin old Indian grandmother, wrinkled kind face, grey hair in a small bun, round glasses, "
             "rudraksha mala, faded maroon cotton saree with a white border"},
]
SHOTS = [
    {"id": "S1", "framing": "extreme_closeup", "characters": ["C1"], "night": True,
     "image": "Meera's face, terrified, tears rolling down her cheeks, lips trembling, cold moonlight on one side "
              "of her face and warm lantern light on the other"},
    {"id": "S2", "framing": "medium", "characters": ["C2"], "night": True,
     "image": "Dadi standing in the doorway of a mud house holding up a glowing brass lantern, worried, looking "
              "out into the dark lane"},
    {"id": "S3", "framing": "establishing", "characters": [], "night": True,
     "image": "a small North-Indian village of mud houses with tiled roofs at night, a few lit windows, fields, "
              "a huge old peepal tree at the edge of the village, low mist"},
    {"id": "S4", "framing": "wide", "characters": ["C1"], "night": True,
     "image": "Meera seen from behind walking alone down a dirt path towards a huge twisted peepal tree; between "
              "its roots in the distance stands a pale woman with very long open hair and her face hidden"},
]


def _sheet(rows: list[tuple[str, dict[str, Path]]]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 34)
        small = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        big = small = ImageFont.load_default()
    tw, th, pw, head, gap = 270, 480, 240, 56, 10
    width = pw + 4 * (tw + gap) + 2 * gap
    sheet = Image.new("RGB", (width, len(rows) * (th + head + gap) + gap), (18, 18, 22))
    d = ImageDraw.Draw(sheet)
    for r, (label, pics) in enumerate(rows):
        y = gap + r * (th + head + gap)
        d.text((gap, y + 10), label, fill=(255, 214, 90), font=big)
        y += head
        for k, cid in enumerate(("C1", "C2")):
            if cid in pics:
                sheet.paste(Image.open(pics[cid]).convert("RGB").resize((pw - gap, pw - gap)), (gap, y + k * pw))
                d.text((gap + 6, y + k * pw + 4), "Meera" if cid == "C1" else "Dadi", fill=(255, 255, 255), font=small)
        for k, sh in enumerate(SHOTS):
            x = pw + gap + k * (tw + gap)
            if sh["id"] in pics:
                sheet.paste(Image.open(pics[sh["id"]]).convert("RGB").resize((tw, th)), (x, y))
            else:
                d.text((x + 20, y + th // 2), "failed", fill=(255, 80, 80), font=small)
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=90)
    return buf.getvalue()


def run(ctx: Context, out_dir: Path) -> dict:
    v = ctx.cfg["visuals"]
    presets = v.get("style_presets") or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, labels = [], {}
    seed = random.randint(1, 2**30)
    chars = {c["id"]: c for c in CHARACTERS}
    for name in presets:
        apply_style(ctx.cfg, name)
        labels[name] = v.get("style_label", name)
        studio = illustrator.Studio(ctx)
        studio.hq = None                      # same model for every style, so the comparison is fair
        pics: dict[str, Path] = {}
        for c in CHARACTERS:
            dest = out_dir / f"style-{name}-{c['name'].lower()}.png"
            try:
                studio.draw(illustrator.portrait_prompt(ctx, c), 768, 768, seed, [], dest)
                pics[c["id"]] = dest
            except illustrator.ImageError as e:
                log(NAME, f"{name}: portrait of {c['name']} failed ({str(e)[:100]})")
        for k, sh in enumerate(SHOTS):
            dest = out_dir / f"style-{name}-{sh['id']}.png"
            ref_ids = [cid for cid in sh["characters"] if cid in pics]
            try:
                studio.draw(illustrator.shot_prompt(ctx, sh, chars, ref_ids), v["width"], v["height"],
                            seed + 31 * (k + 1), [pics[c] for c in ref_ids], dest)
                pics[sh["id"]] = dest
            except illustrator.ImageError as e:
                log(NAME, f"{name}: {sh['id']} failed ({str(e)[:100]})")
        rows.append((labels[name], pics))
        log(NAME, f"{labels[name]}: {len(pics)}/6 pictures · providers {sorted(studio.used)} · ~{studio.neurons:.0f} neurons")

    sheet = _sheet(rows)
    (out_dir / "style-test.jpg").write_bytes(sheet)

    prompt = f"""You are a senior art director for an Instagram channel of Hindi horror-and-emotion story reels for
Indian Gen Z (16-30). The image compares {len(rows)} art styles, one per row (row titles on the image). In each row:
two character portraits on the left (Meera, a 17-year-old village girl; Dadi, her grandmother), then four scenes:
S1 Meera crying in extreme close-up, S2 Dadi with a lantern, S3 the village at night, S4 Meera walking towards a
peepal tree with a ghostly woman.

Rate every style 1-10 on:
- consistency: does Meera look like the same person in her portrait, S1 and S4? (the channel tells stories in parts)
- clean: free of AI flaws a viewer would notice (melted faces, extra fingers, garbled details, plastic look)
- emotion: do the faces carry real feeling (S1 especially)?
- horror_mood: is it eerie and atmospheric?
- scroll_stop: would the best frame stop someone scrolling Reels at night / work as a cover?
- desi_feel: does it feel authentically Indian?
Then pick the best style for this channel to keep for a long time, and say why in 2-3 sentences.

Style keys: {json.dumps(labels, ensure_ascii=False)}
Return JSON: {{"ratings": {{"<style key>": {{"consistency":0,"clean":0,"emotion":0,"horror_mood":0,"scroll_stop":0,
"desi_feel":0,"note":"one sentence"}}}}, "best": "<style key>", "why": "..."}}"""
    result = ctx.llm.json(prompt, temperature=0.2, images=[("image/jpeg", sheet)],
                          mock={"ratings": {}, "best": "", "why": "mock run"})
    for key, r in (result.get("ratings") or {}).items():
        nums = [x for x in r.values() if isinstance(x, (int, float))]
        r["average"] = round(sum(nums) / len(nums), 1) if nums else 0
        log(NAME, f"{labels.get(key, key)}: {r['average']}/10 · {r.get('note', '')}")
    if result.get("best"):
        log(NAME, f"art director's pick: {labels.get(result['best'], result['best'])} — {result.get('why', '')}")
    (out_dir / "style-test.json").write_text(json.dumps({"labels": labels, **result}, indent=2, ensure_ascii=False),
                                             encoding="utf-8")
    log(NAME, "saved out/style-test.jpg — download it from this run's page (Artifacts)")
    return result
