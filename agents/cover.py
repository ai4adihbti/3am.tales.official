"""Cover agent: makes the reel's cover — what people see on the profile grid, in Explore and in shares.

The most emotional close-up of the reel (a frightened or crying face), brightened so it pops on a phone,
with the story title in big bold letters and a small "PART n" / "HINDI HORROR STORY" tag. All text stays
inside the centre 3:4 area, because the profile grid crops covers to 3:4.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from .common import Context, log

NAME = "Cover"
W, H = 1080, 1920
SAFE_TOP, SAFE_BOTTOM = (H - 1440) // 2, (H + 1440) // 2       # the 3:4 grid crop


def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def pick_image(script: dict, image_paths: dict[str, Path]) -> tuple[str, int]:
    """(image id, index of its first cut): an emotional close-up of a person, else the first picture."""
    weight = {"extreme_closeup": 3, "closeup": 2}
    mood_bonus = {"scare": 2, "sad": 2, "tense": 1}
    best, best_score, best_cut = None, -1, 0
    for k, sh in enumerate(script["shots"]):
        if sh["image_id"] not in image_paths or not sh.get("characters"):
            continue
        score = weight.get(sh.get("framing"), 0) * 2 + mood_bonus.get(sh.get("mood"), 0) - k * 0.01
        if score > best_score:
            best, best_score, best_cut = sh["image_id"], score, k
    if best is None:
        best = next(iter(image_paths))
    return best, best_cut


def make(ctx: Context, script: dict, series: dict, image_paths: dict[str, Path], dest: Path) -> Path | None:
    if not ctx.cfg["caption"].get("cover", True):
        return None
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    iid, _ = pick_image(script, image_paths)
    im = Image.open(image_paths[iid]).convert("RGB")
    s = max(W / im.width, H / im.height)
    im = im.resize((int(im.width * s + 0.5), int(im.height * s + 0.5)), Image.LANCZOS)
    x0, y0 = (im.width - W) // 2, (im.height - H) // 2
    im = im.crop((x0, y0, x0 + W, y0 + H))
    im = ImageEnhance.Brightness(im).enhance(1.18)
    im = ImageEnhance.Contrast(im).enhance(1.12)
    im = ImageEnhance.Color(im).enhance(1.15)

    # dark gradient behind the title (lower part of the 3:4 area) so white text always reads
    shade = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(shade)
    g0 = SAFE_BOTTOM - 640
    for y in range(g0, H):
        d.line([(0, y), (W, y)], fill=int(min(215, (y - g0) / 520 * 215)))
    shade = shade.filter(ImageFilter.GaussianBlur(6))
    im = Image.composite(Image.new("RGB", (W, H), (6, 4, 12)), im, shade)

    draw = ImageDraw.Draw(im)
    title = str(series.get("title") or "3AM Tale").strip()
    size = 112
    while size > 64:
        font = _font(size)
        lines = textwrap.wrap(title.upper(), width=max(8, int(15 * 112 / size)))
        widths = [draw.textlength(l, font=font) for l in lines]
        if max(widths) <= W - 120 and len(lines) <= 3:
            break
        size -= 8
    line_h = int(size * 1.08)
    y = SAFE_BOTTOM - 110 - line_h * len(lines)
    for l in lines:
        tw = draw.textlength(l, font=font)
        draw.text(((W - tw) / 2, y), l, font=font, fill=(255, 255, 255), stroke_width=max(4, size // 14),
                  stroke_fill=(0, 0, 0))
        y += line_h

    total = int(series.get("total_parts") or 1)
    tag = f"PART {series.get('part', 1)}" if total > 1 else "HINDI HORROR STORY"
    tf = _font(44)
    tw = draw.textlength(tag, font=tf)
    ty = SAFE_BOTTOM - 92
    draw.rounded_rectangle(((W - tw) / 2 - 26, ty - 8, (W + tw) / 2 + 26, ty + 58), radius=14, fill=(200, 16, 32))
    draw.text(((W - tw) / 2, ty), tag, font=tf, fill=(255, 255, 255))

    im.save(dest, "JPEG", quality=92)
    log(NAME, f"cover from {iid}: \"{title}\" · {tag}")
    return dest
