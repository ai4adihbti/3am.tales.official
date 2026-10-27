"""Illustrator agent: draws every shot and brings it to life with slow camera moves.

1. Character portraits: one reference portrait per character (kept for the whole series), so the
   same face, hair and outfit appear in every shot and every part.
2. Shots: each shot is generated with the portraits of the characters in it as reference images.
3. Motion: every still gets a smooth eased zoom or pan (plus a flash-and-shake on scares),
   rendered at 1080x1920.

Providers (free):
- Cloudflare Workers AI, FLUX.2 [klein] 4B: supports up to 4 reference images → consistent characters.
  Needs CF_ACCOUNT_ID + CF_API_TOKEN. ~10,000 free neurons/day ≈ 55 images.
- Pollinations (backup, no key; POLLINATIONS_TOKEN optional): no reference images, so characters
  are kept consistent through detailed descriptions only.
"""
from __future__ import annotations

import base64
import json
import io
import math
import os
import random
import subprocess
import time
import urllib.parse
from pathlib import Path

import requests

from .common import Context, log

NAME = "Illustrator"

FRAMING = {
    "establishing": "wide establishing shot of the location seen from slightly above, cinematic depth, rich detail, "
                    "characters small in frame or absent",
    "wide": "wide shot, full bodies visible, environment clearly shown",
    "two_shot": "two-shot framing both characters from the waist up, facing each other or the camera",
    "medium": "medium shot from the waist up",
    "closeup": "close-up of the face and shoulders filling the frame, large glossy expressive eyes with catchlights, "
               "emotion clearly readable, detailed hair strands",
    "extreme_closeup": "dramatic extreme close-up on the face and wide glistening eyes filling the frame, "
                       "every eyelash and tear drawn in detail",
    "object": "close-up insert shot of an object or detail, shallow depth of field",
    "over_shoulder": "over-the-shoulder shot",
}
SOFTEN = {"blood": "dark stain", "bloody": "stained", "corpse": "still figure", "dead body": "still figure",
          "skull": "old stone", "gore": "shadow", "knife": "old tool", "hanging": "standing", "kill": "take"}


class ImageError(RuntimeError):
    pass


# ── providers ───────────────────────────────────────────────────────────
def _png(data: bytes) -> bytes:
    from PIL import Image
    im = Image.open(io.BytesIO(data)).convert("RGB")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def _ref_bytes(path: Path, size: int = 448) -> bytes:
    """Reference images must be under 512x512 for FLUX.2 klein (Gemini takes larger ones)."""
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def neurons(model: str, w: int, h: int, n_refs: int = 0) -> float:
    """Cloudflare's published price of one image in neurons (10,000 free per day)."""
    mp = w * h / 1048576
    if "klein-9b" in model:
        return 1363.64 + max(0.0, mp - 1) * 181.82 + n_refs * 0.2 * 181.82
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return tiles * 26.05 + n_refs * 5.37


class Cloudflare:
    name = "cloudflare"

    def __init__(self, cfg: dict, model: str | None = None, name: str = "cloudflare"):
        self.acct = os.getenv("CF_ACCOUNT_ID", "").strip()
        self.token = os.getenv("CF_API_TOKEN", "").strip()
        self.model = model or cfg["visuals"]["cloudflare_model"]
        self.name = name

    @property
    def ready(self) -> bool:
        return bool(self.acct and self.token)

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[Path]) -> bytes:
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.acct}/ai/run/{self.model}"
        fields = {"prompt": (None, prompt[:2000]), "width": (None, str(w)), "height": (None, str(h)),
                  "seed": (None, str(seed))}
        for i, p in enumerate(refs[:4]):
            fields[f"input_image_{i}"] = (f"ref{i}.jpg", _ref_bytes(p), "image/jpeg")
        last = ""
        for attempt in range(1, 4):
            try:
                r = requests.post(url, headers={"Authorization": f"Bearer {self.token}"}, files=fields,
                                  timeout=(15, 180))
            except requests.RequestException as e:
                last = f"network {type(e).__name__}"
                time.sleep(5 * attempt)
                continue
            ctype = r.headers.get("content-type", "")
            if r.status_code == 200 and ctype.startswith("image/"):
                return _png(r.content)
            if r.status_code == 200:
                try:
                    data = r.json()
                    img = (data.get("result") or {}).get("image")
                    if img:
                        return _png(base64.b64decode(img))
                    last = f"no image in reply: {str(data)[:200]}"
                except ValueError:
                    last = "unreadable reply"
                continue
            last = f"HTTP {r.status_code} {r.text[:300]}"
            low = r.text.lower()
            if r.status_code in (401, 403):
                raise ImageError(f"Cloudflare rejected the token/account: {last}")
            if "3030" in r.text or "rejected" in low or "nsfw" in low or "safety" in low:
                raise ImageError(f"SAFETY {last}")
            if "4006" in r.text or "daily free allocation" in low or "neurons" in low:
                raise ImageError(f"QUOTA {last}")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10 * attempt)
                continue
            raise ImageError(last)
        raise ImageError(last)


class NanoBanana:
    """Google's Gemini image model ("Nano Banana"), paid from the Gemini API (your Google AI Pro credit).
    Best character consistency and the closest match to the reference style. Up to 4 character references."""
    name = "gemini"
    PRICE = 0.0336  # USD per 1K image

    def __init__(self, cfg: dict, budget_usd: float):
        self.key = os.getenv("GEMINI_API_KEY", "").strip()
        self.model = cfg["visuals"].get("gemini_image_model", "gemini-nano-banana-2.1")
        self.budget = budget_usd
        self.count = 0
        self.wrap = False

    @property
    def ready(self) -> bool:
        return bool(self.key) and self.budget >= self.PRICE * (self.count + 1)

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[Path]) -> bytes:
        if not self.ready:
            raise ImageError("QUOTA monthly Gemini image budget reached")
        items = [{"type": "text", "text": prompt[:4000]}]
        for p in refs[:4]:
            items.append({"type": "image", "mime_type": "image/jpeg", "data": base64.b64encode(_ref_bytes(p, 768)).decode()})
        fmt = {"type": "image", "mime_type": "image/png", "aspect_ratio": "1:1" if w == h else "9:16", "image_size": "1K"}
        last = ""
        for attempt in range(1, 4):
            body = {"model": self.model, "response_format": fmt,
                    "input": [{"type": "user_input", "content": items}] if self.wrap else items}
            try:
                r = requests.post("https://generativelanguage.googleapis.com/v1beta/interactions", json=body,
                                  headers={"x-goog-api-key": self.key, "Content-Type": "application/json"},
                                  timeout=(15, 240))
            except requests.RequestException as e:
                last = f"network {type(e).__name__}"
                time.sleep(5 * attempt)
                continue
            if r.status_code == 200:
                img = _find_image(r.json())
                if img:
                    self.count += 1
                    return _png(img)
                last = f"no image in reply: {r.text[:200]}"
                if "safety" in r.text.lower() or "blocked" in r.text.lower():
                    raise ImageError(f"SAFETY {last}")
                continue
            last = f"HTTP {r.status_code} {r.text[:300]}"
            low = r.text.lower()
            if r.status_code == 400 and not self.wrap and ("input" in low or "invalid" in low) and attempt == 1:
                self.wrap = True   # try the other request shape once
                continue
            if r.status_code in (401, 403) or "billing" in low or "free tier" in low or "limit: 0" in low:
                raise ImageError(f"QUOTA Gemini images need billing linked to your API project: {last}")
            if "safety" in low or "blocked" in low:
                raise ImageError(f"SAFETY {last}")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(10 * attempt)
                continue
            raise ImageError(last)
        raise ImageError(last)


def _find_image(data: dict) -> bytes | None:
    found = None
    out = data.get("output_image") or {}
    if out.get("data"):
        return base64.b64decode(out["data"])
    for step in data.get("steps", []) or []:
        for item in step.get("content", []) or []:
            if item.get("type") == "image" and item.get("data"):
                found = item["data"]
    return base64.b64decode(found) if found else None


class Pollinations:
    name = "pollinations"
    _last_call = 0.0

    def __init__(self, cfg: dict):
        self.token = os.getenv("POLLINATIONS_TOKEN", "").strip()

    @property
    def ready(self) -> bool:
        return True

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[Path]) -> bytes:
        gap = 5.5 if self.token else 16.0  # free tiers: 1 request / 5 s (token) or / 15 s (anonymous)
        wait = Pollinations._last_call + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        q = urllib.parse.quote(prompt[:1400], safe="")
        params = {"width": w, "height": h, "seed": seed, "model": "flux", "nologo": "true", "enhance": "false"}
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        last = ""
        waits = [12, 20, 30, 45, 60, 60]          # the free service is often briefly "at its limit"
        for attempt, pause in enumerate(waits, 1):
            Pollinations._last_call = time.time()
            try:
                r = requests.get(f"https://image.pollinations.ai/prompt/{q}", params=params, headers=headers,
                                 timeout=(15, 180))
            except requests.RequestException as e:
                last = f"network {type(e).__name__}"
                time.sleep(pause)
                continue
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                return _png(r.content)
            last = f"HTTP {r.status_code} {r.text[:200]}"
            if r.status_code in (400, 401, 403):
                break
            if attempt == 1:
                log(NAME, f"backup image service busy (HTTP {r.status_code}); retrying for up to ~4 min")
            time.sleep(pause)
        raise ImageError(f"BUSY {last}")


class Mock:
    name = "mock"
    ready = True

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[Path]) -> bytes:
        from PIL import Image, ImageDraw
        rnd = random.Random(seed)
        top = (rnd.randint(20, 60), rnd.randint(25, 70), rnd.randint(50, 100))
        im = Image.new("RGB", (w, h))
        d = ImageDraw.Draw(im)
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)], fill=tuple(int(c * (1 - t) + 12 * t) for c in top))
        for _ in range(6):
            x, y, r = rnd.randint(0, w), rnd.randint(int(h * 0.3), h), rnd.randint(40, 160)
            d.ellipse([x - r, y - r, x + r, y + r], fill=(rnd.randint(60, 140), rnd.randint(50, 110), 60))
        d.ellipse([w * 0.7, h * 0.08, w * 0.82, h * 0.08 + w * 0.12], fill=(235, 235, 215))
        d.text((40, h - 120), prompt[:60], fill=(220, 220, 220))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()


# ── prompts ─────────────────────────────────────────────────────────────
def _soften(text: str) -> str:
    for a, b in SOFTEN.items():
        text = text.replace(a, b).replace(a.capitalize(), b)
    return text


def _person(c: dict) -> str:
    fem = str(c.get("gender", "")).lower().startswith("f")
    try:
        age = int(c.get("age"))
    except (TypeError, ValueError):
        age = 25
    g = ("girl" if fem else "boy") if age < 13 else ("woman" if fem else "man")
    kind = "pale ghostly " if c.get("is_entity") else ""
    return f"{c['name']}, a {age}-year-old {kind}Indian {g}: {c.get('look', '')}".strip()


def portrait_prompt(ctx: Context, c: dict) -> str:
    v = ctx.cfg["visuals"]
    return (f"{v['style']}. Character design reference portrait: head and upper body of {_person(c)}. "
            f"Single person only, facing the camera, calm natural expression, soft even studio lighting, "
            f"plain light grey background, centered, face and outfit clearly visible.")


def shot_prompt(ctx: Context, shot: dict, chars: dict, ref_ids: list[str]) -> str:
    v = ctx.cfg["visuals"]
    parts = [v["style"], v["night_palette"] if shot.get("night", True) else v["day_palette"],
             FRAMING.get(shot["framing"], FRAMING["medium"]) + ", vertical 9:16 composition",
             f"Scene: {shot['image']}"]
    people = []
    for cid in shot["characters"]:
        c = chars.get(cid)
        if not c:
            continue
        tag = f" (the same person as reference image {ref_ids.index(cid)})" if cid in ref_ids else ""
        people.append(f"{_person(c)}{tag}")
    if people:
        parts.append("Characters: " + "; ".join(people))
        if ref_ids:
            parts.append("Keep each character's face, age, hairstyle, skin tone and outfit exactly as in their reference "
                         "image; only the pose, expression and camera angle change. Use the reference images for the "
                         "people only, not for the background or composition")
        parts.append(f"Exactly {len(people)} {'person' if len(people) == 1 else 'people'} in the picture")
    return ". ".join(parts)


# ── main entry points ───────────────────────────────────────────────────
class Studio:
    """Picks the provider for each drawing and switches to backups when one fails.

    premium=True (character portraits and the pictures that stay on screen longest) → the high-quality
    Cloudflare model (FLUX.2 klein 9B) while today's free neurons allow it, or Nano Banana if enabled;
    everything else → Cloudflare klein 4B → Pollinations.
    """

    def __init__(self, ctx: Context, gemini_budget_usd: float = 0.0):
        self.ctx = ctx
        self.used: set[str] = set()
        self.dead: set[str] = set()
        self.neurons = 0.0
        self.hq_allowance = 0.0          # neurons main.py lets the high-quality model spend this run
        self.hq_count = 0
        v = ctx.cfg["visuals"]
        if ctx.mock:
            self.base, self.premium, self.hq = [Mock()], None, None
            return
        base = [Cloudflare(ctx.cfg), Pollinations(ctx.cfg)]
        if v.get("provider") == "pollinations":
            base.reverse()
        self.base = [p for p in base if p.ready]
        hq_model = v.get("cloudflare_hq_model")
        self.hq = Cloudflare(ctx.cfg, hq_model, "cloudflare_hq") if hq_model else None
        if self.hq and not self.hq.ready:
            self.hq = None
        use_gemini = v.get("gemini_portraits", False) or v.get("gemini_hero_shots", 0) > 0
        self.premium = NanoBanana(ctx.cfg, gemini_budget_usd) if use_gemini else None
        if not self.base and not (self.premium and self.premium.ready):
            raise ImageError("No image provider configured")

    @property
    def gemini_images(self) -> int:
        return self.premium.count if self.premium else 0

    def plan(self, n_images: int, n_portraits: int) -> None:
        """Share today's free Cloudflare neurons: every picture and redraw on the 4B model first,
        whatever is left goes to the high-quality 9B model (portraits first, then hero shots)."""
        v = self.ctx.cfg["visuals"]
        if not self.hq:
            return
        daily = float(v.get("cloudflare_daily_neurons", 10000)) * 0.95
        per = neurons(v["cloudflare_model"], v["width"], v["height"], 2)
        need = (n_images + n_portraits + int(v.get("max_redraws", 6))) * per
        self.hq_allowance = max(0.0, daily - need)
        log(NAME, f"neuron plan: ~{need:.0f} for {n_images} pictures + redraws; "
                  f"{self.hq_allowance:.0f} for high-quality drawings (~{int(self.hq_allowance // 1460)})")

    def _order(self, premium: bool, w: int, h: int) -> list:
        order = list(self.base)
        if premium and self.hq and self.hq.name not in self.dead:
            cost = neurons(self.hq.model, w, h, 2)
            if self.hq_allowance - cost >= 0:
                order.insert(0, self.hq)
        if premium and self.premium and self.premium.ready:
            order.insert(0, self.premium)
        return [p for p in order if p.name not in self.dead]

    def draw(self, prompt: str, w: int, h: int, seed: int, refs: list[Path], dest: Path,
             premium: bool = False) -> None:
        errors = []
        for p in self._order(premium, w, h):
            use_refs = refs if p.name in ("cloudflare", "cloudflare_hq", "gemini") else []
            for attempt in range(3):
                try:
                    pr = prompt if attempt == 0 else _soften(prompt)
                    rf = use_refs if attempt < 2 else []   # last try: without reference images
                    dest.write_bytes(p.generate(pr, w, h, seed + attempt * 7919, rf))
                    self.used.add(p.name)
                    if p.name.startswith("cloudflare"):
                        cost = neurons(p.model, w, h, len(rf))
                        self.neurons += cost
                        if p.name == "cloudflare_hq":
                            self.hq_allowance -= cost
                            self.hq_count += 1
                    return
                except ImageError as e:
                    msg = str(e)
                    errors.append(f"{p.name}: {msg[:160]}")
                    if p.name.startswith("cloudflare"):
                        self.neurons += neurons(p.model, w, h, len(rf)) * 0.5   # failures may still be billed
                    if msg.startswith("SAFETY"):
                        continue  # retry with softened wording / no refs
                    break
            fails = len([x for x in errors if x.startswith(p.name + ":") and "BUSY" not in x])
            if "QUOTA" in errors[-1] and p.name.startswith("cloudflare"):
                # both Cloudflare models share one daily allowance
                for q in ("cloudflare", "cloudflare_hq"):
                    self.dead.add(q)
                log(NAME, f"Cloudflare daily allowance used up; using the next provider")
            elif "QUOTA" in errors[-1] or "rejected the token" in errors[-1] or fails >= 3 or \
                    (p.name == "cloudflare_hq" and fails >= 1):
                if p.name not in self.dead:
                    log(NAME, f"{p.name} unavailable ({errors[-1][:150]}); using the next provider")
                self.dead.add(p.name)
            # try the next provider for this drawing
        raise ImageError("All image providers failed: " + "; ".join(errors[-4:]))


def character_refs(ctx: Context, studio: Studio, series: dict, script: dict) -> dict[str, Path]:
    from .common import ROOT
    out_dir = ctx.workdir / "refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Path] = {}
    saved = series.get("ref_files") or {}
    for c in script["characters"]:
        p = ROOT / saved[c["id"]] if c["id"] in saved else None
        if p and p.exists():
            refs[c["id"]] = p
            continue
        dest = out_dir / f"{c['id']}.png"
        seed = random.randint(1, 2**31 - 1)
        try:
            studio.draw(portrait_prompt(ctx, c), 768, 768, seed, [], dest, premium=True)
        except ImageError as e:
            log(NAME, f"portrait of {c['name']} failed ({str(e)[:100]}); continuing with text descriptions only")
            continue
        refs[c["id"]] = dest
        log(NAME, f"portrait: {c['name']}")
    return refs


def hero_images(ctx: Context, script: dict, durations: list[float]) -> set[str]:
    """The pictures that stay on screen longest (usually the reused dialogue close-ups)."""
    v = ctx.cfg["visuals"]
    n = max(int(v.get("gemini_hero_shots", 0)), int(v.get("hq_hero_shots", 0)))
    if n <= 0:
        return set()
    screen: dict[str, float] = {}
    for sh, d in zip(script["shots"], durations):
        screen[sh["image_id"]] = screen.get(sh["image_id"], 0.0) + d
    return set(sorted(screen, key=screen.get, reverse=True)[:n])


def illustrate(ctx: Context, studio: Studio, script: dict, refs: dict[str, Path],
               heroes: set[str] | None = None) -> dict[str, Path]:
    """Draws every unique image once. Returns {image_id: png}. Failed drawings reuse a neighbour."""
    v = ctx.cfg["visuals"]
    heroes = heroes or set()
    ctx.notes["heroes"] = sorted(heroes)
    chars = {c["id"]: c for c in script["characters"]}
    out_dir = ctx.workdir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_seed = random.randint(1, 2**30)
    paths: dict[str, Path] = {}
    failed: list[str] = []
    for i, im in enumerate(script["images"]):
        dest = out_dir / f"{im['id']}.png"
        try:
            _draw_image(ctx, studio, im, chars, refs, dest, base_seed + i * 31, premium=im["id"] in heroes)
            paths[im["id"]] = dest
        except ImageError as e:
            log(NAME, f"{im['id']}: could not draw ({str(e)[:120]})")
            failed.append(im["id"])
        if (i + 1) % 8 == 0 or i == len(script["images"]) - 1:
            log(NAME, f"{i + 1}/{len(script['images'])} illustrations drawn")
    if not paths:
        raise ImageError("No illustration could be drawn")
    for iid in failed:  # reuse the closest earlier (or any) drawing so the reel still works
        order = [im["id"] for im in script["images"]]
        k = order.index(iid)
        donor = next((order[j] for j in range(k - 1, -1, -1) if order[j] in paths), next(iter(paths)))
        paths[iid] = paths[donor]
    ctx.notes["image_providers"] = sorted(studio.used)
    ctx.notes["images_failed"] = failed
    return paths


def _draw_image(ctx: Context, studio: Studio, im: dict, chars: dict, refs: dict[str, Path], dest: Path,
                seed: int, extra: str = "", premium: bool = False) -> None:
    v = ctx.cfg["visuals"]
    ref_ids = [cid for cid in im["characters"] if cid in refs][:4]
    prompt = shot_prompt(ctx, im, chars, ref_ids) + (f". {extra}" if extra else "")
    studio.draw(prompt, v["width"], v["height"], seed, [refs[c] for c in ref_ids], dest, premium=premium)


# ── quality inspector ───────────────────────────────────────────────────
def _contact_sheet(script: dict, refs: dict[str, Path], paths: dict[str, Path]) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    cols, tw, th = 6, 200, 356
    ids = [im["id"] for im in script["images"] if im["id"] in paths]
    rows = (len(ids) + cols - 1) // cols
    head = 300
    sheet = Image.new("RGB", (cols * tw, head + rows * (th + 34)), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    names = {c["id"]: c["name"] for c in script["characters"]}
    for k, (cid, p) in enumerate(refs.items()):
        im = Image.open(p).convert("RGB").resize((230, 230))
        sheet.paste(im, (10 + k * 270, 10))
        d.text((10 + k * 270, 245), f"{cid} {names.get(cid, '')}"[:18], fill=(255, 220, 90), font=font)
    for k, iid in enumerate(ids):
        im = Image.open(paths[iid]).convert("RGB").resize((tw - 8, th))
        x, y = (k % cols) * tw + 4, head + (k // cols) * (th + 34)
        sheet.paste(im, (x, y + 30))
        d.text((x + 4, y), iid, fill=(255, 255, 255), font=font)
    import io
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def quality_check(ctx: Context, studio: Studio, script: dict, refs: dict[str, Path],
                  paths: dict[str, Path]) -> list[str]:
    """Gemini looks at all the illustrations at once and flags the bad ones; those get redrawn once."""
    v = ctx.cfg["visuals"]
    if not v.get("quality_check", True) or ctx.mock:
        return []
    chars = {c["id"]: c for c in script["characters"]}
    expected = {im["id"]: {"shows": im["image"][:160], "visible": [chars[c]["name"] for c in im["characters"] if c in chars]}
                for im in script["images"] if im["id"] in paths}
    prompt = f"""You are the art director of an illustrated Indian story reel. The image shows a contact sheet:
top row = the character reference portraits (labelled with id and name); below = the illustrations, labelled I1, I2 …

What each illustration should show:
{json.dumps(expected, ensure_ascii=False, indent=1)}

Flag ONLY serious problems a viewer would notice:
- a character clearly doesn't match their reference (different face, hair, age, gender or outfit)
- wrong people in the picture (missing, extra or duplicated person)
- deformed faces, eyes, hands or extra limbs
- visible text, letters, captions, watermarks or logos
- photographic or 3D render instead of the 2D illustrated style, or very blurry
- off-style pictures: the target look is "{v.get('style_label', 'the channel style')}" — clean crisp line art,
  rich detail, cinematic night light, large expressive eyes on emotional close-ups; flag flat, dull, muddy or
  half-realistic pictures
- the same character drawn with a different hairstyle, hair colour or outfit colour than in their portrait
- anything gory or unsafe

Return JSON: {{"redraw": [{{"id": "I4", "problem": "short", "fix": "one sentence to add to the drawing prompt"}}]}}
At most {v.get('max_redraws', 6)} items, worst first. Return an empty list if everything is fine."""
    try:
        result = ctx.llm.json(prompt, temperature=0.2, images=[("image/jpeg", _contact_sheet(script, refs, paths))],
                              mock={"redraw": []})
    except Exception as e:  # never let the inspection itself break the run
        log(NAME, f"quality check skipped ({str(e)[:120]})")
        return []
    items = [x for x in (result.get("redraw") or []) if isinstance(x, dict) and x.get("id") in paths]
    items = items[: v.get("max_redraws", 6)]
    if not items:
        log(NAME, "quality check: all illustrations passed")
        return []
    ims = {im["id"]: im for im in script["images"]}
    fixed = []
    for x in items:
        iid = x["id"]
        dest = paths[iid].with_name(f"{iid}_redraw.png")
        try:
            _draw_image(ctx, studio, ims[iid], chars, refs, dest, random.randint(1, 2**30),
                        extra=str(x.get("fix", ""))[:200], premium=iid in ctx.notes.get("heroes", []))
            paths[iid] = dest
            fixed.append(iid)
            log(NAME, f"redrew {iid}: {str(x.get('problem', ''))[:80]}")
        except ImageError as e:
            log(NAME, f"redraw of {iid} failed ({str(e)[:80]}); keeping the original")
    ctx.notes["redrawn"] = fixed
    return fixed


# ── motion ──────────────────────────────────────────────────────────────
def _motion_exprs(motion: str, n: int, m: float, scare: bool) -> tuple[str, str, str]:
    pe = f"((1-cos(PI*on/{max(1, n - 1)}))/2)"         # eased progress 0→1
    cx, cy = "(iw/2-iw/zoom/2)", "(ih/2-ih/zoom/2)"
    if motion == "zoom_out":
        z, x, y = f"{1 + m}-{m}*{pe}", cx, cy
    elif motion == "pan_left":
        z, x, y = f"{1 + m}", f"(iw-iw/zoom)*(1-{pe})", cy
    elif motion == "pan_right":
        z, x, y = f"{1 + m}", f"(iw-iw/zoom)*{pe}", cy
    elif motion == "pan_up":
        z, x, y = f"{1 + m}", cx, f"(ih-ih/zoom)*(1-{pe})"
    elif motion == "pan_down":
        z, x, y = f"{1 + m}", cx, f"(ih-ih/zoom)*{pe}"
    elif motion == "push_fast":
        fast = f"(1-pow(1-on/{max(1, n - 1)},3))"
        z, x, y = f"1+{1.7 * m}*{fast}", cx, cy
    elif motion == "hold":  # barely-there breathing zoom
        z, x, y = f"1.03+0.02*{pe}", cx, cy
    else:  # zoom_in
        z, x, y = f"1+{m}*{pe}", cx, cy
    if scare:  # short camera shake in the first second
        x = f"({x})+if(lt(on,24),10*sin(on*2.3)*(1-on/24),0)"
        y = f"({y})+if(lt(on,24),8*cos(on*1.9)*(1-on/24),0)"
    return z, x, y


ALT = {"zoom_in": "zoom_out", "zoom_out": "zoom_in", "pan_left": "pan_right", "pan_right": "pan_left",
       "hold": "zoom_in", "push_fast": "hold", "pan_up": "pan_down", "pan_down": "pan_up"}


def animate(ctx: Context, image_paths: dict[str, Path], shots: list[dict], durations: list[float]) -> list[Path]:
    vcfg = ctx.cfg["video"]
    w, h, fps = vcfg["width"], vcfg["height"], vcfg["fps"]
    xf = vcfg.get("crossfade", 0.12)
    m0 = float(ctx.cfg["visuals"].get("motion_strength", 0.10))
    out_dir = ctx.workdir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    big: dict[str, Path] = {}
    last_motion: dict[str, str] = {}
    clips = []
    for i, (shot, d) in enumerate(zip(shots, durations)):
        iid = shot["image_id"]
        if iid not in big:  # upscale each drawing once (2x canvas → smooth sub-pixel camera moves)
            big[iid] = out_dir / f"big_{iid}.png"
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(image_paths[iid]), "-vf",
                            f"scale={w * 2}:{h * 2}:flags=lanczos", str(big[iid])], check=True)
        motion = shot.get("motion", "zoom_in")
        if last_motion.get(iid) == motion:       # same picture again → move the camera differently
            motion = ALT.get(motion, "zoom_in")
        last_motion[iid] = motion
        length = d + (xf if i < len(shots) - 1 else 0) + 0.05
        n = int(math.ceil(length * fps))
        m = m0 * min(1.0, max(0.5, d / 3.5))      # short cuts move less, so motion never looks rushed
        scare = shot.get("mood") == "scare"
        z, x, y = _motion_exprs(motion, n, m, scare)
        vf = f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={w}x{h}:fps={fps},setsar=1"
        if scare:
            vf += ",fade=t=in:st=0:d=0.15:color=white"
        out = out_dir / f"clip{i:02d}.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(big[iid]), "-vf", vf, "-frames:v", str(n),
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
                        str(out)], check=True)
        clips.append(out)
    log(NAME, f"{len(clips)} cuts animated")
    return clips
