"""Sound designer: real sound effects from Freesound (free, CC0 only — no credit needed, commercial use OK).

- An ambience bed for the whole reel, chosen from the setting (night village crickets, wind, rain, city night…).
- Scene effects on the cuts where the picture or narration mentions something with a sound: a door creaking,
  anklet bells (payal), an owl, a dog howling, footsteps, dripping water in a well or baoli, a temple bell…
Everything is mixed into the score, so the Editor ducks it gently under the storyteller's voice.
Needs FREESOUND_API_KEY; without it (or if Freesound is down) the reel simply has the score alone.
"""
from __future__ import annotations

import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import requests

from .common import ROOT, Context, log

NAME = "Sound"
SR = 48000
API = "https://freesound.org/apiv2"
CACHE = ROOT / ".cache" / "sfx"

# what the picture/narration says → what to search for (first match per cut wins)
CUES = [
    (r"\bdoor|darwaz|kiwad|gate\b", "old wooden door creak"),
    (r"payal|anklet|ghungroo", "anklet bells jingle"),
    (r"\bowl|ullu", "owl hoot night"),
    (r"\bdog|kutt", "dog howling night"),
    (r"knock|khatkhat|dastak", "knocking on wooden door"),
    (r"baoli|stepwell|\bwell\b|kuan|kuaan|drip", "water dripping cave echo"),
    (r"footstep|walk|running|ran |bhaag|chal(ti|ta|te)|kadam|steps", "footsteps on dirt path"),
    (r"temple|mandir|\bbell|ghanti", "temple bell ring"),
    (r"chain|zanjeer", "chains rattling"),
    (r"whisper|phusphus|fusfus|sargoshi", "ghostly whisper"),
    (r"lantern|lalten|diya|\blamp|flame|fire|aag", "fire crackling small"),
    (r"thunder|bijli|storm|toofan", "thunder rumble distant"),
    (r"\brain|baarish|barish", "rain on roof"),
    (r"train|platform", "train passing distant"),
    (r"\bbus\b|truck|highway", "vehicle engine passing night"),
    (r"scream|cheekh|chilla", "distant scream echo"),
    (r"baby|bachcha ro|crying child", "distant baby crying"),
    (r"tree|peepal|banyan|bargad|leaves|patte", "leaves rustling wind"),
    (r"glass|toot|shatter|crash", "glass breaking"),
]
AMBIENCE = [
    (r"rain|baarish|barish|monsoon", "rain night ambience"),
    (r"city|metro|hostel|apartment|office|hospital|flat", "city night ambience distant traffic"),
    (r"forest|jungle", "forest night ambience"),
    (r"train|railway|platform|station", "empty train station night ambience"),
    (r"storm|toofan|wind|aandhi", "wind howling ambience"),
    (r"village|gaon|gaanv|field|khet|baoli|haveli|haweli", "night crickets village ambience"),
]
DEFAULT_AMBIENCE = "night crickets ambience"
OPENING_STING = "horror impact hit"
SCARE_FX = "horror whoosh"


class Freesound:
    def __init__(self, key: str):
        self.key = key
        self.calls = 0

    def find(self, query: str, min_s: float, max_s: float) -> Path | None:
        CACHE.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
        cached = sorted(CACHE.glob(f"{slug}__*.mp3"))
        if cached:
            return cached[0]
        try:
            self.calls += 1
            r = requests.get(f"{API}/search/text/", timeout=30, params={
                "query": query, "token": self.key, "page_size": 8, "sort": "rating_desc",
                "filter": f'license:"Creative Commons 0" duration:[{min_s} TO {max_s}]',
                "fields": "id,name,previews,duration,license"})
            if r.status_code != 200:
                log(NAME, f"search '{query}' failed (HTTP {r.status_code} {r.text[:100]})")
                return None
            results = r.json().get("results") or []
        except (requests.RequestException, ValueError):
            return None
        for res in results:
            if "creative commons 0" not in str(res.get("license", "")).lower() and \
                    "publicdomain/zero" not in str(res.get("license", "")).lower():
                continue                          # only CC0: free for any use, no attribution
            url = (res.get("previews") or {}).get("preview-hq-mp3")
            if not url:
                continue
            try:
                data = requests.get(url, timeout=60).content
            except requests.RequestException:
                continue
            if len(data) > 2000:
                dest = CACHE / f"{slug}__{res['id']}.mp3"
                dest.write_bytes(data)
                return dest
        return None


def _load(path: Path, seconds: float | None = None) -> np.ndarray:
    """Any audio file → float32 stereo at 48 kHz (optionally trimmed)."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if seconds:
        cmd += ["-t", f"{seconds:.2f}"]
    cmd += ["-f", "s16le", "-ac", "2", "-ar", str(SR), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return (np.frombuffer(raw, np.int16).astype(np.float32) / 32768).reshape(-1, 2)


def _db(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20 * np.log10(max(rms, 1e-6))


def _fit_level(x: np.ndarray, target_db: float) -> np.ndarray:
    return x * (10 ** ((target_db - _db(x)) / 20))


def plan(script: dict) -> tuple[str, list[tuple[int, str]]]:
    """(ambience query, [(cut index, effect query)]) from the pictures and narration."""
    shots = script["shots"]
    everything = " ".join(sh.get("image", "") for sh in shots).lower()
    amb = next((q for pat, q in AMBIENCE if re.search(pat, everything)), DEFAULT_AMBIENCE)
    cues, last_at = [], {}
    if shots and shots[0].get("mood") == "scare":
        cues.append((0, OPENING_STING))           # the cold open hits with a real horror sting on frame one
        last_at[OPENING_STING] = 0
    for i, sh in enumerate(shots):
        if sh.get("split_of") is not None or (i == 0 and cues):
            continue                              # extra angles of one line don't repeat its sound
        text = (sh.get("image", "") + " " + " ".join(ln.get("roman", "") for ln in sh.get("lines", []))).lower()
        for pat, q in CUES:
            if re.search(pat, text) and i - last_at.get(q, -99) >= 4:   # don't repeat the same effect too soon
                cues.append((i, q))
                last_at[q] = i
                break
        else:
            if sh.get("mood") == "scare" and i - last_at.get(SCARE_FX, -99) >= 6:
                cues.append((i, SCARE_FX))        # scare cuts without a named sound still get a whoosh
                last_at[SCARE_FX] = i
    return amb, cues[:16]


def add_effects(ctx: Context, score: Path, script: dict, shot_starts: list[float], lead: float,
                total: float) -> Path:
    key = os.getenv("FREESOUND_API_KEY", "").strip()
    if not key or ctx.mock or not ctx.cfg.get("sound", {}).get("enabled", True):
        return score
    fs = Freesound(key)
    sc = ctx.cfg.get("sound", {})
    amb_q, cues = plan(script)
    with wave.open(str(score)) as w:
        mix = (np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768).reshape(-1, 2)
    n = len(mix)
    music_db = _db(mix)
    placed = []

    # ambience bed, looped with crossfades under the whole reel
    amb_file = fs.find(amb_q, 20, 180)
    if amb_file:
        try:
            a = _load(amb_file)
            if len(a) > SR * 4:
                bed = np.zeros_like(mix)
                xf = SR * 2
                pos = 0
                while pos < n:
                    seg = a[: min(len(a), n - pos)].copy()
                    ramp = np.linspace(0, 1, min(xf, len(seg)))[:, None]
                    seg[: len(ramp)] *= ramp
                    if pos + len(seg) < n:
                        seg[-len(ramp):] *= ramp[::-1]
                    bed[pos:pos + len(seg)] += seg
                    if pos + len(seg) >= n:
                        break
                    pos += max(SR, len(seg) - xf)
                bed = _fit_level(bed, music_db + float(sc.get("ambience_db", -6)))
                fade = np.clip(np.arange(n) / (SR * 0.3), 0, 1) * np.clip((n - np.arange(n)) / (SR * 3.0), 0, 1)
                mix += bed * fade[:, None]
                placed.append(f"ambience: {amb_q}")
        except (subprocess.CalledProcessError, ValueError) as e:
            log(NAME, f"ambience unusable ({str(e)[:80]})")

    # scene effects, starting a moment after their cut begins
    for i, q in cues:
        f = fs.find(q, 0.5, 8)
        if not f:
            continue
        try:
            fx = _load(f, seconds=6)
        except subprocess.CalledProcessError:
            continue
        at = int((lead + shot_starts[i] + (0.0 if q == OPENING_STING else 0.25)) * SR)
        if at >= n:
            continue
        fx = fx[: n - at]
        k = min(len(fx), int(0.4 * SR))
        fx[-k:] *= np.linspace(1, 0, k)[:, None]
        fx = _fit_level(fx, music_db + float(sc.get("effects_db", 5)) + (2.0 if q == OPENING_STING else 0.0))
        mix[at:at + len(fx)] += fx
        placed.append(f"cut {i + 1}: {q}")

    if not placed:
        log(NAME, "no sound effects found this time; using the score alone")
        return score
    peak = float(np.max(np.abs(mix))) or 1.0
    out = (mix * min(1.0, 0.89 / peak) * 32767).astype(np.int16)
    dest = score.with_name("score_sfx.wav")
    with wave.open(str(dest), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(out.tobytes())
    log(NAME, f"{len(placed)} real sound layers from Freesound (CC0): " + "; ".join(placed[:6])
              + (" …" if len(placed) > 6 else ""))
    return dest
