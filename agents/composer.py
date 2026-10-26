"""Composer agent: writes an original horror score for every reel, shaped to the story.

Each shot has a mood (calm / uneasy / tense / scare). The score follows it:
- always: a dark cinematic pad moving through Dm → B♭ → Gm → A
- calm/uneasy: sparse, slightly detuned music-box notes
- uneasy+: wind
- tense: high dissonant tremolo + a heartbeat
- scare: a whoosh into a deep boom on the cut
- end card: a final low boom, then fade out
Everything is synthesised here, so there's no copyright risk. Output: 48 kHz stereo WAV.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

from .common import Context, log

NAME = "Composer"
SR = 48000
LEVEL = {"calm": 0.0, "uneasy": 1.0, "tense": 2.0, "scare": 2.0}
CHORDS = [  # Dm, Bb, Gm, A
    [73.42, 110.0, 146.83, 174.61, 220.0],
    [58.27, 87.31, 116.54, 146.83, 174.61],
    [49.0, 73.42, 98.0, 116.54, 146.83],
    [55.0, 82.41, 110.0, 138.59, 164.81],
]
BOX_NOTES = [587.33, 698.46, 880.0, 783.99, 659.25, 932.33, 1174.66]


def _lp(x: np.ndarray, hz: float) -> np.ndarray:
    return sosfilt(butter(2, hz, "low", fs=SR, output="sos"), x)


def _bp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)


def _smooth(x: np.ndarray, seconds: float) -> np.ndarray:
    n = max(1, int(seconds * SR / 100))
    small = x[::100]
    k = np.ones(n) / n
    sm = np.convolve(np.pad(small, (n, n), mode="edge"), k, mode="same")[n:-n]
    return np.interp(np.arange(len(x)), np.arange(len(sm)) * 100, sm).astype(np.float32)


def _pad(t: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    seg = 8.0
    out_l = np.zeros_like(t)
    out_r = np.zeros_like(t)
    n_seg = int(t[-1] // seg) + 2
    for k in range(n_seg):
        chord = CHORDS[k % len(CHORDS)]
        # 2-second crossfade window around each chord
        env = np.clip((t - (k * seg - 1.0)) / 2.0, 0, 1) * np.clip(((k + 1) * seg + 1.0 - t) / 2.0, 0, 1)
        idx = np.nonzero(env)[0]
        if not len(idx):
            continue
        tt = t[idx]
        for f in chord:
            for det, side in ((-0.35, 0), (0.35, 1)):
                wave_ = np.zeros_like(tt)
                for h in range(1, 6):
                    wave_ += np.sin(2 * np.pi * (f + det) * h * tt + rng.uniform(0, 6.28)) / h
                (out_l if side == 0 else out_r)[idx] += wave_ * env[idx]
    lfo = 0.75 + 0.25 * np.sin(2 * np.pi * 0.07 * t)
    return _lp(out_l, 650) * lfo, _lp(out_r, 650) * lfo


def _box(t: np.ndarray, weight: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.zeros_like(t)
    pos = 1.0
    while pos < t[-1] - 2:
        f = rng.choice(BOX_NOTES) * (1 + rng.uniform(-0.006, 0.006))   # slightly out of tune → eerie
        a = int(pos * SR)
        n = int(2.2 * SR)
        tt = np.arange(min(n, len(t) - a)) / SR
        note = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(2 * np.pi * f * 2.01 * tt)) * np.exp(-tt * 2.6)
        out[a:a + len(tt)] += note * weight[a]
        pos += rng.uniform(1.3, 2.6)
    return out


def _heartbeat(t: np.ndarray, weight: np.ndarray) -> np.ndarray:
    out = np.zeros_like(t)
    pos, bpm = 0.5, 68.0
    thump = lambda dur, f: np.sin(2 * np.pi * f * np.arange(int(dur * SR)) / SR) * \
        np.exp(-np.arange(int(dur * SR)) / SR * 22)
    lub, dub = thump(0.25, 52), thump(0.22, 46) * 0.7
    while pos < t[-1] - 1:
        a = int(pos * SR)
        w = weight[a]
        if w > 0.02:
            out[a:a + len(lub)] += lub[: len(out) - a] * w
            b = a + int(0.28 * SR)
            if b < len(out):
                out[b:b + len(dub)] += dub[: len(out) - b] * w
        pos += 60.0 / (bpm + 30 * w)
    return out


def _hit(length: int, rng: np.random.Generator) -> np.ndarray:
    """A whoosh rising for 0.7 s into a deep boom (returned array starts 0.7 s before the cut)."""
    pre = int(0.7 * SR)
    boom_n = int(2.2 * SR)
    tt = np.arange(boom_n) / SR
    sweep = np.sin(2 * np.pi * np.cumsum(90 * np.exp(-tt * 1.6) + 28) / SR)
    boom = sweep * np.exp(-tt * 1.8) + _lp(rng.standard_normal(boom_n), 300) * np.exp(-tt * 9) * 0.6
    swell = _bp(rng.standard_normal(pre), 400, 3000) * np.linspace(0, 1, pre) ** 2.5 * 0.5
    out = np.concatenate([swell, boom])
    return out[:length]


def compose(ctx: Context, shots: list[dict], shot_starts: list[float], voice_len: float,
            lead: float, total: float) -> Path:
    rng = np.random.default_rng()
    n = int(total * SR)
    t = (np.arange(n) / SR).astype(np.float32)

    # mood curve over the whole reel
    level = np.zeros(n, dtype=np.float32)
    starts = [lead + s for s in shot_starts] + [total]
    for i, sh in enumerate(shots):
        a, b = int(starts[i] * SR), int(min(starts[i + 1], total) * SR)
        level[a:b] = LEVEL.get(sh.get("mood", "uneasy"), 1.0)
    level[int(starts[-2] * SR):] = level[int(starts[-2] * SR) - 1] if len(shots) > 1 else 1.0
    level = _smooth(level, 1.6)

    pad_l, pad_r = _pad(t, rng)
    box_w = np.clip(1.0 - level / 1.5, 0, 1) * 0.5
    box = _lp(_box(t, box_w, rng), 5000)
    wind_w = np.clip(level, 0, 1.4)
    wind_l = _bp(rng.standard_normal(n), 280, 1100) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.09 * t)) * wind_w
    wind_r = _bp(rng.standard_normal(n), 280, 1100) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.11 * t + 1)) * wind_w
    tense_w = np.clip(level - 1.0, 0, 1)
    trem = 0.5 + 0.5 * np.sin(2 * np.pi * 6.0 * t)
    strings = (np.sin(2 * np.pi * 1174.66 * t) + np.sin(2 * np.pi * 1244.51 * t)) * trem * tense_w
    heart = _heartbeat(t, tense_w)

    mono_fx = box * 0.55 + strings * 0.05 + heart * 0.9
    left = pad_l * 0.11 + wind_l * 0.10 + mono_fx
    right = pad_r * 0.11 + wind_r * 0.10 + mono_fx

    # boom on every scare cut, and one for the end card
    hits = [starts[i] for i, sh in enumerate(shots) if sh.get("mood") == "scare" and i > 0]
    hits.append(lead + voice_len + 0.4)
    for h in hits:
        a = int((h - 0.7) * SR)
        if a < 0 or a >= n:
            continue
        x = _hit(n - a, rng) * 0.9
        left[a:a + len(x)] += x
        right[a:a + len(x)] += x

    # fade in/out and normalise
    fade_in = np.clip(t / 1.5, 0, 1)
    fade_out = np.clip((total - t) / 2.5, 0, 1)
    stereo = np.stack([left, right], axis=1) * (fade_in * fade_out)[:, None]
    peak = float(np.max(np.abs(stereo))) or 1.0
    stereo = (stereo / peak * 0.89 * 32767).astype(np.int16)

    path = ctx.workdir / "score.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(stereo.tobytes())
    log(NAME, f"original score: {total:.0f}s, {len(hits) - 1} scare hits")
    return path
