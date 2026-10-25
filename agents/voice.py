"""Voice agent: narrates the script with a free neural voice and masters the audio.

- Microsoft neural voices via edge-tts (free, no API key), with word-level timings
  so subtitles land exactly on each spoken word.
- Mastering chain: 48 kHz resample, low-cut, warmth + presence EQ, de-harsh,
  gentle compression and a subtle room ambience for a "storyteller in the dark" feel.
  Final loudness normalisation happens in the Editor after mixing.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

from .common import Context, ffprobe_duration, log

NAME = "Voice"
TICKS = 10_000_000  # edge-tts reports time in 100-nanosecond units

MASTER_CHAIN = ",".join([
    "aresample=48000",
    "highpass=f=75",
    "equalizer=f=160:t=q:w=1:g=2.5",       # warmth / chest
    "equalizer=f=3200:t=q:w=1.2:g=2",       # presence / clarity
    "equalizer=f=7500:t=q:w=1.5:g=-1.5",    # tame harshness
    "acompressor=threshold=-20dB:ratio=3:attack=8:release=120:makeup=3",
    "aecho=0.85:0.6:40|85:0.16|0.08",       # subtle room, not a cave
])


async def _tts(text: str, voice: str, rate: str, pitch: str, mp3: Path) -> list[dict]:
    import edge_tts  # imported lazily so mock mode doesn't need it

    comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, boundary="WordBoundary")
    words: list[dict] = []
    with open(mp3, "wb") as fh:
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                fh.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / TICKS
                words.append({"word": chunk["text"], "start": start,
                              "end": start + chunk["duration"] / TICKS})
    return words


def _estimate_words(text: str, seconds_per_word: float = 0.38) -> list[dict]:
    out, t = [], 0.0
    for w in text.split():
        dur = seconds_per_word + (0.25 if re.search(r"[.!?]$", w) else 0.1 if w.endswith(",") else 0)
        out.append({"word": re.sub(r"[^\w']", "", w), "start": t, "end": t + seconds_per_word * 0.9})
        t += dur
    return out


def narrate(ctx: Context, text: str) -> tuple[Path | None, list[dict], float]:
    """Returns (mastered wav or None, word timings, narration length in seconds)."""
    v = ctx.cfg["voice"]
    raw = ctx.workdir / "voice_raw.mp3"
    wav = ctx.workdir / "voice.wav"

    if ctx.mock:
        words = _estimate_words(text)
        length = words[-1]["end"] + 0.3
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                        "-i", f"sine=frequency=180:duration={length:.2f}:sample_rate=48000",
                        "-af", "volume=0.15", "-ac", "2", str(wav)], check=True)
        log(NAME, f"mock narration {length:.1f}s")
        return wav, words, length

    try:
        words = asyncio.run(_tts(text, v["name"], v["rate"], v["pitch"], raw))
        if not words or raw.stat().st_size < 1000:
            raise RuntimeError("empty TTS output")
    except Exception as e:  # network hiccup or service change → text-only reel
        log(NAME, f"TTS failed ({e}); falling back to a subtitle-only reel")
        words = _estimate_words(text)
        return None, words, words[-1]["end"] + 0.5

    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-af", MASTER_CHAIN,
                    "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(wav)], check=True)
    length = ffprobe_duration(wav)
    log(NAME, f"{v['name']} · {len(words)} words · {length:.1f}s · mastered 48 kHz")
    return wav, words, length
