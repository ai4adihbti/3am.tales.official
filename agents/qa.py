"""QA agent: the last check before anything is posted.

Inspects the finished reel itself (not just what the other agents report):
- length within range, video is 1080x1920, audio present
- loudness close to Instagram's -14 LUFS
- no long black stretches, no long dead-silent stretches in the middle
- not too many illustrations had to be reused because drawing failed
If a hard check fails, the reel is NOT posted (it's still saved so you can see what went wrong).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .common import Context, log

NAME = "QA"


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration:stream=codec_type,width,height,sample_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _analyse(path: Path) -> tuple[float | None, list[tuple[float, float]], list[tuple[float, float]]]:
    p = subprocess.run(["ffmpeg", "-nostats", "-i", str(path),
                        "-vf", "blackdetect=d=1.2:pix_th=0.08", "-af", "ebur128,silencedetect=noise=-45dB:d=3.5",
                        "-f", "null", "-"], capture_output=True, text=True)
    err = p.stderr
    lufs = None
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", err)
    if m:
        lufs = float(m[-1])
    black = [(float(a), float(b)) for a, b in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", err)]
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", err)]
    silences = list(zip(starts, ends + [None] * (len(starts) - len(ends))))
    return lufs, black, silences


def check(ctx: Context, reel: Path, script: dict, total: float) -> dict:
    q = ctx.cfg.get("qa", {})
    v = ctx.cfg["video"]
    errors, warnings = [], []

    info = _probe(reel)
    dur = float(info["format"]["duration"])
    vids = [s for s in info["streams"] if s["codec_type"] == "video"]
    auds = [s for s in info["streams"] if s["codec_type"] == "audio"]
    if not vids or (vids[0].get("width"), vids[0].get("height")) != (v["width"], v["height"]):
        errors.append(f"video size is not {v['width']}x{v['height']}")
    if not auds:
        errors.append("the reel has no audio")
    if not (q.get("min_seconds", 60) <= dur <= q.get("max_seconds", 240)):
        errors.append(f"length {dur:.0f}s is outside {q.get('min_seconds', 60)}-{q.get('max_seconds', 240)}s")

    lufs, black, silences = _analyse(reel)
    if lufs is None:
        warnings.append("could not measure loudness")
    elif abs(lufs + 14) > 2.5:
        warnings.append(f"loudness {lufs:.1f} LUFS (target -14)")
    end_zone = dur - v["tail_seconds"] - 0.5
    mid_black = [(a, b) for a, b in black if a < end_zone and b - a > 1.2]
    if mid_black:
        errors.append(f"black screen for {mid_black[0][1] - mid_black[0][0]:.1f}s at {mid_black[0][0]:.0f}s")
    mid_silence = [(a, b) for a, b in silences if a > 1 and (b or dur) < end_zone]
    if mid_silence:
        a, b = mid_silence[0]
        warnings.append(f"{(b or dur) - a:.1f}s of near-silence at {a:.0f}s")

    images = script.get("images") or []
    failed = ctx.notes.get("images_failed", [])
    if images and len(failed) / len(images) > q.get("max_reused_image_ratio", 0.35):
        errors.append(f"{len(failed)} of {len(images)} illustrations failed to draw")
    elif failed:
        warnings.append(f"{len(failed)} illustration(s) failed and were replaced by neighbours")
    if ctx.notes.get("image_providers") == ["pollinations"]:
        warnings.append("illustrations came from the backup generator (Cloudflare not used): characters may vary")
    if ctx.notes.get("voice_engine") == "edge":
        warnings.append("Microsoft backup voices were used (Gemini/Sarvam voices unavailable)")

    report = {"ok": not errors, "errors": errors, "warnings": warnings, "duration_s": round(dur, 1),
              "loudness_lufs": lufs, "voice_engine": ctx.notes.get("voice_engine"),
              "image_providers": ctx.notes.get("image_providers"), "redrawn": ctx.notes.get("redrawn", [])}
    if errors:
        log(NAME, "FAILED: " + "; ".join(errors))
    else:
        log(NAME, f"passed · {dur:.0f}s · {lufs if lufs is not None else '?'} LUFS"
                  + (" · notes: " + "; ".join(warnings) if warnings else ""))
    return report
