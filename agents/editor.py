"""Editor agent: cuts footage to the narration, adds subtitles, sound design and masters.

Output: 1080x1920, 30 fps, H.264 High (CRF 18, slow preset), AAC 256 kbps 48 kHz stereo,
loudness-normalised to -14 LUFS (Instagram's playback target), faststart for streaming.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .common import Context, log

NAME = "Editor"
XFADE = 0.35  # seconds of crossfade between scenes


# ── timing ──────────────────────────────────────────────────────────────
def _spoken_tokens(text: str) -> list[str]:
    return [t for t in text.split() if re.search(r"\w", t)]


def scene_boundaries(scenes: list[dict], words: list[dict], total: float, lead: float) -> list[float]:
    """Duration of each scene so cuts land where that scene's narration begins."""
    counts = [len(_spoken_tokens(s["narration"])) for s in scenes]
    n_script, n_audio = sum(counts), len(words)
    starts = [0.0]
    cum = 0
    for c in counts[:-1]:
        cum += c
        idx = round(cum * n_audio / max(1, n_script))  # tolerant to tokenisation differences
        idx = min(max(idx, 0), n_audio - 1)
        starts.append(lead + words[idx]["start"] - 0.15)
    starts.append(total)
    durs = [max(1.2, b - a) for a, b in zip(starts, starts[1:])]
    durs[-1] += total - sum(durs)  # absorb rounding so the cut matches exactly
    return durs


# ── subtitles ───────────────────────────────────────────────────────────
def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_color(hex_rgb: str) -> str:
    h = hex_rgb.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _norm(t: str) -> str:
    return re.sub(r"[^\w]", "", t.lower())


def align_display(words: list[dict], script_text: str) -> list[dict]:
    """Put the script's real words (with punctuation) onto the TTS timings.

    The voice engine reports bare words; subtitles read better with "?" and "..." and
    break more naturally at sentence ends. Falls back to the bare words where they differ.
    """
    tokens = _spoken_tokens(script_text)
    out, j = [], 0
    for w in words:
        disp = w["word"]
        # look a few tokens ahead in case the engine merged/split something
        for k in range(j, min(j + 4, len(tokens))):
            if _norm(tokens[k]) == _norm(w["word"]):
                disp, j = tokens[k], k + 1
                break
        out.append({**w, "word": disp})
    return out


def build_subtitles(ctx: Context, words: list[dict], hook: str, lead: float, path: Path,
                    script_text: str = "") -> None:
    v = ctx.cfg["video"]
    if script_text:
        words = align_display(words, script_text)
    hook_n = len(_spoken_tokens(hook))
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < len(words):
        j = i
        while j < len(words) and j - i < 3:
            j += 1
            if re.search(r"[.!?,;:]$", words[j - 1]["word"]):
                break
        chunks.append((i, j))
        i = j

    lines = []
    for k, (a, b) in enumerate(chunks):
        start = lead + words[a]["start"]
        end = lead + (words[chunks[k + 1][0]]["start"] if k + 1 < len(chunks) else words[b - 1]["end"] + 0.6)
        text = " ".join(w["word"] for w in words[a:b]).upper()
        text = text.replace("\\", "").replace("{", "(").replace("}", ")")
        style = "Hook" if a < hook_n else "Main"
        # tiny pop-in: scale from 92% to 100% over 120 ms
        lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,"
                     f"{{\\fscx92\\fscy92\\t(0,120,\\fscx100\\fscy100)}}{text}")

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {v['width']}
PlayResY: {v['height']}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Main,{v['subtitle_font']},{v['subtitle_size']},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,6,3,2,90,90,640,1
Style: Hook,{v['subtitle_font']},{int(v['subtitle_size'] * 1.08)},{_ass_color(v['hook_color'])},&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,6,3,2,90,90,640,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


# ── render ──────────────────────────────────────────────────────────────
def render(ctx: Context, clips: list[Path], durations: list[float], voice: Path | None,
           subs: Path, total: float, out: Path) -> Path:
    v = ctx.cfg["video"]
    w, h, fps = v["width"], v["height"], v["fps"]
    lead = v["lead_in_seconds"]
    n = len(clips)

    args = ["ffmpeg", "-y", "-v", "error", "-stats"]
    for c in clips:
        args += ["-stream_loop", "-1", "-i", str(c)]  # loop only if a clip is shorter than its scene
    vi = n
    if voice:
        args += ["-i", str(voice)]
        vi += 1
    drone_idx = vi
    args += ["-f", "lavfi", "-i", f"anoisesrc=color=brown:amplitude=0.6:sample_rate=48000:duration={total:.3f}"]
    args += ["-f", "lavfi", "-i", f"sine=frequency=55:sample_rate=48000:duration={total:.3f}"]

    f = []
    # Each clip: trim to its scene (+crossfade overlap), cover-fit to 1080x1920 with
    # high-quality Lanczos scaling, conform fps, apply the grade.
    for i, d in enumerate(durations):
        length = d + (XFADE if i < n - 1 else 0)
        grain = ",noise=alls=4:allf=t" if v.get("grain") else ""
        f.append(f"[{i}:v]trim=duration={length:.3f},setpts=PTS-STARTPTS,"
                 f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
                 f"crop={w}:{h},fps={fps},setsar=1,{v['grade']}{grain},format=yuv420p[v{i}]")
    # Chain crossfades; offset k = sum of scene durations up to k → cuts land on narration.
    prev, offset = "v0", 0.0
    for i in range(1, n):
        offset += durations[i - 1]
        f.append(f"[{prev}][v{i}]xfade=transition=fade:duration={XFADE}:offset={offset:.3f}[x{i}]")
        prev = f"x{i}"
    subs_path = str(subs).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    f.append(f"[{prev}]vignette=PI/5,fade=t=in:st=0:d=0.3,"
             f"fade=t=out:st={total - 0.6:.3f}:d=0.6,subtitles='{subs_path}'[vout]")

    # Sound design: narration on top of a low brown-noise drone and a slow 55 Hz pulse.
    dv = v["drone_volume"]
    f.append(f"[{drone_idx}:a]lowpass=f=180,volume={dv},afade=t=in:d=1.5,"
             f"afade=t=out:st={max(0, total - 1.5):.3f}:d=1.5,aformat=channel_layouts=stereo[drone]")
    f.append(f"[{drone_idx + 1}:a]volume={dv * 0.5:.3f},tremolo=f=0.25:d=0.7,"
             f"aformat=channel_layouts=stereo[pulse]")
    mix_inputs = "[drone][pulse]"
    count = 2
    if voice:
        ms = int(lead * 1000)
        f.append(f"[{n}:a]adelay={ms}|{ms},apad,aformat=channel_layouts=stereo[vo]")
        mix_inputs = "[vo]" + mix_inputs
        count = 3
    f.append(f"{mix_inputs}amix=inputs={count}:duration=longest:normalize=0,"
             f"loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")

    args += ["-filter_complex", ";".join(f), "-map", "[vout]", "-map", "[aout]",
             "-t", f"{total:.3f}",
             "-c:v", "libx264", "-preset", os.getenv("X264_PRESET", "slow"), "-crf", "18",
             "-maxrate", "20M", "-bufsize", "40M", "-profile:v", "high",
             "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps * 2),
             "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2",
             "-movflags", "+faststart", str(out)]
    log(NAME, f"rendering {n} scenes · {total:.1f}s · 1080x1920 · CRF 18 …")
    subprocess.run(args, check=True)
    log(NAME, f"rendered → {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
