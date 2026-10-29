"""Editor agent: assembles the animated shots, voices and score into the finished reel.

- cuts land just before each shot's first line; soft crossfades between shots
- small channel watermark bottom-left, "To Be Continued / Follow For Part N" (or "The End") card
- optional Hinglish subtitles coloured per character (off by default, like the reference style)
- music ducks automatically under the voices; final mix loudness-normalised to -14 LUFS
Output: 1080x1920, 30 fps, H.264 High CRF 18, AAC 128 kbps 48 kHz stereo (Instagram's spec).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from .common import Context, log

NAME = "Editor"


# ── timing ──────────────────────────────────────────────────────────────
def shot_durations(shot_starts: list[float], total: float, lead: float) -> list[float]:
    """On-screen time of each shot: from its first line (minus a beat) to the next shot."""
    starts = [0.0] + [lead + s for s in shot_starts[1:]] + [total]
    durs = [max(1.0, b - a) for a, b in zip(starts, starts[1:])]
    durs[-1] += total - sum(durs)
    return durs


# ── subtitles, watermark, end card (one .ass file) ──────────────────────
def _ass_time(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_color(hex_rgb: str) -> str:
    h = hex_rgb.lstrip("#")
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}".upper()


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _chunks(text: str, max_words: int = 4) -> list[str]:
    out, cur = [], []
    for w in text.split():
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.!?,;:…]$", w):
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def build_overlay(ctx: Context, timeline: list[dict], speakers: list[str], lead: float, total: float,
                  end_card: tuple[str, str], path: Path) -> None:
    v = ctx.cfg["video"]
    W, H = v["width"], v["height"]
    fs = v["subtitle_size"]
    font = v["subtitle_font"]
    serif = v.get("end_card_font", "DejaVu Serif")
    colors = v.get("speaker_colors", ["#FFFFFF", "#FFD34D", "#9BE7FF", "#FFB4A2"])
    base = "&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,5,2,2,80,80,520,1"
    styles = [f"Style: S{i},{font},{fs},{_ass_color(colors[i % len(colors)])},{base}" for i in range(5)]
    styles.append(f"Style: Hook,{font},{int(fs * 1.06)},{_ass_color(v['hook_color'])},{base}")
    styles.append(f"Style: Mark,{font},34,&H00F2F2F2,&H00FFFFFF,&H00000000,&H64000000,0,0,0,0,100,100,1,0,1,2,1,1,40,40,46,1")
    styles.append(f"Style: End,{serif},84,&H00FFFFFF,&H00FFFFFF,&H00000000,&H78000000,0,0,0,0,100,100,1,0,1,3,3,5,60,60,0,1")

    ev = []
    watermark = ctx.cfg["channel"].get("watermark", "")
    if watermark:
        ev.append(f"Dialogue: 2,{_ass_time(0)},{_ass_time(total)},Mark,,0,0,0,,{_esc(watermark)}")

    if v.get("subtitles"):
        order = {sp: i for i, sp in enumerate(speakers)}
        for idx, ln in enumerate(timeline):
            parts = _chunks(ln["roman"])
            weights = [len(p) + 3 for p in parts]
            span = max(0.4, ln["end"] - ln["start"])
            t = lead + ln["start"]
            nxt = lead + timeline[idx + 1]["start"] if idx + 1 < len(timeline) else lead + ln["end"] + 0.8
            for k, (p, wgt) in enumerate(zip(parts, weights)):
                dur = span * wgt / sum(weights)
                end = t + dur if k < len(parts) - 1 else max(t + dur, min(nxt, t + dur + 0.6))
                style = "Hook" if idx == 0 else f"S{order.get(ln['speaker'], 0)}"
                ev.append(f"Dialogue: 0,{_ass_time(t)},{_ass_time(end)},{style},,0,0,0,,"
                          f"{{\\fscx92\\fscy92\\t(0,120,\\fscx100\\fscy100)}}{_esc(p.upper())}")
                t += dur

    top, bottom = end_card
    card_at = max(0.0, total - v["tail_seconds"] + 0.3)
    if top:
        ev.append(f"Dialogue: 3,{_ass_time(card_at)},{_ass_time(total)},End,,0,0,0,,"
                  f"{{\\pos({W // 2},{int(H * 0.17)})\\fad(700,300)}}{_esc(top)}")
    if bottom:
        ev.append(f"Dialogue: 3,{_ass_time(card_at + 0.5)},{_ass_time(total)},End,,0,0,0,,"
                  f"{{\\pos({W // 2},{int(H * 0.80)})\\fs76\\fad(700,300)}}{_esc(bottom)}")

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{chr(10).join(styles)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    path.write_text(header + "\n".join(ev) + "\n", encoding="utf-8")


# ── render ──────────────────────────────────────────────────────────────
def render(ctx: Context, clips: list[Path], durations: list[float], voice: Path, score: Path | None,
           overlay: Path, total: float, out: Path) -> Path:
    v = ctx.cfg["video"]
    w, h, fps = v["width"], v["height"], v["fps"]
    xf = v.get("crossfade", 0.4)
    lead = v["lead_in_seconds"]
    n = len(clips)

    args = ["ffmpeg", "-y", "-v", "error", "-stats"]
    for c in clips:
        args += ["-i", str(c)]
    args += ["-i", str(voice)]
    vi = n
    if score:
        args += ["-i", str(score)]

    f = []
    for i, d in enumerate(durations):
        length = d + (xf if i < n - 1 else 0)
        grain = ",noise=alls=3:allf=t" if v.get("grain") else ""
        f.append(f"[{i}:v]tpad=stop_mode=clone:stop_duration=2,trim=duration={length:.3f},setpts=PTS-STARTPTS,"
                 f"scale={w}:{h}:flags=lanczos,fps={fps},setsar=1,{v['grade']}{grain},format=yuv420p[v{i}]")
    prev, offset = "v0", 0.0
    for i in range(1, n):
        offset += durations[i - 1]
        f.append(f"[{prev}][v{i}]xfade=transition=fade:duration={xf}:offset={offset:.3f}[x{i}]")
        prev = f"x{i}"
    ov = str(overlay).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
    card_at = max(0.0, total - v["tail_seconds"])
    f.append(f"[{prev}]fade=t=in:st=0:d=0.4,"
             f"eq=brightness=-0.10:enable='gte(t,{card_at:.2f})',"
             f"fade=t=out:st={total - 0.5:.3f}:d=0.5,subtitles='{ov}'[vout]")

    ms = int(lead * 1000)
    if score:
        mv = ctx.cfg["music"]["volume"]
        f.append(f"[{vi}:a]adelay={ms}|{ms},apad,aformat=channel_layouts=stereo,asplit=2[vo][key]")
        f.append(f"[{vi + 1}:a]aformat=channel_layouts=stereo,volume={mv}[mus]")
        f.append(f"[mus][key]sidechaincompress=threshold=0.04:ratio={ctx.cfg['music'].get('duck_ratio', 3)}:"
                 "attack=60:release=600:makeup=1[duck]")
        f.append("[vo][duck]amix=inputs=2:duration=longest:normalize=0,"
                 "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")
    else:
        f.append(f"[{vi}:a]adelay={ms}|{ms},apad,aformat=channel_layouts=stereo,"
                 "loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]")

    args += ["-filter_complex", ";".join(f), "-map", "[vout]", "-map", "[aout]",
             "-t", f"{total:.3f}",
             "-c:v", "libx264", "-preset", os.getenv("X264_PRESET", "medium"), "-crf", "19",
             "-maxrate", "4M", "-bufsize", "8M", "-profile:v", "high",
             "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps * 2),
             "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
             "-use_editlist", "0", "-movflags", "+faststart", str(out)]   # Instagram: no edit lists, moov first
    log(NAME, f"rendering {n} shots · {total:.1f}s · 1080x1920 · CRF 18 …")
    subprocess.run(args, check=True)
    log(NAME, f"rendered → {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
