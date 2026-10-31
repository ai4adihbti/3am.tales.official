"""Editor agent: assembles the animated shots, voices and score into the finished reel.

- cuts land just before each shot's first line; soft crossfades between shots
- small channel watermark bottom-left, "To Be Continued / Follow For Part N" (or "The End") card
- optional Hinglish subtitles coloured per character (off by default, like the reference style)
- music ducks automatically under the voices; final mix loudness-normalised to -14 LUFS
Output: 1080x1920, 30 fps, H.264 High CRF 19, AAC 128 kbps 48 kHz stereo (Instagram's spec).
"""
from __future__ import annotations

import math
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


def _pauses(voice: Path) -> list[float]:
    """Mid-points (seconds) of the breaths/pauses in the voice track: the natural places to cut a picture."""
    import numpy as np
    sr = 8000
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(voice), "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    hop = sr // 50                                         # 20 ms frames
    n = len(x) // hop
    if n < 10:
        return []
    rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-7
    db = 20 * np.log10(rms)
    quiet = db < np.percentile(db, 95) - 26
    out, run = [], 0
    for i, q in enumerate(list(quiet) + [False]):
        if q:
            run += 1
        else:
            if run >= 5:                                   # ≥ 100 ms of quiet = a breath between phrases
                out.append((i - run / 2) * 0.02)
            run = 0
    return out


ALT_MOTION = {"zoom_in": "pan_right", "zoom_out": "zoom_in", "pan_left": "zoom_in", "pan_right": "zoom_in",
              "pan_up": "zoom_in", "pan_down": "zoom_in", "push_fast": "zoom_in", "hold": "zoom_in"}


def split_long_shots(ctx: Context, script: dict, shot_starts: list[float], voice: Path,
                     voice_len: float) -> list[float]:
    """No picture stays on screen longer than ~6 s: a long narration line is cut into 2-3 pieces at the
    storyteller's breaths, each piece a new camera angle on the same drawing (a punch-in on the face or subject,
    then a different camera move), like an editor cutting coverage. Updates script["shots"]; returns new starts."""
    max_len = float(ctx.cfg["video"].get("max_shot_seconds", 6.0))
    if max_len <= 0 or not script["shots"]:
        return shot_starts
    try:
        pauses = _pauses(voice)
    except (subprocess.CalledProcessError, ValueError):
        pauses = []
    shots, starts, splits = [], [], 0
    ends = shot_starts[1:] + [voice_len]
    for sh, a, b in zip(script["shots"], shot_starts, ends):
        shots.append(sh)
        starts.append(a)
        d = b - a
        if d <= max_len:
            continue
        n = max(2, math.ceil(d / 4.5))
        cuts, prev = [], a
        for k in range(1, n):
            target = a + d * k / n
            ok = [p for p in pauses if abs(p - target) <= 1.6 and p - prev >= 1.8 and b - p >= 1.8
                  and p - prev <= max_len + 0.3]
            at = min(ok, key=lambda p: abs(p - target)) if ok else target
            if at - prev >= 1.8 and b - at >= 1.8:
                cuts.append(at)
                prev = at
        motion = sh.get("motion", "zoom_in")
        for k, at in enumerate(cuts):
            motion = ALT_MOTION.get(motion, "zoom_in")
            shots.append({**sh, "lines": [], "motion": motion, "punch": k % 2 == 0,
                          "mood": "tense" if sh.get("mood") == "scare" else sh.get("mood"), "split_of": len(starts) - 1})
            starts.append(round(at, 3))
            splits += 1
    if splits:
        script["shots"] = shots
        log(NAME, f"{splits} extra cuts at the storyteller's breaths: no picture holds longer than ~{max_len:.0f}s "
                  f"({len(shots)} cuts now)")
    return starts


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
                  end_card: tuple[str, str], path: Path, hook_text: str = "") -> None:
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
    styles.append(f"Style: End,{serif},96,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,1,0,1,5,4,5,60,60,0,1")
    styles.append(f"Style: EndSub,{font},62,{_ass_color(v.get('end_sub_color', '#FFD34D'))},&H00FFFFFF,&H00000000,"
                  "&H96000000,-1,0,0,0,100,100,1,0,1,4,3,5,70,70,0,1")
    styles.append(f"Style: HookBig,{font},{int(v.get('hook_text_size', 92))},&H00FFFFFF,&H00FFFFFF,&H00000000,"
                  "&HA0000000,-1,0,0,0,100,100,1,0,1,7,4,5,90,90,0,1")

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

    if hook_text and v.get("hook_text", True):
        # the first 3 seconds: a big on-screen hook for the many viewers who scroll with the sound off
        hook_red = _ass_color(v.get("hook_color", "#FF3B3B"))
        words = _esc(hook_text).split()
        if len(words) > 2:     # last word in red, like the reference thumbnails
            words[-1] = f"{{\\c{hook_red}}}{words[-1]}"
        ev.append(f"Dialogue: 4,{_ass_time(0.08)},{_ass_time(float(v.get('hook_text_seconds', 3.0)))},HookBig,,0,0,0,,"
                  f"{{\\pos({W // 2},{int(H * 0.21)})\\fad(60,260)\\fscx125\\fscy125\\t(0,180,\\fscx100\\fscy100)}}"
                  + " ".join(words))

    top, bottom = end_card
    card_at = max(0.0, total - v["tail_seconds"] + 0.15)
    # centred, bold and well above Instagram's caption/buttons area (the old bottom line sat under it)
    if top:
        ev.append(f"Dialogue: 3,{_ass_time(card_at)},{_ass_time(total)},End,,0,0,0,,"
                  f"{{\\pos({W // 2},{int(H * 0.40)})\\fad(300,200)}}{_esc(top)}")
    if bottom:
        ev.append(f"Dialogue: 3,{_ass_time(card_at + 0.3)},{_ass_time(total)},EndSub,,0,0,0,,"
                  f"{{\\pos({W // 2},{int(H * 0.52)})\\fad(300,200)}}{_esc(bottom)}")

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
# -14 LUFS for Instagram, with 2 dB of true-peak headroom plus a limiter, so Instagram's/YouTube's re-encode
# can't push it into clipping (the downloaded old reel peaked at +0.1 dBFS)
MASTER = "loudnorm=I=-14:TP=-2.0:LRA=11,aresample=48000,alimiter=limit=0.8:attack=3:release=60:level=disabled"
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
    # no fade-in: the reel opens straight on the cold-open picture (the first frame must already grab)
    f.append(f"[{prev}]eq=brightness=-0.22:saturation=0.8:enable='gte(t,{card_at:.2f})',"
             f"fade=t=out:st={total - 0.5:.3f}:d=0.5,subtitles='{ov}'[vout]")

    ms = int(lead * 1000)
    if score:
        mv = ctx.cfg["music"]["volume"]
        f.append(f"[{vi}:a]adelay={ms}|{ms},apad,aformat=channel_layouts=stereo,asplit=2[vo][key]")
        # music: less low-end mud under the voice, and wider so the storyteller sits alone in the centre
        f.append(f"[{vi + 1}:a]aformat=channel_layouts=stereo,volume={mv},highpass=f=45,"
                 f"bass=g={float(ctx.cfg['music'].get('bass_db', -4))}:f=110,"
                 "stereowiden=delay=14:feedback=0.12:crossfeed=0.2:drymix=0.9,extrastereo=m=1.5[mus]")
        f.append(f"[mus][key]sidechaincompress=threshold=0.04:ratio={ctx.cfg['music'].get('duck_ratio', 3)}:"
                 "attack=60:release=600:makeup=1[duck]")
        f.append("[vo][duck]amix=inputs=2:duration=longest:normalize=0," + MASTER + "[aout]")
    else:
        f.append(f"[{vi}:a]adelay={ms}|{ms},apad,aformat=channel_layouts=stereo," + MASTER + "[aout]")

    args += ["-filter_complex", ";".join(f), "-map", "[vout]", "-map", "[aout]",
             "-t", f"{total:.3f}",
             "-c:v", "libx264", "-preset", os.getenv("X264_PRESET", "medium"), "-crf", "19",
             "-maxrate", "4M", "-bufsize", "8M", "-profile:v", "high",
             "-pix_fmt", "yuv420p", "-r", str(fps), "-g", str(fps * 2),
             "-c:a", "aac", "-b:a", str(v.get("audio_bitrate", "128k")), "-ar", "48000", "-ac", "2",
             "-use_editlist", "0", "-movflags", "+faststart", str(out)]   # Instagram: no edit lists, moov first
    log(NAME, f"rendering {n} shots · {total:.1f}s · 1080x1920 · CRF 18 …")
    subprocess.run(args, check=True)
    log(NAME, f"rendered → {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return out
