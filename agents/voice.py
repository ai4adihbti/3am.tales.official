"""Voice agent: casts and performs the two-character dialogue.

1st choice: Google Gemini TTS (uses your GEMINI_API_KEY). Each scene is performed in one
request with both characters, so the acting flows naturally, with emotion, whispers,
sighs and pauses. Line timings inside a scene are found by detecting the pauses between turns.

Fallback: Microsoft's free Hindi neural voices (hi-IN Madhur / Swara via edge-tts), one line at
a time, with pitch/speed shaping per character and an eerie effect for spirits.

Everything is then mastered: 48 kHz, low-cut, warmth/presence EQ, gentle compression, subtle room.
"""
from __future__ import annotations

import asyncio
import base64
import os
import re
import subprocess
import time
from pathlib import Path

import requests

from .common import Context, ffprobe_duration, log
from .writer import strip_tags

NAME = "Voice"
TTS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"

MASTER_CHAIN = ",".join([
    "highpass=f=70",
    "equalizer=f=180:t=q:w=1:g=2",          # warmth / chest
    "equalizer=f=3000:t=q:w=1.2:g=2",       # presence / clarity
    "equalizer=f=7500:t=q:w=1.5:g=-1.5",    # tame harshness
    "acompressor=threshold=-20dB:ratio=2.5:attack=10:release=150:makeup=2",
    "aecho=0.85:0.5:35|70:0.10|0.05",       # subtle room, keeps it intimate
])
ENTITY_FX = "asetrate=42240,aresample=48000,atempo=1.136,aecho=0.8:0.7:60|120:0.35|0.2,lowpass=f=5000"

EDGE_SHAPE = {  # voice, rate, pitch
    "young_male": ("edge_male", "+4%", "+0Hz"), "adult_male": ("edge_male", "-2%", "-4Hz"),
    "old_male": ("edge_male", "-10%", "-10Hz"), "young_female": ("edge_female", "+4%", "+2Hz"),
    "adult_female": ("edge_female", "+0%", "-3Hz"), "old_female": ("edge_female", "-10%", "-8Hz"),
}


class TTSFailed(RuntimeError):
    pass


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=True)


def _to_wav48(src: Path, dest: Path, extra_af: str = "") -> None:
    af = "aresample=48000" + ("," + extra_af if extra_af else "")
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-af", af, "-ar", "48000", "-ac", "2",
          "-c:a", "pcm_s16le", str(dest)])


def _silence(dest: Path, seconds: float) -> None:
    _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
          f"anullsrc=r=48000:cl=stereo", "-t", f"{seconds:.3f}", "-c:a", "pcm_s16le", str(dest)])


def _concat(parts: list[Path], dest: Path, workdir: Path) -> None:
    lst = workdir / f"{dest.stem}_list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts), encoding="utf-8")
    _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
          "-c:a", "pcm_s16le", str(dest)])


# ── casting ─────────────────────────────────────────────────────────────
def cast(ctx: Context, characters: list[dict]) -> dict[str, dict]:
    v = ctx.cfg["voice"]
    used: set[str] = set()
    out = {}
    for c in characters:
        pool = v["gemini_voices"].get(c["voice_type"], ["Puck"])
        voice = next((p for p in pool if p not in used), pool[0])
        used.add(voice)
        out[c["id"]] = {**c, "gemini_voice": voice, "label": re.sub(r"[^A-Za-z]", "", c.get("name", "")) or c["id"]}
    if out.get("A", {}).get("label") == out.get("B", {}).get("label"):
        out["B"]["label"] += "Two"
    log(NAME, "cast: " + ", ".join(f"{c['name']} → {c['gemini_voice']}" for c in out.values()))
    return out


# ── Gemini TTS ──────────────────────────────────────────────────────────
def _gemini_scene(model: str, key: str, scene: dict, cast_map: dict, dest_raw: Path) -> None:
    content = []
    for ln in scene["lines"]:
        ch = cast_map[ln["speaker"]]
        style = ln.get("style", "")
        if ch.get("is_entity"):
            style = (style + ", hollow and otherworldly").strip(", ")
        content.append({"type": "text", "text": ln["hi"],
                        "annotations": [{"type": "speech_metadata", "speaker": ch["label"], "style": style}]})
    body = {
        "model": model,
        "input": [{"type": "user_input", "content": content}],
        "response_format": {"type": "audio"},
        "generation_config": {"speech_config": {
            "mode": "conversational",
            "speakers": [{"speaker": c["label"], "voice": c["gemini_voice"]} for c in cast_map.values()]}},
    }
    last = ""
    for attempt in range(1, 5):
        try:
            r = requests.post(TTS_URL, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                              json=body, timeout=(15, 240))
        except requests.RequestException as e:
            last = f"network {type(e).__name__}"
            time.sleep(10 * attempt)
            continue
        if r.status_code == 200:
            audio = _extract_audio(r.json())
            if audio:
                dest_raw.write_bytes(audio)
                return
            last = "reply had no audio"
            continue
        last = f"HTTP {r.status_code} {r.text[:200]}"
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(70, 20 * attempt))   # free-tier per-minute limits / busy servers
            continue
        break  # 400/401/403/404 → this model won't work
    raise TTSFailed(f"{model}: {last}")


def _extract_audio(data: dict) -> bytes | None:
    found = None
    for step in data.get("steps", []):
        for item in step.get("content", []) or []:
            if item.get("type") == "audio" and item.get("data"):
                found = item
    if not found:
        return None
    raw = base64.b64decode(found["data"])
    if raw[:4] == b"RIFF":
        return raw
    # headerless 16-bit PCM → wrap it
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(found.get("sample_rate") or 24000))
        w.writeframes(raw)
    return buf.getvalue()


def _pauses(path: Path) -> list[tuple[float, float]]:
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-af",
                        "silencedetect=noise=-38dB:d=0.18", "-f", "null", "-"],
                       capture_output=True, text=True)
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", p.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
    return list(zip(starts, ends))


def _split_scene(lines: list[dict], dur: float, pauses: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Find where each line starts/ends inside a scene's audio."""
    weights = [len(strip_tags(ln["hi"])) + 8 for ln in lines]
    total = sum(weights)
    mids = [(a + b) / 2 for a, b in pauses if 0.2 < (a + b) / 2 < dur - 0.2]
    speech_start = pauses[0][1] if pauses and pauses[0][0] <= 0.05 else 0.0
    speech_end = pauses[-1][0] if pauses and pauses[-1][1] >= dur - 0.05 else dur
    bounds, cum, prev = [], 0, speech_start
    for w in weights[:-1]:
        cum += w
        est = speech_start + (speech_end - speech_start) * cum / total
        near = [m for m in mids if m > prev + 0.4 and abs(m - est) <= 1.6]
        b = min(near, key=lambda m: abs(m - est)) if near else max(est, prev + 0.4)
        bounds.append(b)
        prev = b
    edges = [speech_start] + bounds + [speech_end]
    return list(zip(edges, edges[1:]))


def _perform_gemini(ctx: Context, script: dict, cast_map: dict) -> tuple[list[Path], list[list[tuple]]]:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise TTSFailed("GEMINI_API_KEY not set")
    models = list(ctx.cfg["voice"]["gemini_models"])
    scene_files, scene_spans = [], []
    for i, scene in enumerate(script["scenes"]):
        raw = ctx.workdir / f"g_scene{i}.wav"
        errors = []
        while models:
            try:
                _gemini_scene(models[0], key, scene, cast_map, raw)
                break
            except TTSFailed as e:
                errors.append(str(e))
                log(NAME, f"scene {i + 1}: {e}")
                if i == 0 and len(models) > 1:
                    models.pop(0)       # try the next TTS model, keep it for the rest
                    continue
                raise TTSFailed("; ".join(errors))
        if not models:
            raise TTSFailed("; ".join(errors))
        wav = ctx.workdir / f"scene{i}.wav"
        _to_wav48(raw, wav)
        dur = ffprobe_duration(wav)
        scene_files.append(wav)
        scene_spans.append(_split_scene(scene["lines"], dur, _pauses(wav)))
        if i == 0:
            log(NAME, f"Gemini TTS ({models[0]}) working")
        time.sleep(2)  # be gentle with free-tier limits
    return scene_files, scene_spans


# ── Microsoft edge-tts fallback ─────────────────────────────────────────
async def _edge_line(text: str, voice: str, rate: str, pitch: str, dest: Path) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(dest))


def _perform_edge(ctx: Context, script: dict, cast_map: dict) -> tuple[list[Path], list[list[tuple]]]:
    v = ctx.cfg["voice"]
    shapes = {}
    for cid, c in cast_map.items():
        key, rate, pitch = EDGE_SHAPE[c["voice_type"]]
        shapes[cid] = [v[key], rate, pitch]
    if shapes.get("A") and shapes.get("B") and shapes["A"][0] == shapes["B"][0]:
        # same voice for both → push them apart so they're easy to tell apart
        hz = int(re.sub(r"[^\d-]", "", shapes["B"][2]) or 0) - 7
        shapes["B"][2] = f"{hz:+d}Hz"
        shapes["B"][1] = "-5%" if shapes["B"][1].startswith("+") else shapes["B"][1]
    gap = ctx.workdir / "gap_line.wav"
    _silence(gap, 0.32)

    scene_files, scene_spans = [], []
    for i, scene in enumerate(script["scenes"]):
        parts, spans, t = [], [], 0.0
        for j, ln in enumerate(scene["lines"]):
            voice, rate, pitch = shapes[ln["speaker"]]
            mp3 = ctx.workdir / f"e_{i}_{j}.mp3"
            wav = ctx.workdir / f"e_{i}_{j}.wav"
            text = strip_tags(ln["hi"])
            for attempt in range(3):
                try:
                    asyncio.run(_edge_line(text, voice, rate, pitch, mp3))
                    break
                except Exception as e:  # network hiccup
                    if attempt == 2:
                        raise TTSFailed(f"edge-tts failed: {e}")
                    time.sleep(3)
            fx = ENTITY_FX if cast_map[ln["speaker"]].get("is_entity") else ""
            _to_wav48(mp3, wav, fx)
            d = ffprobe_duration(wav)
            spans.append((t, t + d))
            parts += [wav, gap]
            t += d + 0.32
        out = ctx.workdir / f"scene{i}.wav"
        _concat(parts[:-1], out, ctx.workdir)
        scene_files.append(out)
        scene_spans.append(spans)
    log(NAME, "Microsoft Hindi voices used")
    return scene_files, scene_spans


# ── mock (offline test) ─────────────────────────────────────────────────
def _perform_mock(ctx: Context, script: dict) -> tuple[list[Path], list[list[tuple]]]:
    scene_files, scene_spans = [], []
    for i, scene in enumerate(script["scenes"]):
        parts, spans, t = [], [], 0.0
        for j, ln in enumerate(scene["lines"]):
            d = len(ln["roman"].split()) / 2.3 + 0.3
            f = ctx.workdir / f"m_{i}_{j}.wav"
            freq = 190 if ln["speaker"] == "A" else 260
            _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                  f"sine=frequency={freq}:duration={d:.2f}:sample_rate=48000", "-af", "volume=0.12",
                  "-ac", "2", "-c:a", "pcm_s16le", str(f)])
            g = ctx.workdir / f"m_{i}_{j}_gap.wav"
            _silence(g, 0.3)
            spans.append((t, t + d))
            parts += [f, g]
            t += d + 0.3
        out = ctx.workdir / f"scene{i}.wav"
        _concat(parts[:-1], out, ctx.workdir)
        scene_files.append(out)
        scene_spans.append(spans)
    return scene_files, scene_spans


# ── main entry ──────────────────────────────────────────────────────────
def perform(ctx: Context, script: dict) -> tuple[Path, list[dict], list[tuple[float, float]], float]:
    """Returns (mastered dialogue wav, line timeline, scene spans, total seconds)."""
    cast_map = cast(ctx, script["characters"])

    if ctx.mock:
        scene_files, scene_spans = _perform_mock(ctx, script)
        engine = "mock"
    else:
        engine = ctx.cfg["voice"].get("engine", "gemini")
        scene_files = scene_spans = None
        if engine == "gemini":
            try:
                scene_files, scene_spans = _perform_gemini(ctx, script, cast_map)
            except TTSFailed as e:
                log(NAME, f"Gemini TTS unavailable ({e}); switching to Microsoft Hindi voices")
                engine = "edge"
        if scene_files is None:
            scene_files, scene_spans = _perform_edge(ctx, script, cast_map)

    # Join scenes with breathing room and build the absolute timeline.
    gap_s = ctx.cfg["voice"]["gap_between_scenes"]
    gap = ctx.workdir / "gap_scene.wav"
    _silence(gap, gap_s)
    parts, timeline, scenes_abs, t = [], [], [], 0.0
    for i, (f, spans) in enumerate(zip(scene_files, scene_spans)):
        d = ffprobe_duration(f)
        for ln, (a, b) in zip(script["scenes"][i]["lines"], spans):
            timeline.append({"scene": i, "speaker": ln["speaker"], "roman": ln["roman"],
                             "start": round(t + a, 3), "end": round(t + b, 3)})
        scenes_abs.append((t, t + d))
        parts += [f, gap]
        t += d + gap_s
    joined = ctx.workdir / "dialogue_raw.wav"
    _concat(parts[:-1], joined, ctx.workdir)

    mastered = ctx.workdir / "dialogue.wav"
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(joined), "-af", MASTER_CHAIN,
          "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(mastered)])
    total = ffprobe_duration(mastered)
    log(NAME, f"{engine} · {len(timeline)} lines · {total:.1f}s · mastered 48 kHz")
    ctx.notes["voice_engine"] = engine
    return mastered, timeline, scenes_abs, total
