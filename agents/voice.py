"""Voice agent: casts a voice for the narrator and every character, and performs the part.

1st choice: Google Gemini TTS (uses your GEMINI_API_KEY). Consecutive lines are performed together
(max two speakers per request, the API's limit) so the acting flows naturally, with emotion,
whispers, sighs and pauses. Line timings are found by detecting the pauses between turns.

Fallback: Microsoft's free Hindi neural voices (hi-IN Madhur / Swara via edge-tts), one line at
a time, each character shaped with its own pitch/speed, plus an eerie effect for spirits.

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

from .common import Context, ffprobe_duration, is_hard_quota, log
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

EDGE_SHAPE = {  # voice, rate %, pitch Hz
    "narrator": ("edge_male", -8, -9), "narrator_female": ("edge_female", -6, -3),
    "child_boy": ("edge_female", 6, 34), "child_girl": ("edge_female", 6, 42),
    "young_male": ("edge_male", 4, 2), "adult_male": ("edge_male", -2, -4),
    "old_male": ("edge_male", -12, -14), "young_female": ("edge_female", 4, 3),
    "adult_female": ("edge_female", 0, -3), "old_female": ("edge_female", -12, -10),
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


# ── helpers ─────────────────────────────────────────────────────────────
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




# ── casting ─────────────────────────────────────────────────────────────
def cast(ctx: Context, characters: list[dict], narrator_gender: str = "male") -> dict[str, dict]:
    """Assign a distinct voice to the narrator ("N") and each character."""
    v = ctx.cfg["voice"]
    fem = narrator_gender == "female"
    nv = v.get("narrator_gemini_voice", "Charon")
    narrator_voice = (nv.get("female" if fem else "male", "Charon") if isinstance(nv, dict) else nv)
    used = {narrator_voice}
    out = {"N": {"id": "N", "name": "Narrator", "voice_type": "narrator", "is_entity": False,
                 "gemini_voice": narrator_voice, "label": "Narrator"}}
    edge_used: set[tuple] = set()
    for c in characters:
        pool = v["gemini_voices"].get(c["voice_type"], ["Puck"])
        voice = next((p for p in pool if p not in used), next((p for q in v["gemini_voices"].values()
                                                                for p in q if p not in used), pool[0]))
        used.add(voice)
        label = re.sub(r"[^A-Za-z]", "", c.get("name", "")) or c["id"]
        out[c["id"]] = {**c, "gemini_voice": voice, "label": label}
    # Sarvam: a distinct native Hindi voice per speaker
    sn = v.get("sarvam_narrator", "varun")
    sn = sn.get("female" if fem else "male", "varun") if isinstance(sn, dict) else sn
    s_used = {sn}
    out["N"]["sarvam"] = sn
    if fem:
        out["N"]["voice_type"] = "narrator_female"
    all_sarvam = [x for q in v.get("sarvam_voices", {}).values() for x in q]
    for cid, c in out.items():
        if cid == "N":
            continue
        pool = v.get("sarvam_voices", {}).get(c["voice_type"], ["shubh"])
        pick = next((x for x in pool if x not in s_used), next((x for x in all_sarvam if x not in s_used), pool[0]))
        s_used.add(pick)
        c["sarvam"] = pick
    # Microsoft fallback: same two base voices, so push every speaker to a distinct pitch/speed
    for cid, c in out.items():
        key, rate, pitch = EDGE_SHAPE[c["voice_type"]]
        while (key, pitch) in edge_used:
            pitch -= 6 if pitch <= 0 else -6
            rate -= 3
        edge_used.add((key, pitch))
        c["edge"] = (v[key], f"{rate:+d}%", f"{pitch:+d}Hz")
    log(NAME, "cast: " + ", ".join(f"{c['name']} → {c['gemini_voice']}" for c in out.values()))
    return out


# ── Gemini TTS ──────────────────────────────────────────────────────────
def _gemini_chunk(model: str, key: str, lines: list[dict], cast_map: dict, dest_raw: Path) -> None:
    speakers = list(dict.fromkeys(ln["speaker"] for ln in lines))
    content = []
    for ln in lines:
        ch = cast_map[ln["speaker"]]
        style = ln.get("style", "")
        if ch.get("is_entity"):
            style = (style + ", hollow and otherworldly").strip(", ")
        if ln["speaker"] == "N":
            style = (style or "deep, slow, gripping storyteller")
        if ch.get("voice_type", "").startswith("child"):
            style = (style + ", voice of a small child").strip(", ")
        content.append({"type": "text", "text": ln["hi"],
                        "annotations": [{"type": "speech_metadata", "speaker": ch["label"], "style": style}]})
    if len(speakers) == 1:
        speech_config = [{"voice": cast_map[speakers[0]]["gemini_voice"]}]
        for item in content:  # single-speaker requests don't take speaker names
            item["annotations"][0].pop("speaker", None)
    else:
        speech_config = {
                         "speakers": [{"speaker": cast_map[s]["label"], "voice": cast_map[s]["gemini_voice"]}
                                      for s in speakers]}
    body = {"model": model, "input": [{"type": "user_input", "content": content}],
            "response_format": {"type": "audio"}, "generation_config": {"speech_config": speech_config}}
    last = ""
    for attempt in range(1, 4):
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
        if is_hard_quota(r):
            break
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(min(70, 20 * attempt))
            continue
        break
    raise TTSFailed(f"{model}: {last}")


def _chunks(shots: list[dict]) -> list:
    """Group consecutive lines into TTS requests (max 2 speakers, 8 lines); silent shots stand alone."""
    out, cur = [], []
    for si, sh in enumerate(shots):
        if not sh["lines"]:
            if cur:
                out.append(("speech", cur))
                cur = []
            out.append(("silence", si))
            continue
        for ln in sh["lines"]:
            speakers = {x["speaker"] for _, x in cur} | {ln["speaker"]}
            if cur and (len(speakers) > 2 or len(cur) >= 8):
                out.append(("speech", cur))
                cur = []
            cur.append((si, ln))
    if cur:
        out.append(("speech", cur))
    return out


def _perform_gemini(ctx: Context, script: dict, cast_map: dict) -> list:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise TTSFailed("GEMINI_API_KEY not set")
    models = list(ctx.cfg["voice"]["gemini_models"])
    pieces, n_req = [], 0
    for kind, payload in _chunks(script["shots"]):
        if kind == "silence":
            pieces.append(_silent_piece(ctx, payload))
            continue
        lines = [ln for _, ln in payload]
        raw = ctx.workdir / f"g_{len(pieces):03d}.wav"
        while True:
            try:
                _gemini_chunk(models[0], key, lines, cast_map, raw)
                break
            except TTSFailed as e:
                log(NAME, f"{e}")
                if n_req == 0 and len(models) > 1:
                    models.pop(0)
                    continue
                raise
        n_req += 1
        wav = ctx.workdir / f"g_{len(pieces):03d}_48.wav"
        _to_wav48(raw, wav)
        dur = ffprobe_duration(wav)
        spans = _split_scene(lines, dur, _pauses(wav))
        pieces.append({"wav": wav, "items": [(si, ln, a, b) for (si, ln), (a, b) in zip(payload, spans)]})
        if n_req == 1:
            log(NAME, f"Gemini TTS ({models[0]}) working")
        time.sleep(2)
    return pieces


# ── Sarvam AI (native Indian voices) ────────────────────────────────────
SARVAM_PACE = {"narrator": 0.92, "narrator_female": 0.94, "child_boy": 1.05, "child_girl": 1.05, "old_male": 0.88, "old_female": 0.88}


def _sarvam_line(key: str, model: str, speaker: str, text: str, pace: float, dest: Path) -> None:
    body = {"text": text[:2400], "language_code": "hi-IN", "model": model, "speaker": speaker,
            "pace": pace, "speech_sample_rate": 48000, "output_audio_codec": "wav", "temperature": 0.72}
    last = ""
    for attempt in range(1, 4):
        try:
            r = requests.post("https://api.sarvam.ai/text-to-speech", json=body, timeout=(15, 120),
                              headers={"api-subscription-key": key, "Content-Type": "application/json"})
        except requests.RequestException as e:
            last = f"network {type(e).__name__}"
            time.sleep(4 * attempt)
            continue
        if r.status_code == 200:
            audios = (r.json() or {}).get("audios") or []
            if audios:
                dest.write_bytes(base64.b64decode(audios[0]))
                return
            last = "no audio in reply"
            continue
        last = f"HTTP {r.status_code} {r.text[:200]}"
        if r.status_code in (429, 500, 502, 503):
            time.sleep(6 * attempt)
            continue
        break  # 400/401/403/422: bad key, no credit, or bad request
    raise TTSFailed(f"sarvam: {last}")


def _perform_sarvam(ctx: Context, script: dict, cast_map: dict) -> list:
    key = os.getenv("SARVAM_API_KEY", "").strip()
    if not key:
        raise TTSFailed("SARVAM_API_KEY not set")
    model = ctx.cfg["voice"].get("sarvam_model", "bulbul:v3")
    pieces = []
    for si, sh in enumerate(script["shots"]):
        if not sh["lines"]:
            pieces.append(_silent_piece(ctx, si))
            continue
        for j, ln in enumerate(sh["lines"]):
            ch = cast_map[ln["speaker"]]
            raw = ctx.workdir / f"s_{si:02d}_{j}_raw.wav"
            wav = ctx.workdir / f"s_{si:02d}_{j}.wav"
            pace = SARVAM_PACE.get(ch["voice_type"], 1.0)
            if ln["speaker"] == "N":
                pace = float(ctx.cfg["voice"].get("sarvam_narrator_pace", pace))
            _sarvam_line(key, model, ch["sarvam"], strip_tags(ln["hi"]), pace, raw)
            fx = ENTITY_FX if ch.get("is_entity") else ""
            _to_wav48(raw, wav, ("silenceremove=start_periods=1:start_threshold=-50dB," + fx).rstrip(","))
            pieces.append({"wav": wav, "items": [(si, ln, 0.0, ffprobe_duration(wav))]})
    log(NAME, f"Sarvam {model} voices used")
    return pieces


# ── Microsoft edge-tts fallback ─────────────────────────────────────────
async def _edge_line(text: str, voice: str, rate: str, pitch: str, dest: Path) -> None:
    import edge_tts
    await edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).save(str(dest))


def _perform_edge(ctx: Context, script: dict, cast_map: dict) -> list:
    pieces = []
    for si, sh in enumerate(script["shots"]):
        if not sh["lines"]:
            pieces.append(_silent_piece(ctx, si))
            continue
        for j, ln in enumerate(sh["lines"]):
            voice, rate, pitch = cast_map[ln["speaker"]]["edge"]
            mp3 = ctx.workdir / f"e_{si:02d}_{j}.mp3"
            wav = ctx.workdir / f"e_{si:02d}_{j}.wav"
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
            pieces.append({"wav": wav, "items": [(si, ln, 0.0, ffprobe_duration(wav))]})
    log(NAME, "Microsoft Hindi voices used")
    return pieces


# ── mock (offline test) ─────────────────────────────────────────────────
def _perform_mock(ctx: Context, script: dict, cast_map: dict) -> list:
    freqs = {"N": 140, "C1": 200, "C2": 260, "C3": 170, "C4": 300}
    pieces = []
    for si, sh in enumerate(script["shots"]):
        if not sh["lines"]:
            pieces.append(_silent_piece(ctx, si))
            continue
        for j, ln in enumerate(sh["lines"]):
            d = len(ln["roman"].split()) / 2.3 + 0.3
            f = ctx.workdir / f"m_{si:02d}_{j}.wav"
            _run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                  f"sine=frequency={freqs.get(ln['speaker'], 220)}:duration={d:.2f}:sample_rate=48000",
                  "-af", "volume=0.12", "-ac", "2", "-c:a", "pcm_s16le", str(f)])
            pieces.append({"wav": f, "items": [(si, ln, 0.0, d)]})
    return pieces


def _silent_piece(ctx: Context, shot_index: int) -> dict:
    d = ctx.cfg["voice"]["silent_shot_seconds"]
    f = ctx.workdir / f"silence_{shot_index:02d}.wav"
    _silence(f, d)
    return {"wav": f, "items": [(shot_index, None, 0.0, d)]}


# ── main entry ──────────────────────────────────────────────────────────
def perform(ctx: Context, script: dict) -> tuple[Path, list[dict], list[float], float]:
    """Returns (mastered voice track, line timeline, shot start times, total seconds)."""
    cast_map = cast(ctx, script["characters"], script.get("narrator_gender", "male"))
    if ctx.mock:
        pieces, engine = _perform_mock(ctx, script, cast_map), "mock"
    else:
        vcfg = ctx.cfg["voice"]
        order = list(vcfg.get("engines") or [vcfg.get("engine", "gemini"), "edge"])
        keep = script.get("series_voice_engine")
        if keep in order:   # later parts of a series use the engine Part 1 used, so voices stay the same
            order.remove(keep)
            order.insert(0, keep)
        if "edge" not in order:
            order = list(order) + ["edge"]
        runners = {"gemini": _perform_gemini, "sarvam": _perform_sarvam, "edge": _perform_edge}
        pieces, engine = None, ""
        for engine in order:
            if engine not in runners:
                continue
            try:
                pieces = runners[engine](ctx, script, cast_map)
                break
            except TTSFailed as e:
                log(NAME, f"{engine} voices unavailable ({str(e)[:160]}); trying the next engine")
        if pieces is None:
            raise TTSFailed("No voice engine worked")

    gap_s = ctx.cfg["voice"]["gap_between_lines"]
    gap = ctx.workdir / "gap.wav"
    _silence(gap, gap_s)
    parts, timeline, t = [], [], 0.0
    shot_first: dict[int, float] = {}
    for k, pc in enumerate(pieces):
        d = ffprobe_duration(pc["wav"])
        for si, ln, a, b in pc["items"]:
            shot_first.setdefault(si, t + a)
            if ln is not None:
                timeline.append({"shot": si, "speaker": ln["speaker"], "roman": ln["roman"],
                                 "start": round(t + a, 3), "end": round(t + b, 3)})
        parts.append(pc["wav"])
        t += d
        if k < len(pieces) - 1:
            parts.append(gap)
            t += gap_s
    joined = ctx.workdir / "voice_raw.wav"
    _concat(parts, joined, ctx.workdir)
    mastered = ctx.workdir / "voice.wav"
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(joined), "-af", MASTER_CHAIN,
          "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(mastered)])
    total = ffprobe_duration(mastered)

    n = len(script["shots"])
    starts, prev = [], 0.0
    for si in range(n):
        s = max(prev, shot_first.get(si, prev) - 0.12) if si else 0.0
        starts.append(s)
        prev = s
    log(NAME, f"{engine} · {len(timeline)} lines · {n} shots · {total:.1f}s · mastered 48 kHz")
    ctx.notes["voice_engine"] = engine
    return mastered, timeline, starts, total
