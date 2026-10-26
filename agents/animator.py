"""Animator agent: turns each still illustration into a living 3D shot, for free.

1. Depth: estimates how far every pixel is from the camera, using the open-source Depth Anything V2
   Small model (runs on the CPU via ONNX, downloaded once and cached). If the model isn't available,
   a framing-aware estimate is used instead.
2. Parallax camera: near things move more than far things, so zooms feel like the camera is moving
   *into* the scene, and pans reveal depth, like a 2.5D animated shot.
3. Atmosphere: drifting fog (thicker in the distance), floating dust/embers, lantern-light flicker
   at night, and a flash-and-shake on scares.
Frames are rendered with OpenCV and piped straight into ffmpeg.
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import requests

from .common import ROOT, Context, log

NAME = "Animator"
MODEL_URLS = [
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model_quantized.onnx",
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx",
    "https://huggingface.co/inference4j/depth-anything-v2-small/resolve/main/model.onnx",
]
MODEL_PATH = ROOT / ".cache" / "depth-anything-v2-small.onnx"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


# ── depth ───────────────────────────────────────────────────────────────
class Depth:
    def __init__(self, ctx: Context):
        self.session = None
        self.kind = "estimate"
        if ctx.mock or not ctx.cfg["visuals"].get("depth_model", True):
            return
        try:
            import onnxruntime as ort
            if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1_000_000:
                self._download()
            self.session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            self.inp = self.session.get_inputs()[0].name
            self.kind = "model"
            log(NAME, "depth model ready (Depth Anything V2 Small)")
        except Exception as e:  # no internet / model moved / runtime missing → framing-aware estimate
            log(NAME, f"depth model unavailable ({str(e)[:100]}); using depth estimates")
            self.session = None

    @staticmethod
    def _download() -> None:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        last = ""
        for url in MODEL_URLS:
            try:
                with requests.get(url, stream=True, timeout=(15, 300)) as r:
                    if r.status_code != 200:
                        last = f"HTTP {r.status_code}"
                        continue
                    tmp = MODEL_PATH.with_suffix(".part")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                    if tmp.stat().st_size > 1_000_000:
                        tmp.rename(MODEL_PATH)
                        return
            except requests.RequestException as e:
                last = type(e).__name__
        raise RuntimeError(f"could not download depth model ({last})")

    def of(self, img: np.ndarray, framing: str) -> np.ndarray:
        """Depth map in 0..1, 1 = nearest, same size as img (H, W)."""
        h, w = img.shape[:2]
        d = None
        if self.session is not None:
            try:
                scale = 518 / max(h, w)
                nh, nw = [int(math.ceil(x * scale / 14) * 14) for x in (h, w)]
                x = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255
                x = ((x - MEAN) / STD).transpose(2, 0, 1)[None]
                out = self.session.run(None, {self.inp: x})[0]
                out = np.squeeze(out).astype(np.float32)
                d = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
            except Exception as e:
                log(NAME, f"depth model failed on one image ({str(e)[:80]}); using an estimate")
                d = None
        if d is None:
            d = _estimate_depth(h, w, framing)
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d - lo) / max(1e-6, hi - lo), 0, 1)
        # soften so the warp stretches smoothly instead of tearing at edges
        k = max(3, int(min(h, w) * 0.025) | 1)
        return cv2.GaussianBlur(d, (k, k), 0).astype(np.float32)


def _estimate_depth(h: int, w: int, framing: str) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ground = (yy / h) ** 1.3                                   # lower in frame = nearer
    if framing in ("closeup", "extreme_closeup", "medium", "over_shoulder", "two_shot", "object"):
        cy = 0.42 if framing in ("closeup", "extreme_closeup") else 0.55
        sx, sy = (0.30, 0.30) if framing != "two_shot" else (0.42, 0.34)
        subject = np.exp(-(((xx / w - 0.5) / sx) ** 2 + ((yy / h - cy) / sy) ** 2))
        return 0.35 * ground + 0.75 * subject
    return 0.15 + 0.85 * ground


# ── atmosphere textures (made once per reel) ────────────────────────────
def _fog_texture(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Soft cloud texture at quarter resolution, twice as wide so it can drift sideways."""
    small = rng.random((h // 24 + 4, w // 24 + 4)).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), 2.2)
    fog = cv2.resize(small, (w // 2, h // 4 + 50), interpolation=cv2.INTER_CUBIC)
    fog = (fog - fog.min()) / max(1e-6, fog.max() - fog.min())
    return np.clip((fog - 0.35) * 1.6, 0, 1).astype(np.float32)


class Particles:
    def __init__(self, w: int, h: int, n: int, rng: np.random.Generator, warm: bool):
        self.w, self.h = w, h
        self.p = np.stack([rng.uniform(0, w, n), rng.uniform(0, h, n)], 1).astype(np.float32)
        self.v = np.stack([rng.uniform(-6, 6, n), rng.uniform(-14, -3, n)], 1).astype(np.float32)
        self.r = rng.uniform(1.2, 3.2, n).astype(np.float32)
        self.a = rng.uniform(0.25, 0.7, n).astype(np.float32)
        self.ph = rng.uniform(0, 6.28, n).astype(np.float32)
        self.col = (120, 190, 255) if warm else (235, 235, 235)  # BGR

    def layer(self, t: float, strength: float) -> np.ndarray:
        """Particles drawn at half resolution (caller upscales and adds)."""
        lay = np.zeros((self.h // 2, self.w // 2, 3), np.uint8)
        if strength <= 0:
            return lay
        pos = self.p + self.v * t
        pos[:, 0] = (pos[:, 0] + 8 * np.sin(t * 0.7 + self.ph)) % self.w
        pos[:, 1] = pos[:, 1] % self.h
        for (x, y), r, a in zip(pos, self.r, self.a):
            c = tuple(int(ci * a * strength) for ci in self.col)
            cv2.circle(lay, (int(x / 2), int(y / 2)), max(1, int(r / 2 + 0.5)), c, -1, lineType=cv2.LINE_AA)
        return lay


# ── camera ──────────────────────────────────────────────────────────────
def _ease(p: float) -> float:
    return (1 - math.cos(math.pi * min(1.0, max(0.0, p)))) / 2


def _camera(motion: str, p: float, m: float) -> tuple[float, float, float, float, float]:
    """Returns (base_zoom, depth_zoom, base_dx, depth_dx, depth_dy) as fractions of the frame."""
    e = _ease(p)
    if motion == "zoom_out":
        return 0.35 * m * (1 - e), 0.9 * m * (1 - e), 0, 0, 0
    if motion in ("pan_left", "pan_right"):
        s = -1 if motion == "pan_left" else 1
        return 0.5 * m, 0.1 * m, s * 0.35 * m * (e - 0.5), s * 0.55 * m * (e - 0.5), 0
    if motion in ("pan_up", "pan_down"):
        s = -1 if motion == "pan_up" else 1
        return 0.5 * m, 0.1 * m, 0, 0, s * 0.6 * m * (e - 0.5)
    if motion == "push_fast":
        f = 1 - (1 - min(1.0, p)) ** 3
        return 0.5 * m * f, 1.4 * m * f, 0, 0, 0
    if motion == "hold":
        return 0.02 + 0.01 * e, 0.03 * e, 0, 0, 0
    return 0.35 * m * e, 0.9 * m * e, 0, 0, 0       # zoom_in (dolly)


ALT = {"zoom_in": "zoom_out", "zoom_out": "zoom_in", "pan_left": "pan_right", "pan_right": "pan_left",
       "hold": "zoom_in", "push_fast": "hold", "pan_up": "pan_down", "pan_down": "pan_up"}


def animate(ctx: Context, image_paths: dict[str, Path], shots: list[dict], durations: list[float]) -> list[Path]:
    vcfg, vis = ctx.cfg["video"], ctx.cfg["visuals"]
    W, H, fps = vcfg["width"], vcfg["height"], vcfg["fps"]
    xf = vcfg.get("crossfade", 0.12)
    m0 = float(vis.get("motion_strength", 0.10))
    atmo = vis.get("atmosphere", True)
    margin = 1.0 + 1.6 * m0 + 0.04                 # source is drawn bigger than the frame → no edges
    SW, SH = int(W * margin) // 2 * 2, int(H * margin) // 2 * 2
    rng = np.random.default_rng()
    out_dir = ctx.workdir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)

    depth = Depth(ctx)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    fog_tex = _fog_texture(W, H, rng) if atmo else None
    dust_night = Particles(W, H, 70, rng, warm=True) if atmo else None
    dust_day = Particles(W, H, 45, rng, warm=False) if atmo else None
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = (W - 1) / 2, (H - 1) / 2
    last_motion: dict[str, str] = {}
    clips: list[Path] = []
    used_model = 0

    for i, (shot, d) in enumerate(zip(shots, durations)):
        iid = shot["image_id"]
        if iid not in cache:
            img = cv2.imread(str(image_paths[iid]), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (SW, SH), interpolation=cv2.INTER_LANCZOS4)
            dm = depth.of(img, shot.get("framing", "medium"))
            used_model += depth.kind == "model"
            cache[iid] = (img, dm)
        img, dm = cache[iid]
        # depth seen from the output grid (approximation for the backward warp)
        d_out = cv2.resize(dm[(SH - H) // 2:(SH - H) // 2 + H, (SW - W) // 2:(SW - W) // 2 + W], (W, H))

        motion = shot.get("motion", "zoom_in")
        if last_motion.get(iid) == motion:
            motion = ALT.get(motion, "zoom_in")
        last_motion[iid] = motion
        m = m0 * min(1.0, max(0.55, d / 3.5))
        scare = shot.get("mood") == "scare"
        night = shot.get("night", True)
        length = d + (xf if i < len(shots) - 1 else 0) + 0.05
        n = int(math.ceil(length * fps))

        out = out_dir / f"clip{i:02d}.mp4"
        proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
             "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "14",
             "-pix_fmt", "yuv420p", str(out)], stdin=subprocess.PIPE)
        fog_amt = (0.22 if night else 0.10) + (0.06 if shot.get("mood") in ("tense", "scare", "uneasy") else 0)
        hw, hh = W // 2, H // 2
        d_half = cv2.resize(d_out, (hw, hh), interpolation=cv2.INTER_AREA)
        yh, xh = yy[::2, ::2][:hh, :hw], xx[::2, ::2][:hh, :hw]
        far_q = cv2.resize(((1.0 - d_out) ** 1.5) * fog_amt, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        fog_col = np.full((H, W, 3), (235, 225, 215) if night else (240, 235, 230), np.uint8)
        t0 = i * 7.3
        for f in range(n):
            p = f / max(1, n - 1)
            bz, dz, bx, dx, dy = _camera(motion, p, m)
            shake_x = shake_y = 0.0
            if scare and f < 24:
                k = 1 - f / 24
                shake_x, shake_y = 9 * math.sin(f * 2.3) * k, 7 * math.cos(f * 1.9) * k
            # camera maps at half resolution (depth is smooth), upscaled → same result, ~4x faster
            scale = 1.0 + bz + dz * d_half
            mxh = (SW / 2) + (xh - cx) / scale + (bx + dx * (d_half - 0.5)) * W + shake_x
            myh = (SH / 2) + (yh - cy) / scale + (dy * (d_half - 0.5)) * H + shake_y
            map_x = cv2.resize(mxh, (W, H), interpolation=cv2.INTER_LINEAR)
            map_y = cv2.resize(myh, (W, H), interpolation=cv2.INTER_LINEAR)
            frame = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            if atmo:
                t = t0 + f / fps
                off = int((t * 18) % W) // 4
                fq = fog_tex[12:12 + H // 4, off:off + W // 4]
                fq = fq[:, : W // 4] if fq.shape[1] >= W // 4 else np.pad(fq, ((0, 0), (0, W // 4 - fq.shape[1])))
                alpha = cv2.resize(fq * far_q, (W, H), interpolation=cv2.INTER_LINEAR)
                a8 = (alpha * 255).astype(np.uint8)
                a3 = cv2.merge([a8, a8, a8])
                frame = cv2.add(cv2.subtract(frame, cv2.multiply(frame, a3, scale=1 / 255)),
                                cv2.multiply(fog_col, a3, scale=1 / 255))
                if night:
                    flick = 1.0 + 0.025 * math.sin(t * 9.1) + 0.015 * math.sin(t * 23.7 + 1.3)
                    frame = cv2.convertScaleAbs(frame, alpha=flick)
                dust = (dust_night if night else dust_day).layer(t, 0.55 if night else 0.35)
                frame = cv2.add(frame, cv2.resize(dust, (W, H), interpolation=cv2.INTER_LINEAR))
            if scare and f < 5:
                frame = cv2.addWeighted(frame, 1 - (5 - f) / 6, np.full_like(frame, 255), (5 - f) / 6, 0)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError(f"encoding cut {i} failed")
        clips.append(out)
    ctx.notes["depth"] = "model" if used_model else depth.kind
    log(NAME, f"{len(clips)} cuts animated in 3D ({'depth model' if used_model else 'depth estimates'}"
              f"{', fog + dust + flicker' if atmo else ''})")
    return clips
