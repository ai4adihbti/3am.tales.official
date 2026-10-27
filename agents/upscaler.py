"""Upscaler agent: sharpens every illustration with Real-ESRGAN's anime model before animation.

The free image model draws at 864x1536; a plain resize to 1080x1920 (plus camera-move headroom) makes the
line art soft. realesr-animevideov3 (x2) rebuilds crisp, clean anime lines and flat colour areas.
Runs on the CPU through ncnn + the Mesa "lavapipe" Vulkan driver (installed by the workflow), about 15-30 s
per picture. It runs in a separate process, so a crash or a missing driver can never stop the reel: the
original pictures are used instead.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .common import Context, log

NAME = "Upscaler"


def upscale(ctx: Context, paths: dict[str, Path]) -> dict[str, Path]:
    if ctx.mock or not ctx.cfg["visuals"].get("upscale", True):
        return paths
    unique = sorted({str(p) for p in paths.values()})
    jobs = [(src, str(Path(src).with_name(Path(src).stem + "_hd.png"))) for src in unique]
    t0 = time.time()
    try:
        res = subprocess.run([sys.executable, "-m", "agents.upscaler", json.dumps(jobs)],
                             capture_output=True, text=True, timeout=40 * 60)
    except subprocess.TimeoutExpired:
        log(NAME, "took too long; using the original pictures")
        return paths
    done = {}
    for line in res.stdout.splitlines():
        if line.startswith("DONE "):
            src, dst = json.loads(line[5:])
            done[src] = Path(dst)
    if not done:
        err = (res.stderr or res.stdout).strip().splitlines()[-1:] or ["unknown error"]
        log(NAME, f"not available ({err[0][:160]}); using the original pictures")
        return paths
    log(NAME, f"{len(done)}/{len(unique)} pictures sharpened (anime model, x2) in {time.time() - t0:.0f}s")
    return {iid: done.get(str(p), p) for iid, p in paths.items()}


def _worker(jobs: list[list[str]]) -> None:
    from PIL import Image
    import realesrgan_ncnn_py as R
    up = R.Realesrgan(gpuid=-1, model=0)        # CPU, realesr-animevideov3-x2
    for src, dst in jobs:
        try:
            out = up.process_pil(Image.open(src).convert("RGB"))
            out.save(dst)
            print("DONE " + json.dumps([src, dst]), flush=True)
        except Exception as e:  # noqa: BLE001 — one bad picture keeps its original
            print(f"FAIL {src}: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    _worker(json.loads(sys.argv[1]))
