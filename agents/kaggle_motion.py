"""Real movement on Kaggle's free GPUs (about 30 GPU hours a week, no card).

The Engine sends the reel's most important pictures to a private Kaggle notebook. The notebook runs Lightricks
LTX-Video (open weights, Israel) on a free T4 GPU and turns each picture into a ~3 s clip where the scene really
moves (a ghost turning its head, hair in the wind, mist, flickering lantern light). The clips come back and the
Animator uses them instead of a camera move over the still picture.

Needs the GitHub secrets KAGGLE_USERNAME and KAGGLE_KEY (Kaggle → Settings → API → Create New Token) and a
phone-verified Kaggle account (Kaggle only gives GPUs to verified accounts). Without them nothing here runs.
Everything is optional: on any failure or timeout the reel is made the normal way.
"""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

from .common import log

NAME = "Kaggle"
KERNEL_SLUG = "darawni-motion"

# The notebook that runs on Kaggle. Jobs (pictures as base64 JPEG + prompts) are pasted in where JOBS_B64 is,
# so no dataset upload is needed and the notebook always sees this run's pictures.
KERNEL_SCRIPT = r'''
import base64, io, json, os, subprocess, sys, time, traceback
t0 = time.time()
LOG = {"started": t0, "clips": {}, "errors": []}
OUT = "/kaggle/working/out"
os.makedirs(OUT, exist_ok=True)

def save_log():
    LOG["seconds"] = round(time.time() - t0, 1)
    json.dump(LOG, open(os.path.join(OUT, "log.json"), "w"), indent=1)

JOBS = json.loads(base64.b64decode("JOBS_B64").decode())
RUN_ID = JOBS["run_id"]
LOG["run_id"] = RUN_ID
save_log()
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "diffusers>=0.35", "transformers>=4.51,<5",
                    "accelerate", "sentencepiece", "imageio[ffmpeg]", "ftfy"], check=True)
    import torch
    from PIL import Image
    from diffusers import LTXConditionPipeline
    from diffusers.pipelines.ltx.pipeline_ltx_condition import LTXVideoCondition
    from diffusers.utils import export_to_video
    LOG["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    model = JOBS.get("model", "Lightricks/LTX-Video-0.9.5")
    dtype = torch.float16 if JOBS.get("dtype", "fp16") == "fp16" else torch.bfloat16
    t = time.time()
    pipe = LTXConditionPipeline.from_pretrained(model, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    LOG["model"], LOG["load_seconds"] = model, round(time.time() - t, 1)
    save_log()
    W, H = JOBS.get("width", 480), JOBS.get("height", 832)
    for job in JOBS["jobs"]:
        t = time.time()
        try:
            img = Image.open(io.BytesIO(base64.b64decode(job["image"]))).convert("RGB").resize((W, H), Image.LANCZOS)
            frames = max(9, (int(job.get("frames", 73)) - 1) // 8 * 8 + 1)
            out = pipe(conditions=[LTXVideoCondition(image=img, frame_index=0)], prompt=job["prompt"],
                       negative_prompt=JOBS["negative"], width=W, height=H, num_frames=frames,
                       num_inference_steps=int(JOBS.get("steps", 30)), guidance_scale=float(JOBS.get("cfg", 3.0)),
                       decode_timestep=0.05, decode_noise_scale=0.025,
                       generator=torch.Generator("cpu").manual_seed(int(job.get("seed", 7)))).frames[0]
            import numpy as np
            mean = float(np.asarray(out[len(out) // 2]).mean())
            if mean < 3:                                   # fp16 overflow gives black frames
                raise RuntimeError(f"black frames (mean {mean:.1f}); try dtype bf16")
            path = os.path.join(OUT, f"{RUN_ID}_{job['name']}.mp4")
            export_to_video(out, path, fps=24)
            LOG["clips"][job["name"]] = round(time.time() - t, 1)
        except Exception as e:
            LOG["errors"].append(f"{job['name']}: {e!r}"[:400])
        save_log()
except Exception:
    LOG["errors"].append(traceback.format_exc()[-1500:])
save_log()
'''

NEGATIVE = ("worst quality, inconsistent motion, blurry, jittery, distorted, deformed face, extra limbs, "
            "morphing, melting, text, watermark, static image")


def available() -> bool:
    return bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) and shutil.which("kaggle") is not None


def _kaggle(*args: str, timeout: int = 300) -> str:
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True, timeout=timeout)
    out = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args[:2])}: {out[-300:]}")
    return out


def _jpeg_b64(path: Path, w: int, h: int) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def animate(jobs: list[dict], workdir: Path, timeout_min: float, settings: dict | None = None) -> dict[str, Path]:
    """jobs: [{"name", "image" (Path), "prompt", "frames", "seed"}] → {name: mp4 path} for the clips that worked."""
    if not jobs or timeout_min < 8:
        return {}
    s = settings or {}
    user = os.environ["KAGGLE_USERNAME"]
    run_id = uuid.uuid4().hex[:8]
    w, h = int(s.get("width", 480)), int(s.get("height", 832))
    payload = {"run_id": run_id, "model": s.get("model", "Lightricks/LTX-Video-0.9.5"),
               "dtype": s.get("dtype", "fp16"), "steps": int(s.get("steps", 30)), "cfg": float(s.get("cfg", 3.0)),
               "width": w, "height": h, "negative": NEGATIVE,
               "jobs": [{"name": j["name"], "prompt": j["prompt"][:900], "frames": int(j.get("frames", 73)),
                         "seed": int(j.get("seed", 7)), "image": _jpeg_b64(Path(j["image"]), w, h)} for j in jobs]}
    kdir = workdir / "kaggle_kernel"
    shutil.rmtree(kdir, ignore_errors=True)
    kdir.mkdir(parents=True)
    script = KERNEL_SCRIPT.replace("JOBS_B64", base64.b64encode(json.dumps(payload).encode()).decode())
    (kdir / "script.py").write_text(script, encoding="utf-8")
    meta = {"id": f"{user}/{KERNEL_SLUG}", "title": KERNEL_SLUG, "code_file": "script.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [],
            "kernel_sources": []}
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    log(NAME, f"sending {len(jobs)} pictures to a free Kaggle GPU ({len(script) / 1e6:.1f} MB notebook)")
    _kaggle("kernels", "push", "-p", str(kdir), "--accelerator", "NvidiaTeslaT4",
            "-t", str(int(timeout_min * 60)), timeout=180)

    deadline = time.time() + timeout_min * 60
    status = ""
    time.sleep(45)
    while time.time() < deadline:
        try:
            status = _kaggle("kernels", "status", f"{user}/{KERNEL_SLUG}", timeout=60).lower()
        except Exception as e:  # noqa: BLE001 — a hiccup while polling is not a failure
            status = f"poll error {e}"
        if "complete" in status or "error" in status or "cancel" in status:
            break
        time.sleep(30)
    else:
        log(NAME, f"no result within {timeout_min:.0f} min (last status: {status[-80:]}); using camera moves")
        try:
            _kaggle("kernels", "cancel", f"{user}/{KERNEL_SLUG}", timeout=60)   # free the GPU quota
        except Exception:  # noqa: BLE001
            pass
        return {}

    out = workdir / "kaggle_out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    _kaggle("kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(out), timeout=600)
    logs = list(out.rglob("log.json"))
    info = json.loads(logs[0].read_text()) if logs else {}
    if not logs:
        try:                                            # show why the notebook stopped
            tail = _kaggle("kernels", "logs", f"{user}/{KERNEL_SLUG}", timeout=120)[-600:]
            log(NAME, f"notebook log tail: {tail}")
        except Exception:  # noqa: BLE001
            pass
    if info.get("run_id") != run_id:
        log(NAME, f"the notebook's output is not from this run (status: {status[-60:]}); using camera moves")
        return {}
    for err in info.get("errors", [])[:3]:
        log(NAME, f"notebook error: {err[:300]}")
    clips = {}
    for j in jobs:
        hits = list(out.rglob(f"{run_id}_{j['name']}.mp4"))
        if hits:
            clips[j["name"]] = hits[0]
    log(NAME, f"{len(clips)}/{len(jobs)} moving clips back · GPUs {info.get('gpus')} · model load "
              f"{info.get('load_seconds')}s · per clip {list(info.get('clips', {}).values())} s · "
              f"notebook total {info.get('seconds')}s")
    return clips
