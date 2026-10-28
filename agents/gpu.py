"""GPU agent: uses the Modal GPU engine (gpu_app.py) when it's set up, and stays out of the way when it isn't.

- Pictures: shots with characters are drawn by Qwen-Image-Edit from the character portraits (better faces and
  consistency than the free Cloudflare model). Used as the first picture provider by the Illustrator.
- Motion: the N most important shots become real moving clips (Wan 2.2 image-to-video). The Animator uses those
  clips instead of camera moves for those shots.
- Budget: Modal gives $30 of free GPU time a month. Every call's time is counted; once this month's estimate
  reaches `gpu.monthly_cap_usd`, the engine is skipped until next month (the reel is made the normal way).
Needs the GitHub secrets MODAL_TOKEN_ID and MODAL_TOKEN_SECRET; without them nothing here runs.
"""
from __future__ import annotations

import io
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .common import Context, log

NAME = "GPU"
APP = "3am-tales-gpu"
H100_PER_SECOND = 0.001097 * 1.15        # GPU + the CPU/memory that comes with it (Modal list prices, Oct 2026)

MOOD_MOTION = {
    "scare": "a sudden startled reaction, eyes widening, a sharp breath",
    "sad": "tears rolling slowly down, lips trembling, a slow sad breath",
    "tense": "nervous glances, shallow breathing, very slight trembling",
    "uneasy": "slow cautious movement, eyes searching",
    "calm": "gentle breathing and a soft blink",
}


def _month() -> str:
    return f"{datetime.now(timezone.utc):%Y-%m}"


class Engine:
    def __init__(self, ctx: Context, memory_data: dict):
        self.ctx = ctx
        self.cfg = ctx.cfg.get("gpu") or {}
        self.mem = memory_data
        self.seconds = 0.0
        self.failed: set[str] = set()
        spend = self.mem.setdefault("modal_spend", {"month": _month(), "usd": 0.0})
        if spend.get("month") != _month():
            spend.update(month=_month(), usd=0.0)
        has_keys = bool(os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"))
        self.on = bool(self.cfg.get("enabled", True)) and has_keys and not ctx.mock
        cap = float(self.cfg.get("monthly_cap_usd", 27))
        if self.on and spend["usd"] >= cap:
            log(NAME, f"this month's Modal estimate ${spend['usd']:.2f} reached the ${cap:.0f} cap; "
                      "using the free picture providers until next month")
            self.on = False
        if self.on:
            try:
                import modal  # noqa: F401
            except ImportError:
                log(NAME, "the 'modal' package isn't installed; GPU engine off")
                self.on = False

    # ── bookkeeping ──────────────────────────────────────────────
    @property
    def run_cost(self) -> float:
        return self.seconds * H100_PER_SECOND

    def _left(self) -> float:
        cap = float(self.cfg.get("monthly_cap_usd", 27))
        return cap - self.mem["modal_spend"]["usd"] - self.run_cost

    def _call(self, kind: str, fn):
        t0 = time.time()
        try:
            return fn()
        finally:
            self.seconds += time.time() - t0

    def finish(self) -> None:
        if self.seconds:
            self.mem["modal_spend"]["usd"] = round(self.mem["modal_spend"]["usd"] + self.run_cost, 3)
            log(NAME, f"Modal time this run ≈ {self.seconds / 60:.1f} min ≈ ${self.run_cost:.2f}; "
                      f"this month ≈ ${self.mem['modal_spend']['usd']:.2f} of the free $30")

    # ── pictures ─────────────────────────────────────────────────
    def picture_provider(self):
        return ModalPictures(self) if self.on and self.cfg.get("pictures", True) else None

    # ── motion ───────────────────────────────────────────────────
    def pick_shots(self, shots: list[dict], durations: list[float]) -> list[int]:
        n = int(self.cfg.get("motion_clips", 6))
        cands = []
        for i, (sh, d) in enumerate(zip(shots, durations)):
            if d < 2.2 or not sh.get("characters"):
                continue
            score = min(d, 6) + {"scare": 2.5, "sad": 2, "tense": 1.5}.get(sh.get("mood"), 0) + \
                {"closeup": 1.5, "extreme_closeup": 1.5, "medium": 1, "two_shot": 1}.get(sh.get("framing"), 0)
            cands.append((score, i))
        return sorted(i for _, i in sorted(cands, reverse=True)[:n])

    def motion_clips(self, shots: list[dict], image_paths: dict[str, Path],
                     durations: list[float]) -> dict[int, Path]:
        if not self.on or not self.cfg.get("motion", True) or "motion" in self.failed:
            return {}
        try:
            import modal
            wan = modal.Cls.from_name(APP, "WanI2V")()
        except Exception as e:  # noqa: BLE001
            log(NAME, f"motion engine not reachable ({str(e)[:120]}); using camera moves")
            return {}
        out_dir = self.ctx.workdir / "motion"
        out_dir.mkdir(parents=True, exist_ok=True)
        clips: dict[int, Path] = {}
        by_image: dict[str, Path] = {}
        per_clip = float(self.cfg.get("est_seconds_per_clip", 90))
        for i in self.pick_shots(shots, durations):
            sh = shots[i]
            iid = sh["image_id"]
            if iid in by_image:                          # same picture → reuse the clip
                clips[i] = by_image[iid]
                continue
            if self._left() < per_clip * H100_PER_SECOND:
                log(NAME, "monthly GPU budget nearly used; remaining shots keep camera moves")
                break
            prompt = (f"{sh['image']}. Subtle natural cinematic motion: {MOOD_MOTION.get(sh.get('mood'), MOOD_MOTION['calm'])}; "
                      "hair and clothes move gently, slow camera push-in. Keep the 2D illustrated art style, "
                      "the same faces, outfits and lighting.")
            try:
                data = self._call("motion", lambda: wan.animate.remote(Path(image_paths[iid]).read_bytes(),
                                                                         prompt[:1500], 1000 + i))
            except Exception as e:  # noqa: BLE001
                log(NAME, f"motion clip for cut {i + 1} failed ({str(e)[:140]}); using camera moves from here")
                self.failed.add("motion")
                break
            dest = out_dir / f"motion_{i:02d}.mp4"
            dest.write_bytes(data)
            clips[i] = by_image[iid] = dest
            log(NAME, f"moving clip for cut {i + 1} ({sh.get('mood')}, {durations[i]:.1f}s)")
        if clips:
            log(NAME, f"{len(clips)} cuts have real character movement")
        return clips


class ModalPictures:
    """Picture provider for the Illustrator's Studio: only for shots with character portraits."""
    name = "modal"
    ready = True

    def __init__(self, engine: Engine):
        self.engine = engine
        self.cls = None

    def generate(self, prompt: str, w: int, h: int, seed: int, refs: list[Path]) -> bytes:
        from .illustrator import ImageError, _ref_bytes
        if not refs:
            raise ImageError("SKIP no character references")
        if self.engine._left() < 20 * H100_PER_SECOND:
            raise ImageError("QUOTA monthly GPU budget used")
        try:
            if self.cls is None:
                import modal
                self.cls = modal.Cls.from_name(APP, "QwenEdit")()
            # the portraits are "image 1", "image 2" … for Qwen
            for k in range(len(refs)):
                prompt = prompt.replace(f"reference image {k})", f"image {k + 1})")
            data = self.engine._call("pictures", lambda: self.cls.draw.remote(
                prompt[:2500], [_ref_bytes(p, 768) for p in refs[:3]], w // 16 * 16, h // 16 * 16, seed))
        except ImageError:
            raise
        except Exception as e:  # noqa: BLE001 — anything from Modal → next provider
            raise ImageError(f"modal: {str(e)[:200]}")
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB").resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "PNG")
        return buf.getvalue()


# ── one-time setup + benchmark (workflow mode "gpu_setup") ───────────────
def setup_and_benchmark(ctx: Context, memory_data: dict, out_dir: Path) -> dict:
    """Downloads the models into Modal once, then draws 2 pictures and animates 2 clips of a test character,
    and reports the real time and cost per reel. Nothing is posted."""
    import json
    import random

    from . import illustrator
    report: dict = {"ok": False}
    eng = Engine(ctx, memory_data)
    if not eng.on:
        log(NAME, "Modal isn't set up (add MODAL_TOKEN_ID and MODAL_TOKEN_SECRET as GitHub secrets)")
        return report
    import modal
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log(NAME, "downloading the models into Modal storage (first time only, ~20-40 min; later runs skip this)")
    report["download"] = modal.Function.from_name(APP, "download").remote()
    report["download_minutes"] = round((time.time() - t0) / 60, 1)

    # a test character portrait from the normal free provider
    char = {"id": "C1", "name": "Meera", "gender": "female", "age": 17, "is_entity": False,
            "look": "slim village girl, wheatish skin, long straight black hair, small black bindi, gold jhumkas, "
                    "mustard-yellow cotton salwar kameez with small green prints and a dark green dupatta"}
    studio = illustrator.Studio(ctx)
    portrait = out_dir / "bench-portrait.png"
    studio.draw(illustrator.portrait_prompt(ctx, char), 768, 768, random.randint(1, 2**30), [], portrait)
    shots = [
        {"id": "B1", "framing": "closeup", "characters": ["C1"], "night": True, "mood": "sad",
         "image": "Meera's face in close-up, tears in her eyes, warm lantern light on one side, blue moonlight on the other"},
        {"id": "B2", "framing": "medium", "characters": ["C1"], "night": True, "mood": "scare",
         "image": "Meera standing in a narrow village lane at night holding a lantern, turning back in fear"},
    ]
    pics = ModalPictures(eng)
    v = ctx.cfg["visuals"]
    timings = {"pictures_s": [], "clips_s": []}
    for sh in shots:
        t = time.time()
        data = pics.generate(illustrator.shot_prompt(ctx, sh, {"C1": char}, ["C1"]), v["width"], v["height"],
                             random.randint(1, 2**30), [portrait])
        timings["pictures_s"].append(round(time.time() - t, 1))
        (out_dir / f"bench-{sh['id']}.png").write_bytes(data)
    wan = modal.Cls.from_name(APP, "WanI2V")()
    for sh in shots:
        t = time.time()
        prompt = f"{sh['image']}. Subtle natural cinematic motion: {MOOD_MOTION[sh['mood']]}; keep the 2D illustrated style."
        clip = eng._call("bench", lambda: wan.animate.remote((out_dir / f"bench-{sh['id']}.png").read_bytes(), prompt, 7))
        timings["clips_s"].append(round(time.time() - t, 1))
        (out_dir / f"bench-{sh['id']}.mp4").write_bytes(clip)
    eng.finish()
    # first call of each model includes loading it (a cold start); the second one is the steady speed
    pic_s, clip_s = timings["pictures_s"][-1], timings["clips_s"][-1]
    n_pics = int(ctx.cfg["story"].get("max_images", 18))
    n_clips = int(ctx.cfg.get("gpu", {}).get("motion_clips", 6))
    cold = (timings["pictures_s"][0] - pic_s) + (timings["clips_s"][0] - clip_s)
    per_reel_s = cold + n_pics * pic_s + n_clips * clip_s
    report.update(ok=True, **timings, per_reel_minutes=round(per_reel_s / 60, 1),
                  per_reel_usd=round(per_reel_s * H100_PER_SECOND, 2),
                  per_month_usd=round(per_reel_s * H100_PER_SECOND * 30, 1),
                  benchmark_cost_usd=round(eng.run_cost, 2))
    (out_dir / "gpu-benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(NAME, f"BENCHMARK: picture {pic_s}s, clip {clip_s}s (after loading) → one reel with {n_pics} pictures + "
              f"{n_clips} moving cuts ≈ {per_reel_s / 60:.0f} min ≈ ${report['per_reel_usd']} → "
              f"≈ ${report['per_month_usd']}/month (free credit: $30)")
    if report["per_month_usd"] > 30:
        log(NAME, "→ over the free $30: lower gpu.motion_clips in config.yaml (each clip ≈ "
                  f"${clip_s * H100_PER_SECOND * 30:.2f}/month)")
    return report
