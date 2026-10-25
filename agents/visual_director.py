"""Visual Director agent: finds real, HD stock footage for every scene.

Footage sources (free):
- Pixabay  (PIXABAY_API_KEY), the default; keys are issued instantly at pixabay.com/api/docs
- Pexels   (PEXELS_API_KEY), used only if you already have a key; new keys are paused

Quality rules:
- every clip must fill a 1080x1920 frame WITHOUT upscaling: a vertical clip at least
  1920 tall, or a landscape 4K clip (2160 tall) that is centre-cropped to vertical
- prefer clips at least as long as the scene, so nothing visibly loops
- never reuse a clip the account has already used
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import requests

from .common import Context, log

NAME = "Visual Director"
MIN_EFFECTIVE = 1920       # Full HD vertical after cropping
FALLBACK_EFFECTIVE = 1440  # only if nothing sharper exists for any query


def effective_height(w: int, h: int) -> int:
    """Height of the sharpest 9:16 frame this clip can give without upscaling."""
    return int(min(h, w * 16 / 9))


# ── providers: each returns [{id, duration, url, width, height}] ────────
def _pixabay(query: str, key: str, page: int) -> list[dict]:
    r = requests.get("https://pixabay.com/api/videos/", params={
        "key": key, "q": query[:100], "video_type": "film", "safesearch": "true",
        "order": "popular", "per_page": 50, "page": page}, timeout=30)
    if r.status_code == 400 and "page" in r.text.lower():
        return []  # asked past the last page
    r.raise_for_status()
    out = []
    for hit in r.json().get("hits", []):
        files = [f for f in hit.get("videos", {}).values() if f.get("url") and f.get("width")]
        for f in files:
            out.append({"id": f"pixabay:{hit['id']}", "duration": hit.get("duration", 0),
                        "url": f["url"], "width": f["width"], "height": f["height"]})
    return out


def _pexels(query: str, key: str, page: int) -> list[dict]:
    r = requests.get("https://api.pexels.com/videos/search", headers={"Authorization": key},
                     params={"query": query, "orientation": "portrait", "size": "medium",
                             "per_page": 30, "page": page}, timeout=30)
    r.raise_for_status()
    out = []
    for v in r.json().get("videos", []):
        for f in v.get("video_files", []):
            if f.get("link") and f.get("width") and f.get("height"):
                out.append({"id": f"pexels:{v['id']}", "duration": v.get("duration", 0),
                            "url": f["link"], "width": f["width"], "height": f["height"]})
    return out


PROVIDERS = {"pixabay": ("PIXABAY_API_KEY", _pixabay), "pexels": ("PEXELS_API_KEY", _pexels)}


def active_provider() -> tuple[str, str] | None:
    for name, (env, _) in PROVIDERS.items():
        if os.getenv(env):
            return name, os.environ[env]
    return None


def _pick(provider: str, query: str, key: str, need_s: float, exclude: set, min_eff: int) -> dict | None:
    search = PROVIDERS[provider][1]
    for page in (1, 2):
        try:
            files = search(query, key, page)
        except requests.RequestException as e:
            log(NAME, f"search failed for '{query}': {e}")
            return None
        # For each clip keep its smallest file that is still sharp enough (faster download).
        best: dict[str, dict] = {}
        for f in files:
            if f["id"] in exclude or effective_height(f["width"], f["height"]) < min_eff:
                continue
            cur = best.get(f["id"])
            if cur is None or f["width"] * f["height"] < cur["width"] * cur["height"]:
                best[f["id"]] = f
        if best:
            pool = [f for f in best.values() if f["duration"] >= need_s + 0.5] or list(best.values())
            # Prefer native vertical clips (no crop), then the sharpest.
            return max(pool, key=lambda f: (f["height"] > f["width"],
                                            effective_height(f["width"], f["height"]), f["duration"]))
    return None


def _download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)


def _mock_clip(dest: Path, seconds: float, idx: int, cfg: dict) -> None:
    """Offline test footage: dark moving gradients (no API needed)."""
    pairs = [("0x1d2a3a", "0x05070a"), ("0x3a1d24", "0x0a0505"), ("0x1d3a2c", "0x050a07"),
             ("0x3a331d", "0x0a0905"), ("0x2a2a2a", "0x050505"), ("0x1d2f3a", "0x05080a")]
    c0, c1 = pairs[idx % len(pairs)]
    w, h, fps = cfg["video"]["width"], cfg["video"]["height"], cfg["video"]["fps"]
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", f"gradients=s={w}x{h}:r={fps}:c0={c0}:c1={c1}:speed=0.02:d={seconds + 1:.2f}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(dest)], check=True)


def gather(ctx: Context, scenes: list[dict], durations: list[float], used_ids: set) -> tuple[list[Path], list[str]]:
    out_dir = ctx.workdir / "clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    chosen_ids: list[str] = []

    if ctx.mock:
        for i, d in enumerate(durations):
            p = out_dir / f"scene{i}.mp4"
            _mock_clip(p, d, i, ctx.cfg)
            paths.append(p)
        log(NAME, f"{len(paths)} mock clips generated")
        return paths, chosen_ids

    prov = active_provider()
    if not prov:
        raise RuntimeError("Set PIXABAY_API_KEY (or PEXELS_API_KEY)")
    provider, key = prov

    exclude = {str(i) for i in used_ids}
    fallbacks = list(ctx.cfg["video"]["fallback_queries"])
    for i, (scene, need) in enumerate(zip(scenes, durations)):
        words = scene["visual"].split()
        queries = [scene["visual"], " ".join(words[:2]), words[-1] + " night" if words else ""] + fallbacks
        queries = [q for q in dict.fromkeys(queries) if q]
        pick = None
        for min_eff in (MIN_EFFECTIVE, FALLBACK_EFFECTIVE):
            for q in queries:
                pick = _pick(provider, q, key, need, exclude, min_eff)
                if pick:
                    if q != scene["visual"]:
                        log(NAME, f"scene {i + 1}: used fallback search '{q}'")
                    break
            if pick:
                break
        if not pick:
            raise RuntimeError(f"No usable HD footage found for scene {i + 1} ('{scene['visual']}')")

        dest = out_dir / f"scene{i}.mp4"
        _download(pick["url"], dest)
        exclude.add(pick["id"])
        chosen_ids.append(pick["id"])
        paths.append(dest)
        crop = "" if pick["height"] > pick["width"] else " (centre-cropped to vertical)"
        log(NAME, f"scene {i + 1}: '{scene['visual']}' → {pick['id']} "
                  f"{pick['width']}x{pick['height']} {pick['duration']}s{crop}")
    return paths, chosen_ids
