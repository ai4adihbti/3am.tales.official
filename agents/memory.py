"""Memory agent: the system's long-term state, stored in state/.

- state/history.json      : every post, its metrics, analyst learnings, and the active series
- state/series/<id>/*.png : character reference portraits, so characters look the same in every part

GitHub Actions commits state/ back to the repo after every successful run.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT

STATE_DIR = ROOT / "state"
STATE_FILE = STATE_DIR / "history.json"


class Memory:
    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.data.setdefault("posts", [])
        self.data.setdefault("used_video_ids", [])
        self.data.setdefault("learnings", "")
        self.data.setdefault("format_scores", {})
        self.data.setdefault("series", None)
        self.data.setdefault("series_done", [])
        self.data.setdefault("spend", {"month": "", "usd": 0.0})

    # ── reads ────────────────────────────────────────────────
    @property
    def posts(self) -> list[dict]:
        return self.data["posts"]

    def recent_titles(self, n: int = 60) -> list[str]:
        titles = [p.get("series_title") or p.get("title", "") for p in self.posts[-n:]]
        return list(dict.fromkeys(t for t in titles + self.data["series_done"] if t))

    def used_video_ids(self) -> set:
        return set(self.data["used_video_ids"])

    def active_series(self) -> dict | None:
        s = self.data.get("series")
        if s and s.get("parts_done", 0) < s.get("total_parts", 0) and s.get("characters"):
            return s
        return None

    def series_dir(self, series_id: str) -> Path:
        return STATE_DIR / "series" / series_id

    def new_series_id(self, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "series"
        return f"{datetime.now(timezone.utc):%Y%m%d}-{slug}"

    # ── writes ───────────────────────────────────────────────
    def add_post(self, post: dict) -> None:
        post.setdefault("posted_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.posts.append(post)

    def advance_series(self, series: dict, script: dict, refs: dict[str, Path]) -> None:
        """Record that a part was published; keep characters + their portraits for the next part."""
        d = self.series_dir(series["id"])
        d.mkdir(parents=True, exist_ok=True)
        ref_files = {}
        for cid, p in refs.items():
            dest = d / f"{cid}.png"
            if Path(p).resolve() != dest.resolve():
                shutil.copy(p, dest)
            ref_files[cid] = str(dest.relative_to(ROOT))
        s = {k: series[k] for k in ("id", "title", "format", "total_parts", "outline", "concept",
                                    "voice_style", "narrator_gender", "voice_engine") if k in series}
        if series.get("part") == 1:
            self.data["last_voice_style"] = series.get("voice_style")
            if series.get("voice_style") == "narrator":
                self.data["last_narrator_gender"] = series.get("narrator_gender")
        s["characters"] = script["characters"]
        s["ref_files"] = ref_files
        s["parts_done"] = series["part"]
        s["previously"] = list(series.get("previously") or []) + [script.get("summary", "")]
        if s["parts_done"] >= s["total_parts"]:
            self.data["series_done"].append(s["title"])
            self.data["series"] = None
            # free repo space: portraits aren't needed once a series is finished
            shutil.rmtree(d, ignore_errors=True)
        else:
            self.data["series"] = s

    # ── paid Gemini usage (your Google AI Pro credit) ─────────
    def month_spend(self) -> float:
        s = self.data["spend"]
        return s["usd"] if s.get("month") == f"{datetime.now(timezone.utc):%Y-%m}" else 0.0

    def add_spend(self, usd: float) -> None:
        month = f"{datetime.now(timezone.utc):%Y-%m}"
        s = self.data["spend"]
        if s.get("month") != month:
            s.update(month=month, usd=0.0)
        s["usd"] = round(s["usd"] + usd, 4)

    def remember_videos(self, ids: list, keep: int = 400) -> None:
        merged = self.data["used_video_ids"] + [i for i in ids if i not in self.data["used_video_ids"]]
        self.data["used_video_ids"] = merged[-keep:]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
