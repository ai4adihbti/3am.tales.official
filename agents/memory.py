"""Memory agent: the system's long-term state, stored in state/history.json.

GitHub Actions commits this file back to the repo after every successful run,
so the system remembers what it has posted and how each post performed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT

STATE_FILE = ROOT / "state" / "history.json"


class Memory:
    def __init__(self, path: Path = STATE_FILE):
        self.path = path
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.data = {}
        self.data.setdefault("posts", [])
        self.data.setdefault("used_video_ids", [])
        self.data.setdefault("learnings", "")
        self.data.setdefault("format_scores", {})

    # ── reads ────────────────────────────────────────────────
    @property
    def posts(self) -> list[dict]:
        return self.data["posts"]

    def recent_titles(self, n: int = 60) -> list[str]:
        return [p["title"] for p in self.posts[-n:]]

    def used_video_ids(self) -> set[int]:
        return set(self.data["used_video_ids"])

    # ── writes ───────────────────────────────────────────────
    def add_post(self, post: dict) -> None:
        post.setdefault("posted_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
        self.posts.append(post)

    def remember_videos(self, ids: list[int], keep: int = 400) -> None:
        merged = self.data["used_video_ids"] + [i for i in ids if i not in self.data["used_video_ids"]]
        self.data["used_video_ids"] = merged[-keep:]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
