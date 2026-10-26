"""Analyst agent: reads each Reel's insights and teaches the Researcher what works.

Runs at the start of every cycle. Posts older than 24 h get their metrics refreshed
(up to 14 days old). Scores weight shares and saves highest, because those drive reach.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import requests

from .common import Context, log
from .memory import Memory

NAME = "Analyst"
METRIC_SETS = [
    "views,reach,likes,comments,shares,saved,total_interactions",
    "reach,likes,comments,shares,saved",
]


def _fetch_insights(graph: str, media_id: str, token: str) -> dict | None:
    last = ""
    for metrics in METRIC_SETS:
        r = requests.get(f"{graph}/{media_id}/insights",
                         params={"metric": metrics, "access_token": token}, timeout=30)
        last = f"HTTP {r.status_code} {r.text[:160]}"
        if r.status_code == 200:
            out = {}
            for item in r.json().get("data", []):
                vals = item.get("values") or [{}]
                out[item["name"]] = vals[0].get("value", item.get("total_value", {}).get("value", 0)) or 0
            return out
    log(NAME, f"couldn't read insights for {media_id} ({last}); check the token has "
              "instagram_business_manage_insights")
    return None


def score(m: dict) -> float:
    views = m.get("views") or m.get("reach") or 0
    eng = 3 * m.get("shares", 0) + 3 * m.get("saved", 0) + 2 * m.get("comments", 0) + m.get("likes", 0)
    # engagement per 1,000 views, plus a little credit for raw reach
    return round((eng / max(views, 1)) * 1000 + views / 1000, 2)


def run(ctx: Context, memory: Memory) -> None:
    token = os.getenv("IG_ACCESS_TOKEN", "")
    if ctx.mock or not token:
        log(NAME, "skipped (no Instagram token / mock mode)")
        return
    ig = ctx.cfg["instagram"]
    graph = f"https://{os.getenv('IG_API_HOST') or ig['api_host']}/{ig['api_version']}"
    now = datetime.now(timezone.utc)

    updated = 0
    for p in memory.posts:
        if not p.get("media_id") or not p.get("posted_at"):
            continue
        posted = datetime.fromisoformat(p["posted_at"])
        if not (timedelta(hours=24) <= now - posted <= timedelta(days=14)):
            continue
        last = p.get("metrics_at")
        if last and now - datetime.fromisoformat(last) < timedelta(hours=20):
            continue
        try:
            m = _fetch_insights(graph, p["media_id"], token)
        except requests.RequestException:
            m = None
        if m:
            p["metrics"], p["score"] = m, score(m)
            p["metrics_at"] = now.isoformat(timespec="seconds")
            updated += 1

    measured = [p for p in memory.posts if p.get("score") is not None]
    if not measured:
        log(NAME, f"no measured posts yet ({len(memory.posts)} posted)")
        return

    # Per-format averages feed the Researcher's format choice.
    fmt: dict[str, dict] = {}
    for p in measured:
        f = fmt.setdefault(p["format"], {"n": 0, "total": 0.0})
        f["n"] += 1
        f["total"] += p["score"]
    memory.data["format_scores"] = {k: {"n": v["n"], "avg": round(v["total"] / v["n"], 2)} for k, v in fmt.items()}

    ranked = sorted(measured, key=lambda p: p["score"], reverse=True)
    top, bottom = ranked[:3], ranked[-3:] if len(ranked) > 3 else []

    def row(p: dict) -> str:
        m = p.get("metrics", {})
        return (f"- [{p['format']}] \"{p['title']}\" · hook: \"{p.get('hook', '')}\" · "
                f"views {m.get('views', m.get('reach', '?'))}, shares {m.get('shares', 0)}, "
                f"saves {m.get('saved', 0)}, comments {m.get('comments', 0)}")

    learnings = "Best performers:\n" + "\n".join(row(p) for p in top)
    if bottom:
        learnings += "\nWeakest performers:\n" + "\n".join(row(p) for p in bottom)
    learnings += "\nFormat averages (higher is better): " + json.dumps(memory.data["format_scores"])

    if len(measured) >= 4:
        summary = ctx.llm.json(
            f"""You analyse a horror Reels account. From this data, write 3-5 short, concrete lessons
about which settings, hooks, twists and formats the audience responds to, and what to avoid.

{learnings}

Return JSON: {{"lessons": ["...", "..."]}}""", temperature=0.4, mock={"lessons": []})
        if summary.get("lessons"):
            learnings += "\nLessons:\n" + "\n".join("- " + l for l in summary["lessons"])

    memory.data["learnings"] = learnings
    log(NAME, f"refreshed {updated} post(s); {len(measured)} measured; learnings updated")
