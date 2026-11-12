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
        if r.status_code == 400 and "does not exist" in r.text:
            return {"_deleted": True}          # the post was deleted on Instagram
        if r.status_code == 200:
            out = {}
            for item in r.json().get("data", []):
                vals = item.get("values") or [{}]
                out[item["name"]] = vals[0].get("value", item.get("total_value", {}).get("value", 0)) or 0
            return out
    log(NAME, f"couldn't read insights for {media_id} ({last}); check the token has "
              "instagram_business_manage_insights")
    return None


def _watch_time(graph: str, media_id: str, token: str) -> dict:
    """Average watch time (Instagram's #1 ranking signal). Asked separately: older API versions may not have it."""
    try:
        r = requests.get(f"{graph}/{media_id}/insights",
                         params={"metric": "ig_reels_avg_watch_time", "access_token": token}, timeout=30)
        if r.status_code != 200:
            return {}
        for item in r.json().get("data", []):
            vals = item.get("values") or [{}]
            v = vals[0].get("value", item.get("total_value", {}).get("value", 0)) or 0
            return {"avg_watch_s": round(float(v) / 1000, 1)}          # Instagram reports milliseconds
    except (requests.RequestException, ValueError, TypeError):
        pass
    return {}


def score(m: dict, duration: float | None = None) -> float:
    """What the algorithm rewards: people watching to the end and sending it on, then saves/comments/likes."""
    views = m.get("views") or m.get("reach") or 0
    eng = 5 * m.get("shares", 0) + 3 * m.get("saved", 0) + 2 * m.get("comments", 0) + m.get("likes", 0)
    out = (eng / max(views, 1)) * 1000 + views / 1000
    if m.get("avg_watch_s") and duration:
        out += 100 * min(1.5, m["avg_watch_s"] / duration)        # completion counts most
    return round(out, 2)


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
        if not p.get("media_id") or not p.get("posted_at") or p.get("deleted"):
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
        if m and m.get("_deleted"):
            p["deleted"] = True
            log(NAME, f"post \"{p.get('title', p['media_id'])}\" no longer exists on Instagram; skipping it from now on")
            continue
        if m:
            if not m.get("_deleted"):
                m.update(_watch_time(graph, p["media_id"], token))
            p["metrics"], p["score"] = m, score(m, p.get("duration_s"))
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

    # YouTube Shorts: public views/likes/comments for the last 30 posts (cheap: one API call)
    try:
        from . import youtube
        recent = [p for p in memory.posts[-30:] if p.get("youtube_id")]
        yt = youtube.stats([p["youtube_id"] for p in recent])
        for p in recent:
            if p["youtube_id"] in yt:
                p["youtube_stats"] = yt[p["youtube_id"]]
        if yt:
            total = sum(v["viewCount"] for v in yt.values())
            log(NAME, f"YouTube: {total} views on the last {len(yt)} Shorts")
    except Exception as e:  # never let this break the run
        log(NAME, f"YouTube stats skipped ({str(e)[:100]})")

    # Art-style A/B test (v7.4.0): average watch seconds and reach per style, posts from 8 Nov on
    sty: dict[str, dict] = {}
    for p in measured:
        if "style" not in p:
            continue
        m = p.get("metrics", {})
        d = sty.setdefault(p["style"], {"n": 0, "watch": 0.0, "reach": 0.0})
        d["n"] += 1
        d["watch"] += float(m.get("avg_watch_s") or 0)
        d["reach"] += float(m.get("reach") or 0)
    memory.data["style_scores"] = {k: {"n": d["n"], "avg_watch_s": round(d["watch"] / d["n"], 1),
                                       "avg_reach": round(d["reach"] / d["n"])} for k, d in sty.items()}
    if sty:
        log(NAME, f"art-style test: {memory.data['style_scores']}")

    ranked = sorted(measured, key=lambda p: p["score"], reverse=True)
    top, bottom = ranked[:3], ranked[-3:] if len(ranked) > 3 else []

    def row(p: dict) -> str:
        m = p.get("metrics", {})
        return (f"- [{p['format']}] \"{p['title']}\" · hook: \"{p.get('hook', '')}\" · "
                f"views {m.get('views', m.get('reach', '?'))}, shares {m.get('shares', 0)}, "
                f"saves {m.get('saved', 0)}, comments {m.get('comments', 0)}"
                + (f", watched on average {m['avg_watch_s']}s of {p.get('duration_s', 0):.0f}s "
                   f"({100 * m['avg_watch_s'] / max(1, p.get('duration_s') or 1):.0f}%)" if m.get("avg_watch_s") else ""))

    learnings = "Best performers:\n" + "\n".join(row(p) for p in top)
    if bottom:
        learnings += "\nWeakest performers:\n" + "\n".join(row(p) for p in bottom)
    learnings += "\nStory-type average SCORES (a points score, not seconds; higher is better): " + json.dumps(memory.data["format_scores"])

    if len(measured) >= 4:
        summary = ctx.llm.json(
            f"""You analyse a horror Reels account. From this data, write 3-5 short, concrete lessons
about which settings, hooks, twists and formats the audience responds to, and what to avoid.
Watch time (how much of the reel people watch) and shares matter most to Instagram and YouTube.

{learnings}

Return JSON: {{"lessons": ["...", "..."]}}""", temperature=0.4, light=True, mock={"lessons": []})
        if summary.get("lessons"):
            learnings += "\nLessons:\n" + "\n".join("- " + l for l in summary["lessons"])

    memory.data["learnings"] = learnings
    log(NAME, f"refreshed {updated} post(s); {len(measured)} measured; learnings updated")
