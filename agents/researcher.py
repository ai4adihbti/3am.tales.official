"""Researcher agent: decides what today's reel is.

If a series is in progress, today is simply its next part (no AI call needed).
Otherwise it picks a format (rotating, then favouring what performs best) and plans a
brand-new 2-3 part series that has never been done on the account.
"""
from __future__ import annotations

import random

from .common import SAFETY_RULES, Context, log
from .memory import Memory

NAME = "Researcher"


def choose_format(ctx: Context, memory: Memory) -> dict:
    formats = ctx.cfg["formats"]
    scores = memory.data.get("format_scores", {})
    scored = {f["id"]: scores[f["id"]] for f in formats if f["id"] in scores and scores[f["id"]].get("n", 0) >= 2}

    if len(scored) >= 2 and random.random() < 0.7:
        weights = [max(0.1, scored.get(f["id"], {}).get("avg", 0.0)) + 0.1 for f in formats]
        choice = random.choices(formats, weights=weights, k=1)[0]
        log(NAME, f"format by performance → {choice['label']}")
        return choice

    history = [p.get("format") for p in memory.posts]

    def last_used(f: dict) -> int:
        return max((i for i, x in enumerate(history) if x == f["id"]), default=-1)

    choice = min(formats, key=last_used)
    log(NAME, f"format by rotation → {choice['label']}")
    return choice


def _voice_style(ctx: Context, memory: Memory) -> dict:
    """Alternate between a single storyteller and acted dialogue, and between male/female storytellers."""
    mode = ctx.cfg["voice"].get("style", "mix")
    last = memory.data.get("last_voice_style")
    if mode in ("narrator", "dialogue"):
        style = mode
    else:
        style = "dialogue" if last == "narrator" else "narrator"
    last_g = memory.data.get("last_narrator_gender")
    gender = "male" if last_g == "female" else "female"
    return {"voice_style": style, "narrator_gender": gender}


def todays_series(ctx: Context, memory: Memory, avoid_extra: list[str] | None = None) -> tuple[dict, dict]:
    """Returns (series, format). `series` has title, part, total_parts, outline, characters, previously."""
    active = memory.active_series()
    if active and not avoid_extra:
        fmt = next((f for f in ctx.cfg["formats"] if f["id"] == active["format"]), ctx.cfg["formats"][0])
        series = {**active, "part": active["parts_done"] + 1}
        log(NAME, f'continuing "{series["title"]}" → part {series["part"]}/{series["total_parts"]}')
        return series, fmt

    fmt = choose_format(ctx, memory)
    lo, hi = ctx.cfg["story"]["parts_per_series"]
    ch = ctx.cfg["channel"]
    avoid = memory.recent_titles() + (avoid_extra or [])
    learnings = memory.data.get("learnings") or "No performance data yet — go with your best instinct."
    prompt = f"""You are the head of story for "{ch['name']}", an Instagram account of illustrated Hindi horror
story series. Each story is told in {lo}-{hi} daily parts of ~2.5 minutes, each ending on a cliffhanger.
Audience: {ch['audience']}.

Format for the new series: {fmt['label']} — {fmt['brief']}

What our analytics say so far:
{learnings}

Series we have ALREADY done (do not repeat or closely imitate):
{chr(10).join('- ' + t for t in avoid[-60:]) or '- (none yet)'}

Brainstorm 5 distinct series ideas internally, then return ONLY the strongest one.
Strong = an Indian setting every young Indian recognises, a local legend or rule with a concrete creepy
image, 2-4 characters with distinct personalities, a cliffhanger for each part, and a final twist.
Use fresh Indian place and character names (invent the village/town names).

{SAFETY_RULES}

Return JSON:
{{"title": "Hinglish series title, e.g. 'Bheempura ka Qabristan'", "total_parts": {lo} to {hi},
  "setting": "...", "legend": "the local legend in 2 sentences",
  "outline": ["part 1 in 1-2 sentences ending on its cliffhanger", "part 2 …", "…"],
  "final_twist": "...", "why_it_will_work": "1 sentence"}}"""

    idea = ctx.llm.json(prompt, temperature=1.0, mock={
        "title": "Bheempura ka Qabristan", "total_parts": 2,
        "setting": "a college town near a village with an abandoned hill cemetery",
        "legend": "Whoever enters the cemetery after midnight hears their name called from inside.",
        "outline": ["Friends decide to go; an old man warns them; a handprint appears. Cliffhanger: a figure behind them.",
                    "Inside the cemetery they find a grave with one of their names. Twist."],
        "final_twist": "The old man died years ago.", "why_it_will_work": "College dares are relatable."})
    total = idea.get("total_parts", lo)
    total = max(lo, min(hi, int(total))) if str(total).isdigit() else lo
    outline = list(idea.get("outline") or [])[:total]
    series = {"id": memory.new_series_id(idea.get("title", "series")), "title": idea.get("title", "Untitled"),
              "format": fmt["id"], "total_parts": total, "part": 1, "parts_done": 0,
              "outline": outline, "characters": None, "previously": [],
              "concept": {k: idea.get(k) for k in ("setting", "legend", "final_twist")}}
    series.update(_voice_style(ctx, memory))
    log(NAME, f'new series "{series["title"]}" ({total} parts) · '
              f'{"storyteller (" + series["narrator_gender"] + ")" if series["voice_style"] == "narrator" else "character dialogue"}')
    return series, fmt
