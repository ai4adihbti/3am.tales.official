"""Researcher agent: decides what today's reel is.

- A story in progress → today is simply its next part (no AI call needed).
- Otherwise a new story starts. The STORY TYPE stays the same for a run of 2-3 reels, then changes to a
  different type. Types unlock gradually as the channel grows (config `unlock_after`), and once there is
  views/shares data the better-performing types are chosen more often.
- The agent itself invents the story and decides how many parts it needs (1-3, fitted to the run), and every new story must be
  different from the recent ones in region, setting, kind of legend and hero ("gradual diversification").
"""
from __future__ import annotations

import random

from .common import SAFETY_RULES, Context, log
from .memory import Memory

NAME = "Researcher"


def _available(ctx: Context, memory: Memory) -> list[dict]:
    n = len(memory.posts)
    return [f for f in ctx.cfg["formats"] if int(f.get("unlock_after", 0)) <= n] or ctx.cfg["formats"][:1]


def choose_format(ctx: Context, memory: Memory) -> tuple[dict, int]:
    """Returns (story type, the most parts the new story may have so the type's run stays 2-3 reels)."""
    formats = _available(ctx, memory)
    lo, hi = ctx.cfg.get("format_block_reels", [2, 3])
    max_parts = int(ctx.cfg["story"]["parts_per_series"][1])
    history = [p.get("format") for p in memory.posts]
    block = memory.data.get("format_block") or {}

    # 0. the very first story is in the channel's main (reference) style
    main_id = ctx.cfg.get("main_format")
    if not history:
        first = next((f for f in formats if f["id"] == main_id), formats[0])
        memory.data["format_block"] = {"format": first["id"], "target": random.randint(int(lo), int(hi))}
        log(NAME, f"first story type: {first['label']}")
        return first, max_parts

    # 1. keep the current story type until its run of 2-3 reels is complete
    cur = history[-1] if history else None
    run = 0
    for f in reversed(history):
        if f != cur:
            break
        run += 1
    cur_fmt = next((f for f in formats if f["id"] == cur), None)
    if cur_fmt and block.get("format") == cur and run < int(block.get("target", lo)):
        left = int(block["target"]) - run
        log(NAME, f"story type stays: {cur_fmt['label']} (reel {run + 1} of this {block['target']}-reel run)")
        return cur_fmt, max(1, min(max_parts, left))

    # 2. switch to a different type: new/rarely used types first, performance and the main style as bonuses
    scores = memory.data.get("format_scores", {})
    cands = [f for f in formats if f["id"] != cur] or formats
    weights = []
    for f in cands:
        last = max((i for i, x in enumerate(history) if x == f["id"]), default=-1)
        since = len(history) - last if last >= 0 else 99                     # reels since last used
        w = min(since, 30) / 10 + (1.5 if last < 0 else 0)                   # freshness
        sc = scores.get(f["id"], {})
        if sc.get("n", 0) >= 2:
            best = max((v["avg"] for v in scores.values() if v.get("n", 0) >= 2), default=1) or 1
            w *= 0.5 + sc["avg"] / best                                       # 0.5x … 1.5x by performance
        if f["id"] == main_id:
            w *= 1.6
        weights.append(max(w, 0.05))
    choice = random.choices(cands, weights=weights, k=1)[0]
    target = random.randint(int(lo), int(hi))
    memory.data["format_block"] = {"format": choice["id"], "target": target}
    log(NAME, f"new story type: {choice['label']} for the next {target} reels "
              f"({len(formats)} of {len(ctx.cfg['formats'])} types unlocked)")
    return choice, min(max_parts, target)


def _diversity_rules(memory: Memory) -> str:
    stories = memory.data.get("story_log", [])
    n = len(stories)
    if n < 3:
        stage = ("The channel is new: stay close to its proven style (emotional Indian family horror, village or "
                 "small-town, storyteller kahani), but still make every story fresh.")
    elif n < 9:
        stage = ("The channel is growing: keep the core feeling, but change at least TWO big elements from recent "
                 "stories — a new region of India, a new kind of legend or entity, a new kind of hero.")
    else:
        stage = ("The audience knows our style now: surprise them. Change at least THREE big elements: region "
                 "(Rajasthan, Bengal, Kerala, Himachal, Bihar, Punjab, Konkan, the Northeast, Bundelkhand …), era "
                 "(today, the 1990s, 1970s), hero (child, teenager, bahu, old man, a couple, a working woman), kind of "
                 "mystery (spirit, curse, time slip, a living person's secret), and emotion (love, guilt, sacrifice, "
                 "revenge, longing).")
    recent = "\n".join(f"- \"{s.get('title')}\" [{s.get('format')}] · region: {s.get('region', '?')} · setting: "
                        f"{s.get('setting', '?')} · legend/entity: {s.get('entity', '?')} · hero: "
                        f"{s.get('protagonist', '?')} · emotion: {s.get('emotion', '?')}"
                        for s in reversed(stories[-8:])) or "- (none yet)"
    return f"""Diversity (newest first):
{recent}
Rules: the new story must differ from the last 3 above in region, setting, kind of legend/entity and hero, and
must not reuse their twist. {stage}"""


def _parts_rule(lo: int, hi: int) -> str:
    if hi <= 1:
        return "This story must be complete in ONE reel (about 2.5 minutes, never over 2:50): a full arc with a strong twist at the end."
    return (f"Decide the number of parts YOURSELF ({lo}-{hi}), by what this story truly needs: 1 if it's a single "
            "punchy tale; more only if each part is a full chapter with its own escalation and a cliffhanger. "
            "Never pad a story to make more parts — weak middle parts lose followers. Each part is about 2.5 minutes (never over 2:50).")


INDIAN_RULE = """FOR AN INDIAN AUDIENCE, ALWAYS: set in India (a real Indian state/region, invented village or town
name), Indian characters with Indian names, Indian family relationships and values, Indian beliefs, festivals,
food, clothes and everyday details, an Indian folk legend or belief (chudail, pichhal-pairi, brahmarakshas, daayan,
pret, munjya, yakshi, nishi daak, a cursed well or tree …) — never Western ghosts, names or settings. Told in
Hindi-heavy Hinglish."""


def _voice_style(ctx: Context, memory: Memory) -> dict:
    """Alternate between a single storyteller and acted dialogue, and between male/female storytellers."""
    mode = ctx.cfg["voice"].get("style", "mix")
    last = memory.data.get("last_voice_style")
    if mode in ("narrator", "dialogue"):
        style = mode
    else:
        style = "dialogue" if last == "narrator" else "narrator"
    pref = ctx.cfg["voice"].get("narrator_gender", "female")
    if pref in ("female", "male"):
        gender = pref
    else:
        gender = "male" if memory.data.get("last_narrator_gender") == "female" else "female"
    return {"voice_style": style, "narrator_gender": gender}


def todays_series(ctx: Context, memory: Memory, avoid_extra: list[str] | None = None) -> tuple[dict, dict]:
    """Returns (series, format). `series` has title, part, total_parts, outline, characters, previously."""
    active = memory.active_series()
    if active and not avoid_extra:
        fmt = next((f for f in ctx.cfg["formats"] if f["id"] == active["format"]), ctx.cfg["formats"][0])
        series = {**active, "part": active["parts_done"] + 1}
        log(NAME, f'continuing "{series["title"]}" → part {series["part"]}/{series["total_parts"]}')
        return series, fmt

    fmt, max_parts = choose_format(ctx, memory)
    lo = int(ctx.cfg["story"]["parts_per_series"][0])
    hi = max(lo, max_parts)
    ch = ctx.cfg["channel"]
    avoid = memory.recent_titles() + (avoid_extra or [])
    learnings = memory.data.get("learnings") or "No performance data yet — go with your best instinct."
    prompt = f"""You are the head of story for "{ch['name']}", an Instagram account of illustrated Hindi horror
and mystery stories, posted one ~2.5-minute reel a day (it must stay under 3 minutes to be a YouTube Short too); a story can be one reel or a series of parts that each end
on a cliffhanger. Audience: {ch['audience']}. Quality of the story is everything.

Story type for the new story: {fmt['label']} — {fmt['brief']}

What our analytics say so far:
{learnings}

{_diversity_rules(memory)}

Titles ALREADY done (never repeat or closely imitate):
{chr(10).join('- ' + t for t in avoid[-60:]) or '- (none yet)'}

Brainstorm 6 very different story ideas internally, judge each like a viewer scrolling at 1 AM (would they stop,
feel something, and wait for the next part?), then return ONLY the strongest one.
Strong = a concrete Indian setting young Indians recognise, a local legend or rule with one unforgettable creepy
image, 2-4 characters we care about, an emotional core, escalating events, and a twist that lands.
Use fresh Indian place and character names (invent the village/town names); vary the region of India.

{_parts_rule(lo, hi)}

{INDIAN_RULE}

{SAFETY_RULES}

Return JSON:
{{"title": "Hinglish title, e.g. 'Bheempura ka Qabristan'", "total_parts": {lo if lo == hi else f"a number {lo}-{hi}"},
  "parts_reason": "why this many parts",
  "region": "state/region of India", "setting": "...", "entity": "the legend / entity / mystery in a few words",
  "protagonist": "hero: age, gender, relation", "emotion": "the emotional core in a few words",
  "legend": "the local legend in 2 sentences",
  "outline": ["part 1 in 1-2 sentences (ending on its cliffhanger if more parts follow)", "…"],
  "final_twist": "...", "why_it_will_work": "1 sentence"}}"""

    idea = ctx.llm.json(prompt, temperature=1.0, mock={
        "title": "Bheempura ka Qabristan", "total_parts": 2, "parts_reason": "two clear chapters",
        "region": "Madhya Pradesh", "setting": "a college town near a village with an abandoned hill cemetery",
        "entity": "a voice that calls names", "protagonist": "19-year-old college boy", "emotion": "friendship",
        "legend": "Whoever enters the cemetery after midnight hears their name called from inside.",
        "outline": ["Friends decide to go; an old man warns them; a handprint appears. Cliffhanger: a figure behind them.",
                    "Inside the cemetery they find a grave with one of their names. Twist."],
        "final_twist": "The old man died years ago.", "why_it_will_work": "College dares are relatable."})
    total = idea.get("total_parts", 2)
    try:
        total = max(lo, min(hi, int(total)))
    except (TypeError, ValueError):
        total = min(2, hi)
    outline = list(idea.get("outline") or [])[:total]
    total = max(lo, min(total, len(outline) or total))      # parts can't exceed the outline the agent wrote
    series = {"id": memory.new_series_id(idea.get("title", "series")), "title": idea.get("title", "Untitled"),
              "format": fmt["id"], "total_parts": total, "part": 1, "parts_done": 0,
              "outline": outline, "characters": None, "previously": [],
              "concept": {k: idea.get(k) for k in ("setting", "legend", "final_twist")},
              "profile": {k: idea.get(k) for k in ("region", "setting", "entity", "protagonist", "emotion")}}
    series.update(_voice_style(ctx, memory))
    log(NAME, f'new story "{series["title"]}" · {total} part{"s" if total > 1 else ""} '
              f'({idea.get("parts_reason", "")[:80]}) · {idea.get("region", "")} · '
              f'{"storyteller (" + series["narrator_gender"] + ")" if series["voice_style"] == "narrator" else "character dialogue"}')
    return series, fmt
