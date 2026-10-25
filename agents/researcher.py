"""Researcher agent: picks today's format and a fresh story idea.

It reads the Analyst's learnings (what worked, what flopped) and the list of
past titles, so it never repeats itself and leans into what the audience likes.
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

    # Exploit: once there's enough data, usually pick by performance (70%)…
    if len(scored) >= 2 and random.random() < 0.7:
        weights = [max(0.1, scored.get(f["id"], {}).get("avg", 0.0)) + 0.1 for f in formats]
        choice = random.choices(formats, weights=weights, k=1)[0]
        log(NAME, f"format by performance → {choice['label']}")
        return choice

    # …otherwise explore: the format used least recently.
    history = [p.get("format") for p in memory.posts]

    def last_used(f: dict) -> int:
        # index of the most recent post in this format; -1 if never used
        return max((i for i, x in enumerate(history) if x == f["id"]), default=-1)

    choice = min(formats, key=last_used)
    log(NAME, f"format by rotation → {choice['label']}")
    return choice


def find_idea(ctx: Context, memory: Memory, fmt: dict, avoid_extra: list[str] | None = None) -> dict:
    ch = ctx.cfg["channel"]
    avoid = memory.recent_titles() + (avoid_extra or [])
    learnings = memory.data.get("learnings") or "No performance data yet — go with your best instinct."

    prompt = f"""You are the head of research for "{ch['name']}", a faceless Instagram Reels account.
Audience: {ch['audience']}.

Today's format: {fmt['label']} — {fmt['brief']}

What our analytics say so far:
{learnings}

Titles we have ALREADY posted (do not repeat or closely imitate):
{chr(10).join('- ' + t for t in avoid[-60:]) or '- (none yet)'}

The story will be performed as a DIALOGUE between exactly two Indian characters (no narrator),
so the idea must work through conversation alone: what they say, hear and notice.

Brainstorm 5 distinct ideas internally, then return ONLY the single strongest one.
Strong = a setting every young Indian recognises (hostel/PG, sleeper train, Ola/Uber at night,
nani's old house in the village, a shaadi at a haveli, night shift, Zomato delivery, college trip to
Shimla/Rishikesh/Rajasthan, a society lift, a power cut), a concrete creepy detail, and a twist that
recontextualises everything. Use fresh Indian names (not Rahul/Priya every time). Avoid clichés unless
you subvert them.

{SAFETY_RULES}

Return JSON:
{{"title": "short internal title", "setting": "...", "characters": "who the two people are, 1 line each",
  "premise": "2 sentences", "creepy_detail": "the one moment people will remember",
  "twist": "the final reveal", "why_it_will_work": "1 sentence"}}"""

    idea = ctx.llm.json(prompt, temperature=1.0, mock={
        "title": "Hostel ka Teesra Bed",
        "setting": "girls' PG in Pune during a power cut, 3 AM",
        "characters": "Ishita, 20, sleepy and practical; Tanvi, 21, her roommate, on a call from home",
        "premise": "Ishita calls Tanvi because she hears someone breathing on the empty third bed. Tanvi keeps telling her to ignore it.",
        "creepy_detail": "Tanvi describes exactly what Ishita is wearing in the dark",
        "twist": "Tanvi went home last week — and the third bed has been empty since a girl left it a year ago",
        "why_it_will_work": "Every PG student knows the empty bed nobody talks about.",
    })
    idea["format"] = fmt["id"]
    log(NAME, f"idea → {idea['title']}")
    return idea
