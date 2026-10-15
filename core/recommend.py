"""Next-best-action recommendations — every action carries an explicit WHY and source links.

Inspired by the "Next Best Action" idea in the LearnFlow AI project, rebuilt on our learner model:
the reasons come from real numbers (BKT mastery, prerequisite blame, forgetting-curve retention,
recorded misconceptions), never from templates that ignore the data.
"""
from __future__ import annotations

import time

from .learner import STRONG, WEAK, LearnerModel
from .rootcause import blame_scores, should_probe


def recommend(
    learner: LearnerModel,
    graph: dict[str, list[str]],
    topics: list[str],
    sources,
    limit: int = 8,
) -> list[dict]:
    now = time.time()
    recs: list[dict] = []

    def add(priority: int, kind: str, topic: str, title: str, why: str, **extra) -> None:
        recs.append({"priority": priority, "action": kind, "topic": topic, "title": title,
                     "why": why, "sources": sources(topic, 2), **extra})

    if not any(r["attempts"] for r in learner.data.values()):
        first = topics[0] if topics else ""
        return [{"priority": 1, "action": "diagnostic", "topic": first, "title": "Take the diagnostic quiz",
                 "why": "No answers yet, so mastery is only the prior (25%). A short diagnostic across the course "
                        "tells the tutor where you really stand.", "sources": []}]

    for t in topics:
        rec = learner.data.get(t)
        m, n = learner.get(t), learner.attempts(t)
        if n and m < WEAK:
            blame = blame_scores(graph, t, learner)
            if blame and should_probe(blame):
                b = blame[0]
                state = "has never been tested" if not b["probed"] else f"is at {b['mastery']:.0%}"
                add(1, "rootcause", t, f"Find the root cause behind “{t}”",
                    f"“{t}” is at {m:.0%} after {n} answer(s). Its prerequisite “{b['topic']}” {state} "
                    f"(blame {b['blame']:.0%}), so the gap may be there rather than in “{t}” itself.",
                    suspect=b["topic"])
            else:
                add(2, "review", t, f"Review “{t}”",
                    f"Mastery is {m:.0%} after {n} answer(s) and there is no weaker prerequisite to blame.")
        for note in (rec or {}).get("misconceptions", [])[-1:]:
            if m < STRONG:
                add(2, "misconception", t, f"Clear a misconception in “{t}”", f"Recorded from a missed question: {note}")
        if n and m >= WEAK:
            ret = learner.retention(t, now)
            if ret < 0.5:
                days = (now - (rec.get("last_seen") or 0)) / 86400 if rec else 0
                add(3, "spaced-review", t, f"Spaced review: “{t}”",
                    f"You know it ({m:.0%}) but the forgetting curve puts retention at ~{ret:.0%} "
                    f"({days:.0f} day(s) since you last touched it).")
        if n == 0:
            prereqs = graph.get(t, [])
            if all(learner.attempts(p) > 0 and learner.get(p) >= WEAK for p in prereqs):
                why = (f"You have the prerequisites ({', '.join(prereqs)}) covered." if prereqs
                       else "It is a foundation topic you have not started.")
                add(4, "learn-next", t, f"Next up: “{t}”", why)
        if n >= 3 and m >= STRONG:
            add(5, "challenge", t, f"Challenge quiz on “{t}”",
                f"Mastery {m:.0%} over {n} answers: time for hard questions.", difficulty="hard")

    recs.sort(key=lambda r: (r["priority"], r["topic"]))
    seen, out = set(), []
    for r in recs:  # one action per (kind, topic)
        k = (r["action"], r["topic"])
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out[:limit]
