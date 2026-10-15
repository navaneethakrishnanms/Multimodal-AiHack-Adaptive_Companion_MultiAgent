"""Learner model — Bayesian Knowledge Tracing style mastery per topic.

State per (learner, topic):
  p          — P(knows the skill), updated after every graded answer
  attempts, correct — counters for the dashboard (attempts == 0 means "never probed")
  history    — recent events (for spacing / forgetting curves)
  misconceptions — wrong-idea notes collected from missed questions

Update rule (standard BKT):
  P(L_n | correct)   = P(L_{n-1})(1-slip) / [P(L_{n-1})(1-slip) + (1-P(L_{n-1}))guess]
  P(L_n | incorrect) = P(L_{n-1})slip / [P(L_{n-1})slip + (1-P(L_{n-1}))(1-guess)]
  P(L_{n+1}) = P(L_n) + (1-P(L_n)) * learn   (after each opportunity)

Question-type aware guess/slip priors:
  mcq (4 options)  guess=0.25  slip=0.10
  short/numerical  guess=0.02  slip=0.08

Mastery only ever moves because of answer evidence (quiz or probe answers); reading
material, chatting or the blame heuristics in core/rootcause.py never change it.
`LearnerModel(None)` is an in-memory model (used by the simulator and unit tests).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

PRIOR = 0.25
WEAK = 0.6
STRONG = 0.85

PRIORS = {
    "mcq": {"guess": 0.25, "slip": 0.10},
    "short": {"guess": 0.02, "slip": 0.08},
    "numerical": {"guess": 0.02, "slip": 0.08},
}
LEARN = 0.12


def bkt_update(p: float, correct: bool, qtype: str = "mcq", learn: float = LEARN) -> float:
    """One BKT step; pure function so simulators can share the exact production math."""
    pr = PRIORS.get(qtype, PRIORS["mcq"])
    g, s = pr["guess"], pr["slip"]
    if correct:
        num = p * (1 - s)
        den = num + (1 - p) * g
    else:
        num = p * s
        den = num + (1 - p) * (1 - g)
    post = num / den if den > 0 else p
    return round(min(max(post + (1 - post) * learn, 0.01), 0.995), 4)


def _new_record(prior: float) -> dict:
    return {"p": prior, "attempts": 0, "correct": 0, "history": [], "misconceptions": [], "last_seen": 0}


class LearnerModel:
    def __init__(self, store: Path | None = None, prior: float = PRIOR):
        self.store = store
        self.prior = prior
        self.data: dict[str, dict] = {}
        if store is not None and store.exists():
            self.data = json.loads(store.read_text(encoding="utf-8"))

    # ------------------------------------------------------------- core
    def _topic(self, topic: str) -> dict:
        return self.data.setdefault(topic, _new_record(self.prior))

    def get(self, topic: str) -> float:
        """Current mastery estimate; unseen topics sit at the prior."""
        rec = self.data.get(topic)
        return rec["p"] if rec else self.prior

    def attempts(self, topic: str) -> int:
        rec = self.data.get(topic)
        return rec["attempts"] if rec else 0

    def update(self, topic: str, correct: bool, qtype: str = "mcq", source: str = "quiz") -> float:
        rec = self._topic(topic)
        rec["p"] = bkt_update(rec["p"], correct, qtype)
        rec["attempts"] += 1
        rec["correct"] += int(correct)
        now = int(time.time())
        rec["last_seen"] = now
        rec["history"].append({"t": now, "correct": bool(correct), "type": qtype, "src": source})
        rec["history"] = rec["history"][-50:]
        self.save()
        return rec["p"]

    def seed_prior(self, topic: str, p: float) -> None:
        """Intake conversation: set the starting belief for a topic that has no answers yet."""
        rec = self._topic(topic)
        if rec["attempts"] == 0:
            rec["p"] = round(min(max(p, 0.05), 0.6), 4)
            self.save()

    def touch(self, topic: str) -> None:
        """Student read/discussed the topic (chat, repair material). Exposure only: mastery is unchanged."""
        self._topic(topic)["last_seen"] = int(time.time())
        self.save()

    def record_misconception(self, topic: str, note: str) -> None:
        note = (note or "").strip()
        if not note:
            return
        rec = self._topic(topic)
        notes = rec.setdefault("misconceptions", [])
        if note not in notes:
            notes.append(note)
            rec["misconceptions"] = notes[-5:]
            self.save()

    def mastery(self) -> dict[str, float]:
        return {t: r["p"] for t, r in self.data.items()}

    def weak_topics(self, threshold: float = WEAK) -> list[str]:
        return [t for t, r in self.data.items() if r["attempts"] and r["p"] < threshold]

    def strong_topics(self, threshold: float = STRONG) -> list[str]:
        return [t for t, r in self.data.items() if r["attempts"] and r["p"] >= threshold]

    def save(self) -> None:
        if self.store is None:
            return
        self.store.parent.mkdir(parents=True, exist_ok=True)
        self.store.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    def reset(self) -> None:
        self.data = {}
        self.save()

    # --------------------------------------------------------- study plan
    def retention(self, topic: str, now: float | None = None) -> float:
        """Ebbinghaus-style retention estimate; stronger mastery decays more slowly."""
        rec = self.data.get(topic)
        p = rec["p"] if rec else self.prior
        last = (rec.get("last_seen") or (rec["history"][-1]["t"] if rec["history"] else 0)) if rec else 0
        if not last:
            return 0.0
        days = ((now or time.time()) - last) / 86400
        return math.exp(-days / max(1.5, 6 * p))

    def study_plan(self, all_topics: list[str], days_to_exam: int = 7, minutes_per_day: int = 45) -> list[dict]:
        """Spaced schedule: priority = (1 - mastery) x forgetting risk, spread over the days left.

        Weak topics are placed early and repeated (spaced review) later in the window when
        time allows, mirroring expanding-interval review.
        """
        now = time.time()
        items = []
        for t in all_topics:
            p = self.get(t)
            ret = self.retention(t, now)
            priority = (1 - p) * (1.2 - min(ret, 1.0))
            items.append({"topic": t, "mastery": p, "retention": round(ret, 3), "priority": round(priority, 3)})
        items.sort(key=lambda x: -x["priority"])
        days_to_exam = max(days_to_exam, 1)
        per_day = max(1, math.ceil(len(items) / days_to_exam))
        plan = []
        for d in range(days_to_exam):
            chunk = items[d * per_day : (d + 1) * per_day]
            review = []
            if d >= 2 and plan:  # expanding-interval revisit of the weakest earlier topics
                earlier = [c for day in plan[:-1] for c in day["_items"] if c["mastery"] < WEAK]
                review = [c for c in earlier[:2] if c["topic"] not in {x["topic"] for x in chunk}]
            if not chunk and not review:
                break
            share = max(10, minutes_per_day // max(len(chunk) + len(review), 1))
            plan.append(
                {
                    "day": d + 1,
                    "topics": [c["topic"] for c in chunk],
                    "review": [c["topic"] for c in review],
                    "minutes_each": share,
                    "_items": chunk,
                }
            )
        for day in plan:
            day.pop("_items", None)
        return plan

    # -------------------------------------------------------- diagnostics
    def report(self) -> dict:
        weak = self.weak_topics()
        strong = self.strong_topics()
        return {
            "topics": {
                t: {
                    "mastery": r["p"],
                    "attempts": r["attempts"],
                    "accuracy": round(r["correct"] / r["attempts"], 3) if r["attempts"] else None,
                    "retention": round(self.retention(t), 3),
                    "misconceptions": r.get("misconceptions", []),
                }
                for t, r in self.data.items()
            },
            "weak_topics": weak,
            "strong_topics": strong,
        }
