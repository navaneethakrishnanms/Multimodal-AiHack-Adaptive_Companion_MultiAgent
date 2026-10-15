"""Learner model — Bayesian Knowledge Tracing style mastery per topic.

State per (learner, topic):
  p_mastery  — P(knows the skill), updated after every graded answer
  attempts, correct — counters for the dashboard
  history    — recent events (for spacing / forgetting curves)

Update rule (standard BKT):
  P(L_n | correct)   = P(L_{n-1})(1-slip) / [P(L_{n-1})(1-slip) + (1-P(L_{n-1}))guess]
  P(L_n | incorrect) = P(L_{n-1})slip / [P(L_{n-1})slip + (1-P(L_{n-1}))(1-guess)]
  P(L_{n+1}) = P(L_n) + (1-P(L_n)) * learn   (after each opportunity)

Question-type aware guess/slip priors:
  mcq (4 options)  guess=0.25  slip=0.10
  short/numerical  guess=0.02  slip=0.08
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

DEFAULT = {"p": 0.25, "attempts": 0, "correct": 0, "history": []}

PRIORS = {
    "mcq": {"guess": 0.25, "slip": 0.10},
    "short": {"guess": 0.02, "slip": 0.08},
    "numerical": {"guess": 0.02, "slip": 0.08},
}
LEARN = 0.12


class LearnerModel:
    def __init__(self, store: Path):
        self.store = store
        self.data: dict[str, dict] = {}
        if store.exists():
            self.data = json.loads(store.read_text(encoding="utf-8"))

    # ------------------------------------------------------------- core
    def _topic(self, topic: str) -> dict:
        return self.data.setdefault(topic, dict(DEFAULT, history=[]))

    def update(self, topic: str, correct: bool, qtype: str = "mcq") -> float:
        rec = self._topic(topic)
        priors = PRIORS.get(qtype, PRIORS["mcq"])
        p = rec["p"]
        g, s = priors["guess"], priors["slip"]
        if correct:
            num = p * (1 - s)
            den = num + (1 - p) * g
        else:
            num = p * s
            den = num + (1 - p) * (1 - g)
        p_post = num / den if den > 0 else p
        p_new = p_post + (1 - p_post) * LEARN
        rec["p"] = round(min(max(p_new, 0.01), 0.995), 4)
        rec["attempts"] += 1
        rec["correct"] += int(correct)
        rec["history"].append(
            {"t": int(time.time()), "correct": bool(correct), "type": qtype}
        )
        rec["history"] = rec["history"][-50:]
        self.save()
        return rec["p"]

    def mastery(self) -> dict[str, float]:
        return {t: r["p"] for t, r in self.data.items()}

    def weak_topics(self, threshold: float = 0.6) -> list[str]:
        return [t for t, r in self.data.items() if r["p"] < threshold]

    def strong_topics(self, threshold: float = 0.85) -> list[str]:
        return [t for t, r in self.data.items() if r["p"] >= threshold]

    def save(self) -> None:
        self.store.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    # --------------------------------------------------------- study plan
    def study_plan(self, all_topics: list[str], days_to_exam: int = 7) -> list[dict]:
        """Spaced schedule weighted toward weak topics + forgetting risk."""
        now = time.time()
        items = []
        for t in all_topics:
            rec = self.data.get(t)
            p = rec["p"] if rec else 0.25
            last = rec["history"][-1]["t"] if rec and rec["history"] else 0
            days_since = (now - last) / 86400 if last else 99
            # Ebbinghaus-style retention estimate
            retention = math.exp(-days_since / max(1.5, 6 * p))
            priority = (1 - p) * (1.2 - min(retention, 1.0))
            items.append({"topic": t, "mastery": p, "retention": round(retention, 3), "priority": round(priority, 3)})
        items.sort(key=lambda x: -x["priority"])
        per_day = max(1, math.ceil(len(items) / max(days_to_exam, 1)))
        plan = []
        for d in range(days_to_exam):
            chunk = items[d * per_day : (d + 1) * per_day]
            if not chunk:
                break
            plan.append({"day": d + 1, "topics": [c["topic"] for c in chunk]})
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
                }
                for t, r in self.data.items()
            },
            "weak_topics": weak,
            "strong_topics": strong,
        }
