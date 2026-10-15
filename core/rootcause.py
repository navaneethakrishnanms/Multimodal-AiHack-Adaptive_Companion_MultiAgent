"""RootCause — find *why* a student missed a question, not just that they missed it.

A wrong answer on topic T is often a symptom: the real gap sits in something T depends on.
Plain mastery tracking only says "weak on T". RootCause walks the prerequisite graph:

  1. blame(p) for each direct prerequisite p of T
         blame(p) ∝ (1 - mastery(p)) × (1 + bonus  if p was never probed)
     normalised to sum to 1 — a weak or never-tested prerequisite is the likeliest culprit.
  2. probe the argmax with a couple of verified questions (generation is injected, so this
     module has no LLM dependency).
  3. mastery(p) is lowered ONLY by the probe answers (ordinary BKT update, source="probe");
     the blame score itself never touches the learner model.
  4. if p is confirmed weak, build a repair path: the top source units for p (file +
     page/slide/timestamp), one level of recursion into p's own weak prerequisites, then
     re-test T.

All functions take a LearnerModel so the simulator (core/evaluate.py) and the web app share
exactly the same decision code.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable

from .learner import LearnerModel, WEAK

NEVER_PROBED_BONUS = 0.5
PROBE_QUESTIONS = 2


# ------------------------------------------------------------------ graph

def _words(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) > 2}


def _resolve(name: str, known: dict[str, str]) -> str | None:
    """Map a free-text prerequisite onto a real topic name (exact, then word overlap)."""
    key = name.strip().lower()
    if key in known:
        return known[key]
    w = _words(name)
    best, best_score = None, 0.0
    for k, canon in known.items():
        kw = _words(k)
        if not w or not kw:
            continue
        score = len(w & kw) / len(w | kw)
        if score > best_score:
            best, best_score = canon, score
    return best if best_score >= 0.5 else None


def build_graph(topics: list[dict]) -> dict[str, list[str]]:
    """topic name -> validated direct prerequisites.

    LLM-written prerequisite strings are matched to real topic names; unknown names, self
    references and cycle-closing edges are dropped so every consumer can assume a DAG.
    """
    names = [t["name"] for t in topics]
    known = {n.lower(): n for n in names}
    graph: dict[str, list[str]] = {n: [] for n in names}
    for t in topics:
        for raw in t.get("prerequisites", []) or []:
            p = _resolve(str(raw), known)
            if p and p != t["name"] and p not in graph[t["name"]]:
                graph[t["name"]].append(p)
    return _break_cycles(graph)


def _break_cycles(graph: dict[str, list[str]]) -> dict[str, list[str]]:
    state: dict[str, int] = {}  # 1 = on stack, 2 = done
    out = {k: list(v) for k, v in graph.items()}

    def visit(n: str) -> None:
        state[n] = 1
        for p in list(out[n]):
            if state.get(p) == 1:
                out[n].remove(p)  # back edge closes a cycle
            elif state.get(p) is None:
                visit(p)
        state[n] = 2

    for n in out:
        if n not in state:
            visit(n)
    return out


def ancestors(graph: dict[str, list[str]], topic: str) -> set[str]:
    seen: set[str] = set()
    stack = list(graph.get(topic, []))
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(graph.get(p, []))
    return seen


def topo_order(graph: dict[str, list[str]]) -> list[str]:
    """Foundations first (stable for equal depth)."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(n: str) -> None:
        if n in seen:
            return
        seen.add(n)
        for p in graph.get(n, []):
            visit(p)
        order.append(n)

    for n in graph:
        visit(n)
    return order


# ------------------------------------------------------------------ blame

def blame_scores(
    graph: dict[str, list[str]],
    topic: str,
    learner: LearnerModel,
    *,
    exclude: Iterable[str] = (),
    bonus: float = NEVER_PROBED_BONUS,
) -> list[dict]:
    """Normalised blame for each direct prerequisite of `topic`, highest first."""
    skip = set(exclude)
    rows = []
    for p in graph.get(topic, []):
        if p in skip:
            continue
        probed = learner.attempts(p) > 0
        m = learner.get(p)
        raw = (1.0 - m) * (1.0 + (0.0 if probed else bonus))
        rows.append({"topic": p, "mastery": round(m, 3), "probed": probed, "raw": raw})
    total = sum(r["raw"] for r in rows)
    for r in rows:
        r["blame"] = round(r["raw"] / total, 3) if total > 0 else round(1 / len(rows), 3)
        del r["raw"]
    rows.sort(key=lambda r: (-r["blame"], r["topic"]))
    return rows


def should_probe(rows: list[dict], strong: float = 0.85) -> bool:
    """Skip probing when every prerequisite is already well established by real answers."""
    return any(not (r["probed"] and r["mastery"] >= strong) for r in rows)


# ------------------------------------------------------------ repair path

def repair_path(
    graph: dict[str, list[str]],
    weak_topic: str,
    target: str,
    learner: LearnerModel,
    units_for_topic: Callable[[str, int], list[dict]],
    *,
    weak: float = WEAK,
    units_per_step: int = 2,
) -> dict:
    """Study order that fixes `weak_topic` (a confirmed-weak prerequisite of `target`).

    Recurses one level only: prerequisites of `weak_topic` that are *known* weak (answered
    and still below threshold) come first; never-tested ones are listed as unverified.
    """
    steps = []
    foundations = [
        q for q in graph.get(weak_topic, []) if learner.attempts(q) > 0 and learner.get(q) < weak
    ]
    unverified = [q for q in graph.get(weak_topic, []) if learner.attempts(q) == 0]
    for q in foundations:
        steps.append(_step(q, "foundation", learner, units_for_topic, units_per_step))
    steps.append(_step(weak_topic, "root cause", learner, units_for_topic, units_per_step))
    for i, s in enumerate(steps, 1):
        s["order"] = i
    return {
        "steps": steps,
        "unverified_foundations": unverified,
        "retest_topic": target,
    }


def _step(topic: str, role: str, learner: LearnerModel, units_for_topic, k: int) -> dict:
    return {
        "topic": topic,
        "role": role,
        "mastery": round(learner.get(topic), 3),
        "sources": units_for_topic(topic, k),
    }


# ---------------------------------------------------------------- facade

class RootCause:
    """Stateless helper around one student's learner model and the course graph."""

    def __init__(
        self,
        graph: dict[str, list[str]],
        learner: LearnerModel,
        units_for_topic: Callable[[str, int], list[dict]],
        make_probe: Callable[[str, int], list[dict]] | None = None,
        *,
        weak: float = WEAK,
        bonus: float = NEVER_PROBED_BONUS,
    ):
        self.graph = graph
        self.learner = learner
        self.units_for_topic = units_for_topic
        self.make_probe = make_probe
        self.weak = weak
        self.bonus = bonus

    def diagnose(self, topic: str, exclude: Iterable[str] = (), n_questions: int = PROBE_QUESTIONS) -> dict:
        """Blame the prerequisites of `topic` and prepare a probe for the argmax."""
        rows = blame_scores(self.graph, topic, self.learner, exclude=exclude, bonus=self.bonus)
        out: dict = {"topic": topic, "blame": rows, "target": None, "probe": [], "reason": ""}
        if topic not in self.graph:
            out["reason"] = "Topic is not in the course graph."
            return out
        if not rows:
            out["reason"] = (
                "This topic has no prerequisites left to blame — review the topic itself."
                if not self.graph.get(topic)
                else "Every prerequisite has already been probed — review the topic itself."
            )
            return out
        if not should_probe(rows):
            out["reason"] = "All prerequisites are already strong from real answers — review the topic itself."
            return out
        target = rows[0]["topic"]
        out["target"] = target
        out["reason"] = (
            f"“{target}” gets the highest blame ({rows[0]['blame']:.0%}): "
            + ("never tested" if not rows[0]["probed"] else f"mastery {rows[0]['mastery']:.0%}")
            + f", and “{topic}” depends on it."
        )
        if self.make_probe:
            out["probe"] = self.make_probe(target, n_questions)
        return out

    def conclude(self, topic: str, target: str) -> dict:
        """Call after the probe answers have updated the learner model (evidence only)."""
        p = self.learner.get(target)
        confirmed = self.learner.attempts(target) > 0 and p < self.weak
        out = {
            "topic": topic,
            "target": target,
            "target_mastery": round(p, 3),
            "confirmed_weak": confirmed,
        }
        if confirmed:
            out["repair"] = repair_path(
                self.graph, target, topic, self.learner, self.units_for_topic, weak=self.weak
            )
            out["message"] = (
                f"Your mistake on “{topic}” traces back to “{target}” "
                f"(probe mastery {p:.0%}). Review these sources first, then re-test."
            )
        else:
            nxt = blame_scores(self.graph, topic, self.learner, exclude=[target], bonus=self.bonus)
            out["next_candidate"] = nxt[0]["topic"] if nxt else None
            out["message"] = (
                f"“{target}” looks fine (probe mastery {p:.0%}), so the gap is more likely in "
                f"“{topic}” itself" + (f" or “{nxt[0]['topic']}”." if nxt else ".")
            )
        return out
