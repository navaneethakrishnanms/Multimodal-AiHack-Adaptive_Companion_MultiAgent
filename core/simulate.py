"""Simulated-student evaluation of personalization policies (no LLM needed, fully seeded).

Student model ("latent gap"): a hidden set U of 20-30% of the course topics is *unknown*.
  can(c)   = c not in U  AND  no ancestor of c in U      (a topic needs all its prerequisites)
  answer   = correct with prob 1 - slip if can(c), else with prob guess (MCQ: 0.25)
  review   = reading the cited source for c teaches c with prob `review_success`, but only if
             every prerequisite of c is already known (you cannot learn Regularization before Overfitting)
The same latent student (same U) is run under every policy, so comparisons are paired.

Policies (each gets the same budget: sessions x questions_per_session answered questions):
  A fixed   : walk the topics in curriculum order, cycling; review a topic after missing it
  B bkt     : diagnostic pass over untested topics (curriculum order), then always the lowest-BKT-mastery
              topic (weakest-first); review the topic after a miss
  C rootcause: B's question choice, but after a miss run core.rootcause: blame the prerequisites,
               spend 2 probe questions (from the same budget) on the argmax, and if the probe confirms
               weakness follow the repair path (foundations first) instead of re-reading the missed topic

Reported gap set (what the dashboard would tell the student to fix):
  A, B : every answered topic with BKT mastery < gap_thr
  C    : the same, restricted to root causes (weak topics that have no weak ancestor) — this is the
         RootCause dashboard behaviour. Ablation "B_rootfilter" applies that same filter to B so the
         effect of the question strategy can be separated from the effect of the presentation.
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, asdict

from .learner import LearnerModel, WEAK
from .rootcause import ancestors, blame_scores, repair_path, should_probe, topo_order

POLICIES = ("A_fixed", "B_bkt", "C_rootcause")


@dataclass
class SimConfig:
    students: int = 30
    sessions: int = 6
    per_session: int = 4
    unknown_lo: float = 0.20
    unknown_hi: float = 0.30
    slip: float = 0.10
    guess: float = 0.25
    review_success: float = 0.7
    pool: int = 8  # distinct questions available per topic
    gap_thr: float = 0.5
    recall_target: float = 0.8
    sweep: bool = True  # B and C start with a diagnostic pass over untested topics (cold start, req. 4b)
    seed: int = 0


class SimStudent:
    def __init__(self, graph: dict[str, list[str]], cfg: SimConfig, idx: int):
        rng = random.Random(f"latent-{cfg.seed}-{idx}")
        names = sorted(graph)
        k = max(1, round(len(names) * rng.uniform(cfg.unknown_lo, cfg.unknown_hi)))
        self.initial_unknown = frozenset(rng.sample(names, k))
        self.unknown = set(self.initial_unknown)
        self.anc = {t: ancestors(graph, t) for t in graph}
        self.cfg = cfg

    def can(self, c: str) -> bool:
        return c not in self.unknown and not (self.anc[c] & self.unknown)

    def answer(self, c: str, rng: random.Random) -> bool:
        p = 1 - self.cfg.slip if self.can(c) else self.cfg.guess
        return rng.random() < p

    def review(self, c: str, rng: random.Random) -> None:
        if c in self.unknown and not (self.anc[c] & self.unknown) and rng.random() < self.cfg.review_success:
            self.unknown.discard(c)


class QuestionBank:
    """Per-student pool; always serves the least-used question, so repeats only appear when the pool runs dry."""

    def __init__(self, pool: int):
        self.pool = pool
        self.count: dict[tuple[str, int], int] = {}

    def next(self, topic: str) -> tuple[int, bool]:
        qi = min(range(self.pool), key=lambda i: (self.count.get((topic, i), 0), i))
        repeated = self.count.get((topic, qi), 0) > 0
        self.count[(topic, qi)] = self.count.get((topic, qi), 0) + 1
        return qi, repeated


def root_gaps(weak: set[str], anc: dict[str, set[str]]) -> set[str]:
    return {c for c in weak if not (anc[c] & weak)}


def _prf(rep: set[str], truth: set[str] | frozenset[str]) -> dict:
    hit = len(rep & truth)
    prec = hit / len(rep) if rep else 0.0
    rec = hit / len(truth) if truth else 1.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def run_student(policy: str, graph: dict[str, list[str]], order: list[str], idx: int, cfg: SimConfig) -> dict:
    stu = SimStudent(graph, cfg, idx)
    rng = random.Random(f"answers-{cfg.seed}-{idx}-{policy}")
    learner = LearnerModel(None)
    bank = QuestionBank(cfg.pool)
    budget = cfg.sessions * cfg.per_session
    state = {"asked": 0, "repeats": 0, "probes": 0, "correct": [], "ttf": {}}
    truth_all = stu.initial_unknown  # U: every hidden unknown topic
    truth_root = frozenset(root_gaps(set(truth_all), stu.anc))  # R: unknown topics with no unknown ancestor
    main_filtered = policy == "C_rootcause"

    def weak_set() -> set[str]:
        return {t for t, r in learner.data.items() if r["attempts"] and r["p"] < cfg.gap_thr}

    def reported(filtered: bool) -> set[str]:
        w = weak_set()
        return root_gaps(w, stu.anc) if filtered else w

    def ask(topic: str, probe: bool = False) -> bool:
        _qi, rep = bank.next(topic)
        ok = stu.answer(topic, rng)
        learner.update(topic, ok, "mcq", source="probe" if probe else "quiz")
        state["asked"] += 1
        state["repeats"] += int(rep)
        state["probes"] += int(probe)
        state["correct"].append(ok)
        for tag, filt, truth in (("U", main_filtered, truth_all), ("R", main_filtered, truth_root),
                                 ("U_filter", True, truth_all), ("R_filter", True, truth_root)):
            if tag not in state["ttf"] and truth and len(reported(filt) & truth) / len(truth) >= cfg.recall_target:
                state["ttf"][tag] = state["asked"]
        return ok

    def pick() -> str:
        if policy == "A_fixed":
            return order[state["asked"] % len(order)]
        if cfg.sweep:
            untested = [t for t in order if learner.attempts(t) == 0]
            if untested:
                return untested[0]
        return min(order, key=lambda t: (learner.get(t), learner.attempts(t), order.index(t)))

    while state["asked"] < budget:
        topic = pick()
        if ask(topic):
            continue
        handled = False
        if policy == "C_rootcause":
            rows = blame_scores(graph, topic, learner)
            if rows and should_probe(rows) and state["asked"] < budget:
                target = rows[0]["topic"]
                for _ in range(2):
                    if state["asked"] < budget:
                        ask(target, probe=True)
                if learner.attempts(target) > 0 and learner.get(target) < WEAK:
                    path = repair_path(graph, target, topic, learner, lambda t, k: [])
                    for step in path["steps"]:  # foundations first, then the blamed prerequisite
                        stu.review(step["topic"], rng)
                    handled = True
        if not handled:
            stu.review(topic, rng)

    out: dict = {"n_unknown": len(truth_all), "n_root_unknown": len(truth_root)}
    for suffix, filt in (("", main_filtered), ("_filter", True)):
        rep = reported(filt)
        for tag, truth in (("U", truth_all), ("R", truth_root)):
            for k, v in _prf(rep, truth).items():
                out[f"{k}_{tag}{suffix}"] = v
    for tag in ("U", "R", "U_filter", "R_filter"):
        out[f"ttf_{tag}"] = state["ttf"].get(tag, budget)
        out[f"ttf_{tag}_reached"] = tag in state["ttf"]
    half = len(state["correct"]) // 3
    out.update({
        "gain": len(truth_all - stu.unknown) / len(truth_all) if truth_all else 1.0,
        "repeat_rate": state["repeats"] / max(state["asked"], 1),
        "probe_share": state["probes"] / max(state["asked"], 1),
        "acc_first": sum(state["correct"][:half]) / max(half, 1),
        "acc_last": sum(state["correct"][-half:]) / max(half, 1),
    })
    return out


def _ms(xs: list[float]) -> dict:
    return {"mean": round(statistics.fmean(xs), 3), "std": round(statistics.pstdev(xs), 3)}


def _boot_ci(diffs: list[float], reps: int = 2000, seed: int = 1) -> list[float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(reps))
    return [round(means[int(0.025 * reps)], 3), round(means[int(0.975 * reps)], 3)]


METRICS = ["ttf_U", "precision_U", "recall_U", "f1_U", "ttf_R", "precision_R", "recall_R", "f1_R",
           "gain", "repeat_rate", "acc_first", "acc_last", "probe_share"]


def run_experiment(graph: dict[str, list[str]], cfg: SimConfig | None = None) -> dict:
    cfg = cfg or SimConfig()
    order = topo_order(graph)
    runs = {p: [run_student(p, graph, order, i, cfg) for i in range(cfg.students)] for p in POLICIES}
    summary: dict = {}
    for p, rs in runs.items():
        summary[p] = {m: _ms([r[m] for r in rs]) for m in METRICS}
        summary[p]["ttf_U_reached_pct"] = round(100 * sum(r["ttf_U_reached"] for r in rs) / len(rs), 1)
        summary[p]["ttf_R_reached_pct"] = round(100 * sum(r["ttf_R_reached"] for r in rs) / len(rs), 1)
    b = runs["B_bkt"]
    ab = {m: _ms([r[m.replace("ttf_U", "ttf_U_filter").replace("ttf_R", "ttf_R_filter")
                    if m.startswith("ttf") else m + "_filter"] for r in b])
          for m in METRICS if m.startswith(("ttf", "precision", "recall", "f1"))}
    ab["ttf_U_reached_pct"] = round(100 * sum(r["ttf_U_filter_reached"] for r in b) / len(b), 1)
    ab["ttf_R_reached_pct"] = round(100 * sum(r["ttf_R_filter_reached"] for r in b) / len(b), 1)
    summary["B_rootfilter(ablation)"] = ab
    paired = {}
    for m in ("ttf_U", "f1_U", "f1_R", "gain", "acc_last"):
        for base, tag in (("B_bkt", "B"), ("A_fixed", "A")):
            d = [c[m] - x[m] for c, x in zip(runs["C_rootcause"], runs[base])]
            paired[f"C_minus_{tag}.{m}"] = {"mean_diff": round(statistics.fmean(d), 3), "ci95": _boot_ci(d)}
    return {
        "config": asdict(cfg),
        "topics": len(graph),
        "mean_unknown_topics": round(statistics.fmean(r["n_unknown"] for r in runs["A_fixed"]), 2),
        "mean_root_unknown_topics": round(statistics.fmean(r["n_root_unknown"] for r in runs["A_fixed"]), 2),
        "summary": summary,
        "paired_vs_baselines": paired,
    }
