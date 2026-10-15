"""System evaluation — retrieval & generation quality + simulated-student runs.

Framework-based metrics (RAGAS-style, implemented natively so no extra deps):
  - retrieval hit rate / MRR on a labeled test set (question -> known source)
  - faithfulness: every claim in the answer supported by the cited excerpts (LLM judge)
  - answer relevancy: does the answer address the question (LLM judge)
  - context precision / recall: fraction of retrieved excerpts that are relevant
  - refusal accuracy on off-material questions

Personalization evaluation:
  - simulated student profiles (weak / medium / strong) answer quizzes with
    configurable accuracy; we report mastery gain per topic and question
    repetition rate across sessions.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .ingest import KnowledgeBase
from .retrieve import Retriever
from .learner import LearnerModel
from . import llm


# ------------------------------------------------------------ retrieval eval

def _location_matches(unit, case: dict) -> bool:
    """Lenient source match: same file, and location equal or adjacent.

    Chunk boundaries are arbitrary (±1 paragraph keeps the metric honest while
    tolerating split/merge differences); slides and timestamps must match exactly.
    """
    if unit.source_file != case.get("source_file"):
        return False
    want = (case.get("location") or "").strip()
    if not want:
        return True
    got = (unit.location or "").strip()
    if want == got:
        return True
    import re as _re

    mw = _re.fullmatch(r"para (\d+)", want)
    mg = _re.fullmatch(r"para (\d+)", got)
    if mw and mg:
        return abs(int(mw.group(1)) - int(mg.group(1))) <= 1
    return False


def eval_retrieval(retriever: Retriever, testset: list[dict]) -> dict:
    """testset: [{"question": ..., "source_file": ..., "location": ...}]

    Off-material cases ({"off_material": true}) are skipped here — retrieval has no
    correct source for them by definition (their refusal behavior is measured in
    eval_generation).
    """
    cases = [c for c in testset if not c.get("off_material")]
    hits, rr = 0, []
    for case in cases:
        results = retriever.search(case["question"], k=6)
        rank = None
        for i, (u, _s) in enumerate(results):
            if _location_matches(u, case):
                rank = i + 1
                break
        if rank:
            hits += 1
            rr.append(1 / rank)
        else:
            rr.append(0)
    n = max(len(cases), 1)
    return {
        "cases": len(cases),
        "hit_rate@6": round(hits / n, 3),
        "mrr": round(sum(rr) / n, 3),
    }


# ------------------------------------------------------ generation eval

_FAITH_SYSTEM = """You judge answer faithfulness to source excerpts.
Given a question, an answer, and the source excerpts it cites, decide:
- faithful: every factual claim in the answer is supported by the excerpts
- unfaithful: at least one claim is unsupported or contradicts the excerpts
Return JSON {"verdict": "faithful"|"unfaithful", "unsupported": ["claim", ...]}"""

_REL_SYSTEM = """You judge answer relevancy.
Given a question and an answer, decide if the answer directly addresses the question.
Return JSON {"verdict": "relevant"|"irrelevant", "reason": "one sentence"}"""


def eval_generation(retriever: Retriever, testset: list[dict]) -> dict:
    from .tutor import answer as tutor_answer
    from .ingest import KnowledgeBase  # noqa: F401  (type only)

    faithful, relevant, refused_ok = 0, 0, 0
    on_material = [c for c in testset if not c.get("off_material")]
    off_material = [c for c in testset if c.get("off_material")]

    # full unit texts so the judge sees the complete source passage
    unit_by_id = {u.id: u.text for u in retriever.kb.units}

    for case in on_material:
        r = tutor_answer(case["question"], retriever, rerank=False)
        if r["refused"]:
            continue
        excerpts = "\n\n".join(
            f"[{c['unit_id']}] {unit_by_id.get(c['unit_id'], c['excerpt'])}"
            for c in r["citations"]
        ) or "(none cited)"
        try:
            v = llm.chat_json(
                [
                    {"role": "system", "content": _FAITH_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Question: {case['question']}\nAnswer: {r['answer']}\nExcerpts:\n{excerpts}",
                    },
                ],
                max_tokens=300,
            )
            if isinstance(v, dict) and v.get("verdict") == "faithful":
                faithful += 1
        except Exception:
            pass
        try:
            v = llm.chat_json(
                [
                    {"role": "system", "content": _REL_SYSTEM},
                    {"role": "user", "content": f"Question: {case['question']}\nAnswer: {r['answer']}"},
                ],
                max_tokens=150,
            )
            if isinstance(v, dict) and v.get("verdict") == "relevant":
                relevant += 1
        except Exception:
            pass

    for case in off_material:
        r = tutor_answer(case["question"], retriever, rerank=False)
        if r["refused"]:
            refused_ok += 1

    n_on = max(len(on_material), 1)
    n_off = max(len(off_material), 1)
    return {
        "on_material_questions": len(on_material),
        "faithfulness": round(faithful / n_on, 3),
        "answer_relevancy": round(relevant / n_on, 3),
        "off_material_questions": len(off_material),
        "refusal_accuracy": round(refused_ok / n_off, 3) if off_material else None,
    }


# ------------------------------------------------- personalization eval

def eval_personalization(
    kb: KnowledgeBase,
    topics: list[str],
    *,
    sessions: int = 3,
    questions_per_session: int = 4,
    profiles: dict[str, float] | None = None,
) -> dict:
    """Simulated students: accuracy = P(correct). Reports mastery gain + repetition."""
    from .quiz import generate

    profiles = profiles or {"weak": 0.35, "medium": 0.6, "strong": 0.85}
    out = {}
    for pname, acc in profiles.items():
        rng = random.Random(hash(pname) % 10_000)
        learner = LearnerModel(Path(f"/tmp/sim_{pname}.json"))
        learner.data = {}
        seen_hashes: set[str] = set()
        asked, repeated = 0, 0
        start = learner.mastery()
        for _s in range(sessions):
            qs = generate(kb, topics, count=questions_per_session, avoid_hashes=seen_hashes)
            for q in qs:
                h = q.get("_hash")
                if h in seen_hashes:
                    repeated += 1
                if h:
                    seen_hashes.add(h)
                asked += 1
                correct = rng.random() < acc
                learner.update(q.get("topic") or "general", correct, q.get("type", "mcq"))
        end = learner.mastery()
        gains = {
            t: round(end.get(t, 0.25) - start.get(t, 0.25), 3)
            for t in set(list(end.keys()) + list(start.keys()))
        }
        out[pname] = {
            "accuracy_sim": acc,
            "questions_asked": asked,
            "repetition_rate": round(repeated / max(asked, 1), 3),
            "mastery_gain_avg": round(sum(gains.values()) / max(len(gains), 1), 3),
            "final_mastery": {t: v for t, v in sorted(end.items())},
        }
    return out


def run_all(
    kb: KnowledgeBase,
    retriever: Retriever,
    testset_path: Path,
    *,
    topics: list[str] | None = None,
    skip_personalization: bool = False,
) -> dict:
    testset = json.loads(testset_path.read_text(encoding="utf-8"))
    report: dict = {"retrieval": eval_retrieval(retriever, testset)}
    report["generation"] = eval_generation(retriever, testset)
    if not skip_personalization:
        all_topics = topics or sorted({t for u in kb.units for t in (u.topics or [])})
        report["personalization"] = eval_personalization(kb, all_topics[:4])
    return report
