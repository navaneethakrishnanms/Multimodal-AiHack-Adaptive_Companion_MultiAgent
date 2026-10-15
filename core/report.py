"""Post-assessment report + misconception notes.

A wrong answer is turned into a short, topic-specific misconception note:
  1. the misconception the question generator attached to the question, else
  2. an LLM one-liner on why a student would pick that distractor, else
  3. a plain statement of what was chosen vs the key.
Notes are stored on the learner (per topic) and surfaced in the report and the recommendations.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from . import llm
from .learner import WEAK, LearnerModel
from .rootcause import blame_scores, should_probe


def misconception_note(key: dict, response: str) -> str:
    base = (key.get("misconception") or "").strip()
    if base:
        return base[:240]
    if llm.available() and key.get("type") == "mcq":
        try:
            text = llm.chat(
                [
                    {"role": "system", "content": "In ONE short sentence, name the likely misconception behind a wrong "
                     "multiple-choice answer. No preamble."},
                    {"role": "user", "content": f"Question: {key.get('question', '')}\nCorrect: {key.get('answer')}\n"
                     f"Student chose: {response}"},
                ],
                max_tokens=70,
            ).strip()
            if text:
                return text[:240]
        except Exception:
            pass
    return f"Chose “{response[:80]}” where the material supports “{str(key.get('answer'))[:80]}”."


def assessment_report(
    learner: LearnerModel,
    results: list[dict],
    graph: dict[str, list[str]],
    sources,
    quiz_id: str | None = None,
) -> dict:
    """Summarise one assessment (default: the most recent quiz) with weak topics and likely misconceptions."""
    if not results:
        return {"answered": 0, "message": "No answers yet."}
    if quiz_id is None:
        quiz_id = next((r["quiz_id"] for r in reversed(results) if r.get("source") == "quiz" and r.get("quiz_id")), None)
    rows = [r for r in results if r.get("quiz_id") == quiz_id] if quiz_id else results
    by_topic: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r)
    topics = []
    for t, rs in by_topic.items():
        acc = sum(r["correct"] for r in rs) / len(rs)
        m = learner.get(t)
        weak = acc < 0.6 or m < WEAK
        item = {
            "topic": t, "answered": len(rs), "correct": sum(r["correct"] for r in rs),
            "accuracy": round(acc, 3), "mastery": round(m, 3), "weak": weak,
        }
        if weak:
            item["sources"] = sources(t, 2)
            blame = blame_scores(graph, t, learner)
            if blame and should_probe(blame):
                item["suspect_prerequisite"] = {"topic": blame[0]["topic"], "blame": blame[0]["blame"]}
        topics.append(item)
    topics.sort(key=lambda x: (not x["weak"], x["accuracy"]))
    notes = Counter(r["misconception"] for r in rows if r.get("misconception") and not r["correct"])
    correct = sum(r["correct"] for r in rows)
    return {
        "quiz_id": quiz_id,
        "answered": len(rows),
        "correct": correct,
        "score": round(correct / len(rows), 3),
        "topics": topics,
        "weak_topics": [t["topic"] for t in topics if t["weak"]],
        "misconceptions": [{"note": n, "count": c} for n, c in notes.most_common(5)],
    }
