"""Adaptive assessment — quiz/exam generation with answer verification.

Flow:
  1. scope = chosen topic(s) + difficulty + count + question types
  2. generation call produces questions tagged to {topic, source unit, difficulty}
  3. verification pass: a second LLM call independently solves each question
     and confirms the key (cross-model validation); mismatches are dropped
     or repaired, and structural checks catch broken options.
  4. novelty: previously asked question stems are stored per learner, so new
     assessments avoid repetition.
"""
from __future__ import annotations

import hashlib
import json
import re

from .ingest import KnowledgeBase, Unit
from .retrieve import Retriever
from . import llm

GEN_SYSTEM = """You write exam questions from study material excerpts.

Return JSON:
{
 "questions": [
   {
     "id": "q1",
     "type": "mcq" | "short" | "numerical",
     "difficulty": "easy" | "medium" | "hard",
     "topic": "topic name",
     "source_unit_id": "unit id used",
     "question": "...",
     "options": ["A ...", "B ...", "C ...", "D ..."],   // mcq only, exactly 4
     "answer": "the exact correct option text (mcq) OR model answer (short/numerical)",
     "explanation": "why, citing [unit_id]",
     "misconception": "the common wrong idea this question detects (optional)"
   }
 ]
}
Rules:
- Questions must be answerable purely from the provided excerpts.
- MCQ distractors must be plausible but clearly wrong for someone who studied.
- Numerical questions must include a definite numeric answer.
- Tag difficulty honestly. Tag topic using the provided topic list.
"""

VERIFY_SYSTEM = """You are an independent exam reviewer. You get a question and the source excerpt.
Solve the question yourself from the excerpt, then compare with the proposed answer.

Return JSON:
{"verdict": "ok" | "mismatch" | "unclear",
 "solved_answer": "...", "reason": "one sentence"}
"""


def _stem_hash(text: str) -> str:
    norm = re.sub(r"\W+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def _pick_units(kb: KnowledgeBase, topics: list[str], n: int = 10) -> list[Unit]:
    if topics:
        picked = [u for u in kb.units if u.topics and any(t in u.topics for t in topics)]
        if picked:
            return picked[:n]
    return kb.units[:n]


def generate(
    kb: KnowledgeBase,
    topics: list[str],
    *,
    count: int = 5,
    difficulty: str = "mixed",
    types: list[str] | None = None,
    avoid_hashes: set[str] | None = None,
) -> list[dict]:
    types = types or ["mcq", "short"]
    units = _pick_units(kb, topics)
    if not units:
        return []
    listing = "\n\n".join(f"[{u.id}] ({u.location}) {u.text[:450]}" for u in units[:8])
    topic_names = sorted({t for u in kb.units for t in (u.topics or [])}) or topics
    user = (
        f"Topics available: {topic_names}\n"
        f"Requested scope: {topics or 'any'}\n"
        f"Difficulty: {difficulty}\n"
        f"Types: {types}\n"
        f"Number of questions: {count}\n\n"
        f"Excerpts:\n{listing}"
    )
    out = llm.chat_json(
        [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=4000,
    )
    questions = out.get("questions", []) if isinstance(out, dict) else []

    # ---- structural + novelty checks
    unit_by_id = {u.id: u for u in kb.units}
    avoid = avoid_hashes or set()
    kept = []
    for q in questions:
        qtext = str(q.get("question", "")).strip()
        if not qtext or len(qtext) < 12:
            continue
        if _stem_hash(qtext) in avoid:
            continue
        if q.get("type") == "mcq":
            opts = q.get("options") or []
            ans = str(q.get("answer", "")).strip()
            if len(opts) != 4 or not any(ans and ans in o for o in opts):
                continue  # broken MCQ
        suid = str(q.get("source_unit_id", "")).strip()
        if suid not in unit_by_id:
            # try to fix: match by first citation in explanation
            m = re.search(r"\[(u[0-9a-f]{8})\]", q.get("explanation", "") + qtext)
            suid = m.group(1) if m and m.group(1) in unit_by_id else ""
        q["source_unit_id"] = suid
        q["_hash"] = _stem_hash(qtext)
        kept.append(q)

    # ---- independent verification (cross-validation)
    verified = []
    for q in kept:
        src = unit_by_id.get(q["source_unit_id"])
        if not src:
            continue
        try:
            v = llm.chat_json(
                [
                    {"role": "system", "content": VERIFY_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {q['question']}\n"
                            f"Proposed answer: {q.get('answer')}\n"
                            f"Options: {q.get('options')}\n\n"
                            f"Source excerpt [{src.id}] ({src.location}):\n{src.text[:800]}"
                        ),
                    },
                ],
                max_tokens=300,
            )
            verdict = v.get("verdict") if isinstance(v, dict) else None
            q["verified"] = verdict == "ok"
            if verdict == "mismatch":
                # repair the key with the independent answer when possible
                solved = str(v.get("solved_answer", "")).strip()
                if solved and q.get("type") != "mcq":
                    q["answer"] = solved
                    q["verified"] = True
                    q["repaired"] = True
                else:
                    continue
            elif verdict == "unclear":
                q["verified"] = False
            verified.append(q)
        except Exception:
            q["verified"] = False
            verified.append(q)
    return verified


def to_assessment(questions: list[dict]) -> dict:
    """Strip private fields for the client."""
    public = []
    for q in questions:
        c = {k: v for k, v in q.items() if not k.startswith("_")}
        public.append(c)
    return {"questions": public, "count": len(public)}
