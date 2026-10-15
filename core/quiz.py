"""Adaptive assessment — quiz/exam generation with answer verification.

Flow:
  1. scope = chosen topic(s) + difficulty + count + question types
  2. generation call produces questions tagged to {topic, source unit, difficulty}
  3. verification pass: an independent solve from the source excerpt confirms the key.
     Set VERIFY_MODEL to run it on a different model (cross-model validation); mismatching
     keys are dropped (MCQ) or repaired from the independent answer (short/numerical).
  4. novelty: stems asked before are remembered per learner; exact and near-duplicate
     (token-Jaccard) stems are rejected, and source passages rotate so scopes are not
     re-sampled from the same few units.
  No LLM available -> offline cloze questions whose key is a literal span of the source
  (verification = "extractive").
"""
from __future__ import annotations

import hashlib
import os
import random
import re

from .ingest import KnowledgeBase, Unit
from . import llm, offline

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
- Questions must be answerable purely from the provided excerpts. Figure excerpts are image descriptions;
  you may ask about what a figure shows.
- MCQ distractors must be plausible but clearly wrong for someone who studied.
- Numerical questions must include a definite numeric answer.
- Tag difficulty honestly. Tag topic using ONLY the provided topic list.
- Do not repeat any question in the "Already asked" list or ask the same thing in different words.
"""

VERIFY_SYSTEM = """You are an independent exam reviewer. You get a question and the source excerpt.
Solve the question yourself from the excerpt, then compare with the proposed answer.

Return JSON:
{"verdict": "ok" | "mismatch" | "unclear",
 "solved_answer": "...", "reason": "one sentence"}
"""

GRADE_SYSTEM = (
    "Grade the student answer against the model answer. Accept equivalent wording and equivalent numeric "
    'forms. Return JSON {"correct": true|false, "feedback": "one short sentence"}'
)


def stem_norm(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _stem_hash(text: str) -> str:
    return hashlib.sha1(stem_norm(text).encode()).hexdigest()[:16]


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def is_repeat(stem: str, avoid_hashes: set[str], avoid_stems: list[str], near: float = 0.8) -> bool:
    if _stem_hash(stem) in avoid_hashes:
        return True
    n = stem_norm(stem)
    return any(_jaccard(n, s) >= near for s in avoid_stems)


def canonical_topic(name: str, known: list[str]) -> str:
    """Snap an LLM-written topic tag onto a real topic name so learner mastery is not fragmented."""
    if not known or not name:
        return name or "general"
    low = {k.lower(): k for k in known}
    if name.lower() in low:
        return low[name.lower()]
    w = set(re.findall(r"[a-z0-9]+", name.lower())) - offline.STOP
    best, best_s = None, 0.0
    for k in known:
        kw = set(re.findall(r"[a-z0-9]+", k.lower())) - offline.STOP
        if w and kw:
            s = len(w & kw) / len(w | kw)
            if s > best_s:
                best, best_s = k, s
    return best if best and best_s >= 0.5 else name


def pick_units(
    kb: KnowledgeBase, topics: list[str], n: int = 8, rng: random.Random | None = None,
    used: dict[str, int] | None = None,
) -> list[Unit]:
    """Units for the scope, least-used first so repeated quizzes rotate through the material."""
    rng = rng or random.Random()
    used = used or {}
    pool = [u for u in kb.units if u.topics and any(t in u.topics for t in topics)] if topics else []
    pool = pool or list(kb.units)
    rng.shuffle(pool)
    pool.sort(key=lambda u: used.get(u.id, 0))
    return pool[:n]


def generate(
    kb: KnowledgeBase,
    topics: list[str],
    *,
    count: int = 5,
    difficulty: str = "mixed",
    types: list[str] | None = None,
    avoid_hashes: set[str] | None = None,
    avoid_stems: list[str] | None = None,
    used_units: dict[str, int] | None = None,
    known_topics: list[str] | None = None,
    topic_terms: dict[str, set[str]] | None = None,
    seed: int | None = None,
    verify: bool = True,
) -> list[dict]:
    types = types or ["mcq", "short"]
    rng = random.Random(seed)
    units = pick_units(kb, topics, 8, rng, used_units)
    if not units:
        return []
    known = known_topics or sorted({t for u in kb.units for t in (u.topics or [])})
    avoid_h, avoid_s = set(avoid_hashes or ()), list(avoid_stems or [])

    questions: list[dict] = []
    if llm.available():
        try:
            questions = _llm_questions(units, topics, known, count, difficulty, types, avoid_s)
        except Exception:
            questions = []
    if not questions:
        focus = set().union(*(topic_terms.get(x, set()) for x in topics)) if topic_terms and topics else None
        questions = offline.cloze_questions(
            kb, units, count + 2, types, seed=rng.randint(0, 10**6), topic_names=topics or known, focus_terms=focus or None
        )

    unit_by_id = {u.id: u for u in kb.units}
    kept: list[dict] = []
    for q in questions:
        qtext = str(q.get("question", "")).strip()
        if len(qtext) < 12 or is_repeat(qtext, avoid_h, avoid_s):
            continue
        if q.get("type") == "mcq":
            opts = q.get("options") or []
            ans = str(q.get("answer", "")).strip()
            if len(opts) != 4 or not any(ans and ans in o for o in opts):
                continue  # broken MCQ
        suid = str(q.get("source_unit_id", "")).strip()
        if suid not in unit_by_id:
            m = re.search(r"\[(u[0-9a-f]{8})\]", str(q.get("explanation", "")) + qtext)
            suid = m.group(1) if m and m.group(1) in unit_by_id else ""
        if not suid:
            continue
        q["source_unit_id"] = suid
        q["topic"] = canonical_topic(str(q.get("topic", "")), known)
        if topics and q["topic"] not in topics:
            # the model drifted out of scope: use a scoped tag from the source passage instead
            scoped = [t for t in (unit_by_id[suid].topics or []) if t in topics]
            q["topic"] = scoped[0] if scoped else q["topic"]
        q["_hash"] = _stem_hash(qtext)
        q["_stem"] = stem_norm(qtext)
        avoid_h.add(q["_hash"])
        avoid_s.append(q["_stem"])  # also dedupes within this batch
        kept.append(q)

    if verify:
        kept = [q for q in (_verify(q, unit_by_id[q["source_unit_id"]]) for q in kept) if q]
    if difficulty != "mixed":
        matching = [q for q in kept if q.get("difficulty") == difficulty]
        kept = matching or kept
    return kept[:count]


def _llm_questions(units, topics, known, count, difficulty, types, avoid_stems) -> list[dict]:
    listing = "\n\n".join(f"[{u.id}] ({u.location}) {u.text[:450]}" for u in units[:8])
    asked = "\n".join(f"- {s[:120]}" for s in avoid_stems[-25:]) or "(none)"
    user = (
        f"Topics available: {known}\n"
        f"Requested scope: {topics or 'any'}\n"
        f"Difficulty: {difficulty}\n"
        f"Types: {types}\n"
        f"Number of questions: {count + 1}\n\n"
        f"Already asked:\n{asked}\n\n"
        f"Excerpts:\n{listing}"
    )
    out = llm.chat_json(
        [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": user}],
        max_tokens=4000,
    )
    return out.get("questions", []) if isinstance(out, dict) else []


def _verify(q: dict, src: Unit) -> dict | None:
    """Independent re-solve. Returns the question (annotated) or None when the key is wrong."""
    if q.get("verification") == "extractive":
        return q
    if not llm.available():
        q["verified"], q["verification"] = False, "unverified"
        return q
    vmodel = os.environ.get("VERIFY_MODEL") or None
    kind = "cross-model" if vmodel and vmodel != os.environ.get("LLM_MODEL") else "independent-pass"
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
            model=vmodel,
        )
    except Exception:
        q["verified"], q["verification"] = False, "unverified"
        return q
    verdict = v.get("verdict") if isinstance(v, dict) else None
    q["verification"] = kind
    if verdict == "ok":
        q["verified"] = True
    elif verdict == "mismatch":
        solved = str(v.get("solved_answer", "")).strip()
        if solved and q.get("type") != "mcq":
            q["answer"], q["verified"], q["repaired"] = solved, True, True
        else:
            return None
    else:
        q["verified"] = False
    return q


# ----------------------------------------------------------------- grading

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^[A-D][.):]\s*", "", s.strip(), flags=re.I)).strip().lower()


def grade(key: dict, response: str) -> dict:
    """Grade one response against a stored key. Returns {correct, feedback, mode}."""
    qtype = key.get("type", "mcq")
    answer = str(key.get("answer", ""))
    response = response.strip()
    if qtype == "mcq":
        letter = re.fullmatch(r"([A-D])[.)]?", response, flags=re.I)
        if letter:
            ok = answer.strip().upper().startswith(letter.group(1).upper())
        else:
            ok = _norm(response) == _norm(answer)
        return {"correct": ok, "feedback": "", "mode": "exact"}
    if llm.available():
        try:
            v = llm.chat_json(
                [
                    {"role": "system", "content": GRADE_SYSTEM},
                    {"role": "user", "content": f"Question type: {qtype}\nQuestion: {key.get('question', '')}\n"
                     f"Model answer: {answer}\nStudent answer: {response}"},
                ],
                max_tokens=200,
            )
            return {"correct": bool(v.get("correct")), "feedback": str(v.get("feedback", "")), "mode": "llm"}
        except Exception:
            pass
    return {"correct": offline.grade_short(response, answer), "feedback": "", "mode": "offline"}


def to_assessment(questions: list[dict]) -> dict:
    """Strip private fields for the client."""
    public = []
    for q in questions:
        c = {k: v for k, v in q.items() if not k.startswith("_")}
        public.append(c)
    return {"questions": public, "count": len(public)}
