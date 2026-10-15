"""Grounded tutor — answers strictly from retrieved course excerpts, with citations.

Design:
  - The tutor sees ONLY the top retrieved excerpts.
  - It must cite each claim with [unit_id] markers.
  - Two refusal layers: (1) the retrieval gate (best passage below RETRIEVAL_MIN_SCORE) and
    (2) the model's own NOT_COVERED reply when the excerpts do not contain the answer.
  - Outside knowledge is never blended silently. Only if the student explicitly asks
    (`allow_outside`) does the tutor add a separate, clearly flagged general-knowledge answer.
  - No LLM configured/reachable: quote the best sentences (offline extractive mode).
The response's `grounding` field is what the UI uses for its green/amber banners:
  "source-backed" | "not-covered" | "outside-knowledge"
"""
from __future__ import annotations

from .ingest import Unit
from .retrieve import Retriever
from . import llm, offline

TUTOR_SYSTEM = """You are a study companion grounded in the student's own course material.

Rules (strict):
1. Answer ONLY using the provided excerpts. Never add outside facts.
2. Cite every factual claim with the excerpt id in square brackets, e.g. [u1a2b3c4].
3. If the excerpts do not contain the answer, reply exactly:
   "NOT_COVERED: I couldn't find this in your uploaded material."
   followed by one short sentence saying what IS covered nearby (if anything).
4. Keep answers concise and pedagogical: explain, then give one concrete example if the material has one.
5. Figure excerpts are descriptions of images; cite them like any other excerpt.
6. If the student asks about an image/diagram that is referenced but its content is not captured,
   say the visual content itself wasn't captured, and cite the location.{level_hint}"""

OUTSIDE_SYSTEM = """The student's course material does not cover this question. Give a short, accurate
general-knowledge answer (max 4 sentences). Do not cite course material. Start with:
"Outside your materials:" """

NOT_COVERED = "NOT_COVERED: I couldn't find this in your uploaded material."


def _citation(u: Unit, score: float) -> dict:
    return {
        "unit_id": u.id,
        "source_file": u.source_file,
        "location": u.location,
        "source_type": u.source_type,
        "excerpt": u.text[:300],
        "score": round(score, 3),
        "page": u.page,
        "topics": u.topics or [],
        "figure": getattr(u, "figure_path", None),
    }


LANGUAGES = {"hi": "Hindi (Devanagari script)", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "mr": "Marathi"}


def _level_hint(mastery: float | None, language: str = "en") -> str:
    """Personalise tone to the learner's current mastery and answer in their language."""
    hints = []
    if mastery is not None and mastery < 0.4:
        hints.append("The student is new to this topic: define terms first and go step by step.")
    elif mastery is not None and mastery >= 0.8:
        hints.append("The student knows this topic well: be brief and mention one nuance from the excerpts.")
    if language in LANGUAGES:
        hints.append(
            f"Write the explanation in {LANGUAGES[language]}. Keep technical terms, formulas and the [unit_id] "
            "citation markers exactly as in the excerpts (do not translate or alter them)."
        )
    return "".join(f"\n{7 + i}. {h}" for i, h in enumerate(hints))


def answer(
    question: str,
    retriever: Retriever,
    k: int = 6,
    rerank: bool = True,
    *,
    mastery_for: "callable | None" = None,
    allow_outside: bool = False,
    language: str = "en",
) -> dict:
    hits = retriever.search(question, k=k)
    covered = retriever.covered(hits, question)
    if not covered:
        return _refusal(question, [], allow_outside)

    hits = retriever.rerank(question, hits, top=4) if rerank else hits[:4]
    top_topics = hits[0][0].topics or []
    topic = top_topics[0] if top_topics else None
    mastery = mastery_for(topic) if (mastery_for and topic) else None

    context = "\n\n".join(
        f"[{u.id}] (source: {u.source_file}, {u.location})\n{u.text}" for u, _s in hits
    )
    user = (
        f"Student question: {question}\n\n"
        f"Available excerpts:\n{context}\n\n"
        "Answer with citations, or refuse with NOT_COVERED if not covered."
    )
    mode = "llm"
    try:
        text = llm.chat(
            [
                {"role": "system", "content": TUTOR_SYSTEM.format(level_hint=_level_hint(mastery, language))},
                {"role": "user", "content": user},
            ],
            temperature=0.15,
            max_tokens=900,
        ).strip()
    except Exception:
        text = offline.extractive_answer(question, hits)
        mode = "offline-extractive"

    if text.startswith("NOT_COVERED"):
        return _refusal(question, hits, allow_outside, model_text=text)

    citations = [_citation(u, s) for u, s in hits if u.id in text]
    if not citations:  # model forgot markers: attach what it was shown rather than show nothing
        citations = [_citation(u, s) for u, s in hits[:2]]
    return {
        "answer": text,
        "citations": citations,
        "refused": False,
        "grounding": "source-backed",
        "mode": mode,
        "topic": topic,
        "mastery": mastery,
    }


def _refusal(question: str, hits: list[tuple[Unit, float]], allow_outside: bool, model_text: str | None = None) -> dict:
    out = {
        "answer": model_text or NOT_COVERED,
        "citations": [],
        "refused": True,
        "grounding": "not-covered",
        "mode": "llm" if model_text else "gate",
        "nearby": [_citation(u, s) for u, s in hits[:2]],
    }
    if allow_outside and llm.available():
        try:
            general = llm.chat(
                [{"role": "system", "content": OUTSIDE_SYSTEM}, {"role": "user", "content": question}],
                temperature=0.2,
                max_tokens=300,
            ).strip()
            out["outside_answer"] = general
            out["grounding"] = "outside-knowledge"
        except Exception:
            pass
    return out
