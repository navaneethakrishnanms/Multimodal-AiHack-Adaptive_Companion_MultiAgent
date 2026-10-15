"""Grounded tutor — answers strictly from retrieved course excerpts, with citations.

Design:
  - The tutor sees ONLY the top retrieved excerpts.
  - It must cite each claim with [unit_id] markers.
  - If the excerpts don't cover the question it must refuse and say so
    (the API layer also returns the refusal flag for the UI).
  - Outside knowledge is never blended silently: a clearly separated
    "Not in your materials" note is allowed only when the student asked
    something the material misses, and it is flagged.
"""
from __future__ import annotations

from .ingest import Unit
from .retrieve import Retriever
from . import llm

TUTOR_SYSTEM = """You are a study companion grounded in the student's own course material.

Rules (strict):
1. Answer ONLY using the provided excerpts. Never add outside facts.
2. Cite every factual claim with the excerpt id in square brackets, e.g. [u1a2b3c4].
3. If the excerpts do not contain the answer, reply exactly:
   "NOT_COVERED: I couldn't find this in your uploaded material."
   followed by one short sentence saying what IS covered nearby (if anything).
4. Keep answers concise and pedagogical: explain, then give one concrete example if the material has one.
5. If the student asks about an image/diagram that is referenced but its content is not in the text,
   say the visual content itself wasn't captured, and cite the location.
"""


def answer(question: str, retriever: Retriever, k: int = 6, rerank: bool = True) -> dict:
    hits = retriever.search(question, k=k)
    if rerank:
        hits = retriever.rerank(question, hits, top=4)
    else:
        hits = hits[:4]

    if not hits:
        return {
            "answer": "NOT_COVERED: I couldn't find this in your uploaded material.",
            "citations": [],
            "refused": True,
        }

    context = "\n\n".join(
        f"[{u.id}] (source: {u.source_file}, {u.location})\n{u.text}" for u, _s in hits
    )
    user = (
        f"Student question: {question}\n\n"
        f"Available excerpts:\n{context}\n\n"
        "Answer with citations, or refuse with NOT_COVERED if not covered."
    )
    text = llm.chat(
        [{"role": "system", "content": TUTOR_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.15,
        max_tokens=900,
    )

    refused = text.strip().startswith("NOT_COVERED")
    citations = []
    for u, score in hits:
        if u.id in text:
            citations.append(
                {
                    "unit_id": u.id,
                    "source_file": u.source_file,
                    "location": u.location,
                    "source_type": u.source_type,
                    "excerpt": u.text[:300],
                    "score": round(score, 3),
                }
            )
    return {"answer": text.strip(), "citations": citations, "refused": refused}
