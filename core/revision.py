"""Revision material for weak topics: flashcards, short notes and a spoken-style brief.

Everything is built from the student's own uploaded material and keeps its source link.
LLM when available (better wording), deterministic extraction otherwise.
"""
from __future__ import annotations

import re

from . import llm, offline
from .ingest import KnowledgeBase, Unit

CARD_SYSTEM = """Write study flashcards from the excerpts. Return JSON
{"cards": [{"front": "question or cue", "back": "short answer", "source_unit_id": "id"}]}.
Only use facts in the excerpts; one idea per card; max 25 words on the back."""

BRIEF_SYSTEM = """Write a 100-130 word spoken revision brief on the topic using ONLY the excerpts.
Plain sentences, no markdown, no lists, easy to listen to. Mention the source naturally once
("as slide 4 says"). Do not invent facts."""


def _unit_dict(u: Unit) -> dict:
    return {"unit_id": u.id, "source_file": u.source_file, "location": u.location, "source_type": u.source_type}


def topic_units(kb: KnowledgeBase, topic: str, k: int = 4) -> list[Unit]:
    return [u for u in kb.units if topic in (u.topics or [])][:k]


def flashcards(kb: KnowledgeBase, topics: list[str], per_topic: int = 4) -> list[dict]:
    cards: list[dict] = []
    for topic in topics:
        units = topic_units(kb, topic, 4)
        if not units:
            continue
        made = []
        if llm.available():
            try:
                listing = "\n\n".join(f"[{u.id}] ({u.location}) {u.text[:500]}" for u in units)
                out = llm.chat_json(
                    [{"role": "system", "content": CARD_SYSTEM},
                     {"role": "user", "content": f"Topic: {topic}\nCards: {per_topic}\n\n{listing}"}],
                    max_tokens=900,
                )
                by_id = {u.id: u for u in units}
                for c in out.get("cards", []) if isinstance(out, dict) else []:
                    u = by_id.get(str(c.get("source_unit_id")), units[0])
                    made.append({"front": str(c["front"]), "back": str(c["back"]), "topic": topic,
                                 "source": _unit_dict(u), "mode": "llm"})
            except Exception:
                made = []
        if not made:
            for q in offline.cloze_questions(kb, units, per_topic, ["short"], seed=7, topic_names=[topic]):
                u = kb.units[[x.id for x in kb.units].index(q["source_unit_id"])]
                made.append({"front": q["question"].replace("Complete the statement from your material: ", ""),
                             "back": q["answer"], "topic": topic, "source": _unit_dict(u), "mode": "extractive"})
        cards.extend(made[:per_topic])
    return cards


def brief(kb: KnowledgeBase, topic: str) -> dict:
    units = topic_units(kb, topic, 3)
    if not units:
        return {"topic": topic, "text": "", "sources": [], "mode": "none"}
    text, mode = "", "extractive"
    if llm.available():
        try:
            listing = "\n\n".join(f"[{u.id}] ({u.source_file}, {u.location}) {u.text[:600]}" for u in units)
            text = llm.chat(
                [{"role": "system", "content": BRIEF_SYSTEM},
                 {"role": "user", "content": f"Topic: {topic}\n\n{listing}"}],
                max_tokens=260,
            ).strip()
            mode = "llm"
        except Exception:
            text = ""
    if not text:
        sents = [s for u in units for s in offline.sentences(u.text)][:4]
        text = f"Revision brief on {topic}. " + " ".join(sents)
    text = re.sub(r"\[u[0-9a-f]{8}\]", "", text)
    return {"topic": topic, "text": text, "sources": [_unit_dict(u) for u in units], "mode": mode}
