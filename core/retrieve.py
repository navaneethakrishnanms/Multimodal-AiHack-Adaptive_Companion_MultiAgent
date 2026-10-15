"""Retrieval — TF-IDF passage search + topic tagging.

No external embedding service needed: scikit-learn TF-IDF gives a robust,
fully local retriever that works for any language of study material.
An optional LLM re-ranker improves ordering on the top candidates.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import KnowledgeBase, Unit
from . import llm


def _tokens(text: str) -> str:
    """Light normalization; keeps CJK intact, lowercases latin."""
    return re.sub(r"\s+", " ", text.lower())


class Retriever:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._vec = None
        self._matrix = None
        self._rebuild()

    def _rebuild(self) -> None:
        texts = [_tokens(u.text) for u in self.kb.units]
        if not texts:
            self._vec, self._matrix = None, None
            return
        self._vec = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1
        )
        self._matrix = self._vec.fit_transform(texts)

    def search(self, query: str, k: int = 6) -> list[tuple[Unit, float]]:
        if self._matrix is None:
            return []
        qv = self._vec.transform([_tokens(query)])
        sims = cosine_similarity(qv, self._matrix)[0]
        order = np.argsort(-sims)[:k]
        return [(self.kb.units[i], float(sims[i])) for i in order if sims[i] > 0.01]

    def rerank(self, query: str, hits: list[tuple[Unit, float]], top: int = 4) -> list[tuple[Unit, float]]:
        """Optional LLM re-rank of candidates (cheap, one call). Falls back on error."""
        if len(hits) <= 1:
            return hits[:top]
        try:
            listing = "\n".join(
                f"[{i}] {u.text[:400]}" for i, (u, _s) in enumerate(hits)
            )
            out = llm.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You rank study-material passages by relevance to a student question. "
                            'Return JSON {"order": [indices most relevant first, max 4]}. '
                            "Only include passages that genuinely help answer the question."
                        ),
                    },
                    {"role": "user", "content": f"Question: {query}\n\nPassages:\n{listing}"},
                ],
                max_tokens=120,
            )
            order = out.get("order", []) if isinstance(out, dict) else []
            picked = [hits[i] for i in order if 0 <= i < len(hits)]
            # keep any high-score hits the LLM dropped
            for h in hits:
                if h not in picked and h[1] > 0.35:
                    picked.append(h)
            return picked[:top]
        except Exception:
            return hits[:top]


# ---------------------------------------------------------------- topics

TOPIC_PROMPT = """You are building a course map from study material excerpts.

Given the excerpts, produce a topic structure as JSON:
{
 "topics": [
   {
     "name": "Topic name",
     "subtopics": ["..."],
     "key_concepts": ["..."],
     "prerequisites": ["other topic names this depends on"],
     "unit_ids": ["ids of excerpts belonging to this topic"]
   }
 ]
}
Rules:
- 3-8 topics. Use the material's own terminology.
- Every excerpt id must appear in exactly one topic.
- Keep names short (2-5 words).
"""


def build_topics(kb: KnowledgeBase, batch_size: int = 12) -> dict:
    """LLM topic extraction over all units; merges batches; tags units in place."""
    if not kb.units:
        return {"topics": []}
    topics: dict[str, dict] = {}

    for start in range(0, len(kb.units), batch_size):
        batch = kb.units[start : start + batch_size]
        listing = "\n\n".join(f"[{u.id}] ({u.location}) {u.text[:350]}" for u in batch)
        try:
            out = llm.chat_json(
                [
                    {"role": "system", "content": TOPIC_PROMPT},
                    {"role": "user", "content": listing},
                ],
                max_tokens=4000,
            )
        except Exception:
            continue
        for t in out.get("topics", []) if isinstance(out, dict) else []:
            name = str(t.get("name", "")).strip()
            if not name:
                continue
            rec = topics.setdefault(
                name,
                {"name": name, "subtopics": [], "key_concepts": [], "prerequisites": [], "unit_ids": []},
            )
            rec["subtopics"] = list(dict.fromkeys(rec["subtopics"] + [str(s) for s in t.get("subtopics", [])]))
            rec["key_concepts"] = list(dict.fromkeys(rec["key_concepts"] + [str(s) for s in t.get("key_concepts", [])]))
            rec["prerequisites"] = list(dict.fromkeys(rec["prerequisites"] + [str(s) for s in t.get("prerequisites", [])]))
            rec["unit_ids"] = list(dict.fromkeys(rec["unit_ids"] + [str(s) for s in t.get("unit_ids", [])]))

    # tag units
    unit_to_topics: dict[str, list[str]] = {}
    for t in topics.values():
        for uid in t["unit_ids"]:
            unit_to_topics.setdefault(uid, []).append(t["name"])
    for u in kb.units:
        u.topics = unit_to_topics.get(u.id, [])

    return {"topics": _merge_similar(list(topics.values()))}


def _merge_similar(topic_list: list[dict], threshold: float = 0.5) -> list[dict]:
    """Merge topics whose names are near-duplicates across batches (word overlap)."""
    merged: list[dict] = []
    for t in topic_list:
        words = {w.lower() for w in t["name"].split() if len(w) > 3}
        target = None
        for m in merged:
            mw = {w.lower() for w in m["name"].split() if len(w) > 3}
            if not words or not mw:
                continue
            overlap = len(words & mw) / min(len(words), len(mw))
            if overlap >= threshold:
                target = m
                break
        if target is None:
            merged.append(t)
        else:
            target["subtopics"] = list(dict.fromkeys(target["subtopics"] + t["subtopics"]))
            target["key_concepts"] = list(dict.fromkeys(target["key_concepts"] + t["key_concepts"]))
            target["prerequisites"] = list(dict.fromkeys(target["prerequisites"] + t["prerequisites"]))
            target["unit_ids"] = list(dict.fromkeys(target["unit_ids"] + t["unit_ids"]))
    return merged
