"""Retrieval — TF-IDF passage search + topic mapping.

No external embedding service needed: scikit-learn TF-IDF gives a robust,
fully local retriever. An optional LLM re-ranker improves ordering on the top candidates.

Refusal gate: a query whose best passage scores below `min_score` is treated as not covered
by the material (the value is calibrated on a dev split in core/evaluate.py).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import KnowledgeBase, Unit
from . import llm, offline

# Calibrated in Phase 3 (see eval_report.json -> retrieval_gate); override with RETRIEVAL_MIN_SCORE.
DEFAULT_MIN_SCORE = 0.10


def _tokens(text: str) -> str:
    """Light normalization; keeps CJK intact, lowercases latin."""
    return re.sub(r"\s+", " ", text.lower())


GATE_FILE = Path(__file__).with_name("gate.json")


def _gate_value(key: str, env: str, default: float) -> float:
    """env override > core/gate.json (written by scripts/run_eval.py --write-gate) > default."""
    if os.environ.get(env):
        try:
            return float(os.environ[env])
        except ValueError:
            pass
    try:
        return float(json.loads(GATE_FILE.read_text(encoding="utf-8"))[key])
    except Exception:
        return default


def min_score() -> float:
    """Strict gate when no LLM is available; recall-first gate when the LLM can refuse on its own."""
    if llm.available():
        return _gate_value("min_score_online", "RETRIEVAL_MIN_SCORE_ONLINE", DEFAULT_MIN_SCORE)
    return _gate_value("min_score", "RETRIEVAL_MIN_SCORE", DEFAULT_MIN_SCORE)


def min_coverage() -> float:
    return _gate_value("min_coverage", "RETRIEVAL_MIN_COVERAGE", 0.0)


def stem(word: str) -> str:
    """Very light suffix stripping so 'overfit' matches 'overfitting' and 'sets' matches 'set'."""
    for suf in ("ing", "ation", "ions", "ion", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


class Retriever:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self._vec = None
        self._matrix = None
        self._vocab = None
        self._rebuild()

    def _rebuild(self) -> None:
        self._vocab = None
        texts = [_tokens(u.text) for u in self.kb.units]
        if not texts:
            self._vec, self._matrix = None, None
            return
        self._vec = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=1, stop_words="english"
        )
        self._matrix = self._vec.fit_transform(texts)

    def scores(self, query: str) -> np.ndarray:
        if self._matrix is None:
            return np.zeros(0)
        qv = self._vec.transform([_tokens(query)])
        return cosine_similarity(qv, self._matrix)[0]

    def search(self, query: str, k: int = 6) -> list[tuple[Unit, float]]:
        """Top-k units by similarity (no gate; callers decide with `covered`)."""
        sims = self.scores(query)
        if not len(sims):
            return []
        order = np.argsort(-sims)[:k]
        return [(self.kb.units[i], float(sims[i])) for i in order if sims[i] > 0.01]

    def vocab(self) -> set[str]:
        if getattr(self, "_vocab", None) is None:
            self._vocab = {stem(w) for u in self.kb.units for w in offline.content_words(u.text)}
        return self._vocab

    def coverage(self, query: str) -> float:
        """Share of the query's content words (stemmed) that occur anywhere in the course material."""
        words = {stem(w) for w in offline.content_words(query)}
        return len(words & self.vocab()) / len(words) if words else 0.0

    def covered(self, hits: list[tuple[Unit, float]], query: str | None = None,
                gate: float | None = None, cov_gate: float | None = None) -> bool:
        """Refusal gate: the best passage must be similar enough AND the query's own terms must exist in the course."""
        if not hits or hits[0][1] < (min_score() if gate is None else gate):
            return False
        if query is not None and self.coverage(query) < (min_coverage() if cov_gate is None else cov_gate):
            return False
        return True

    def rerank(self, query: str, hits: list[tuple[Unit, float]], top: int = 4) -> list[tuple[Unit, float]]:
        """Optional LLM re-rank of candidates (cheap, one call). Falls back on error."""
        if len(hits) <= 1 or not llm.available():
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
            picked = [hits[i] for i in order if isinstance(i, int) and 0 <= i < len(hits)]
            # keep any high-score hits the LLM dropped
            for h in hits:
                if h not in picked and h[1] > 0.35:
                    picked.append(h)
            return (picked or hits)[:top]
        except Exception:
            return hits[:top]


# ---------------------------------------------------------- topic helpers

def topic_query(topic: dict | str) -> str:
    if isinstance(topic, str):
        return topic
    return " ".join([topic["name"]] + topic.get("subtopics", []) + topic.get("key_concepts", []))


def units_for_topic(
    kb: KnowledgeBase, retriever: Retriever, topics: list[dict], name: str, k: int = 2
) -> list[dict]:
    """Best k source units for a topic: tagged units ranked by similarity to the topic's concepts."""
    rec = next((t for t in topics if t["name"] == name), None)
    query = topic_query(rec) if rec else name
    sims = retriever.scores(query)
    tagged = [i for i, u in enumerate(kb.units) if name in (u.topics or [])]
    pool = tagged or list(np.argsort(-sims)[:k]) if len(sims) else []
    ranked = sorted(pool, key=lambda i: -sims[i])[:k]
    out = []
    for i in ranked:
        u = kb.units[i]
        out.append(
            {
                "unit_id": u.id,
                "source_file": u.source_file,
                "location": u.location,
                "source_type": u.source_type,
                "excerpt": u.text[:300],
                "score": round(float(sims[i]), 3),
            }
        )
    return out


def assign_units(kb: KnowledgeBase, topics: list[dict], rel: float = 0.5, floor: float = 0.04) -> None:
    """Tag units to topics by TF-IDF similarity to each topic's concept text (in place).

    Each unit goes to every topic scoring at least `rel` x its best score (and above `floor`),
    so a passage that spans two concepts is tagged with both.
    """
    if not kb.units or not topics:
        return
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2))
    docs = [u.text for u in kb.units] + [topic_query(t) for t in topics]
    M = vec.fit_transform(docs)
    U, T = M[: len(kb.units)], M[len(kb.units) :]
    sims = cosine_similarity(U, T)
    for t in topics:
        t["unit_ids"] = []
    for i, u in enumerate(kb.units):
        best = sims[i].max()
        tags = []
        for j, t in enumerate(topics):
            if best > floor and sims[i, j] >= rel * best:
                t["unit_ids"].append(u.id)
                tags.append(t["name"])
        u.topics = tags


def load_seed_topics(kb: KnowledgeBase, path: Path) -> dict | None:
    """Curated course map shipped with the sample course; applies only if its files are loaded."""
    if not path.exists():
        return None
    seed = json.loads(path.read_text(encoding="utf-8"))
    files = {u.source_file for u in kb.units}
    if not set(seed.get("source_files", [])) & files:
        return None
    topics = [dict(t, unit_ids=[]) for t in seed["topics"]]
    assign_units(kb, topics)
    return {"topics": topics, "mode": "curated"}


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
- 4-10 fine-grained topics (one teachable idea each, e.g. "Learning Rate", not "Optimization").
- Use the material's own terminology; keep names short (2-5 words).
- Every excerpt id must appear in at least one topic.
- prerequisites: only topics a student must understand BEFORE this one.
"""

CONSOLIDATE_PROMPT = """You are merging topic lists extracted from different parts of one course.

Input: JSON list of topics (name, key_concepts, unit_ids) from several batches.
Return JSON:
{"topics": [{"name": "...", "merged_from": ["input names merged into this one"],
             "subtopics": ["..."], "key_concepts": ["..."],
             "prerequisites": ["names from YOUR output list only"]}]}
Rules:
- Merge duplicates and near-duplicates (same idea, different wording or case).
- Keep distinct teachable ideas separate. Aim for 6-14 topics in total.
- Prerequisites must form a DAG over your final names and reflect real learning order.
- Every input name must appear in exactly one merged_from.
"""


def build_topics(kb: KnowledgeBase, batch_size: int = 12) -> dict:
    """Topic map over all units; tags units in place. LLM when available, else heuristic."""
    if not kb.units:
        return {"topics": []}
    if not llm.available():
        out = offline.heuristic_topics(kb)
        _tag_units(kb, out["topics"])
        return out
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
            for field in ("subtopics", "key_concepts", "prerequisites", "unit_ids"):
                rec[field] = list(dict.fromkeys(rec[field] + [str(s) for s in t.get(field, [])]))

    if not topics:  # every LLM call failed: degrade instead of returning nothing
        out = offline.heuristic_topics(kb)
        _tag_units(kb, out["topics"])
        return out

    merged = _consolidate(list(topics.values()))
    _tag_units(kb, merged)
    return {"topics": merged, "mode": "llm"}


def _consolidate(topic_list: list[dict]) -> list[dict]:
    """Second pass: merge near-duplicates across batches and wire prerequisites among final names."""
    if len(topic_list) < 2:
        return topic_list
    try:
        slim = [
            {"name": t["name"], "key_concepts": t["key_concepts"][:6], "prerequisites": t["prerequisites"]}
            for t in topic_list
        ]
        out = llm.chat_json(
            [
                {"role": "system", "content": CONSOLIDATE_PROMPT},
                {"role": "user", "content": json.dumps(slim, ensure_ascii=False)},
            ],
            max_tokens=4000,
        )
        by_name = {t["name"]: t for t in topic_list}
        final, claimed = [], set()
        for m in out.get("topics", []) if isinstance(out, dict) else []:
            name = str(m.get("name", "")).strip()
            sources = [s for s in m.get("merged_from", []) if s in by_name and s not in claimed] or (
                [name] if name in by_name and name not in claimed else []
            )
            if not name or not sources:
                continue
            claimed.update(sources)
            rec = {"name": name, "subtopics": [], "key_concepts": [], "prerequisites": [], "unit_ids": []}
            for s in sources:
                rec["subtopics"] += by_name[s]["subtopics"]
                rec["key_concepts"] += by_name[s]["key_concepts"]
                rec["unit_ids"] += by_name[s]["unit_ids"]
            rec["subtopics"] += [str(x) for x in m.get("subtopics", [])]
            rec["key_concepts"] += [str(x) for x in m.get("key_concepts", [])]
            rec["prerequisites"] = [str(x) for x in m.get("prerequisites", [])]
            for f in ("subtopics", "key_concepts", "unit_ids"):
                rec[f] = list(dict.fromkeys(rec[f]))
            final.append(rec)
        leftover = [t for n, t in by_name.items() if n not in claimed]
        if final:
            return _merge_similar(final + leftover)
    except Exception:
        pass
    return _merge_similar(topic_list)


def _tag_units(kb: KnowledgeBase, topics: list[dict]) -> None:
    """Tag units from topic unit_ids; units nobody claimed are assigned by similarity."""
    unit_to_topics: dict[str, list[str]] = {}
    for t in topics:
        for uid in t["unit_ids"]:
            unit_to_topics.setdefault(uid, []).append(t["name"])
    for u in kb.units:
        u.topics = unit_to_topics.get(u.id, [])
    orphans = [u for u in kb.units if not u.topics]
    if orphans:
        vec = TfidfVectorizer(stop_words="english", sublinear_tf=True)
        M = vec.fit_transform([u.text for u in kb.units] + [topic_query(t) for t in topics])
        sims = cosine_similarity(M[: len(kb.units)], M[len(kb.units) :])
        for i, u in enumerate(kb.units):
            if not u.topics and sims[i].max() > 0.02:
                t = topics[int(sims[i].argmax())]
                u.topics = [t["name"]]
                t["unit_ids"].append(u.id)


def _merge_similar(topic_list: list[dict], threshold: float = 0.75) -> list[dict]:
    """Merge topics whose names are near-identical (word Jaccard); "Supervised" vs "Unsupervised"
    learning stay separate because they share only one of two words."""
    def toks(name: str) -> set[str]:
        return {w.rstrip("s") for w in re.findall(r"[a-z0-9]+", name.lower()) if w not in offline.STOP}

    merged: list[dict] = []
    for t in topic_list:
        a = toks(t["name"])
        target = None
        for m in merged:
            b = toks(m["name"])
            if a and b and len(a & b) / len(a | b) >= threshold:
                target = m
                break
        if target is None:
            merged.append(t)
        else:
            for f in ("subtopics", "key_concepts", "prerequisites", "unit_ids"):
                target[f] = list(dict.fromkeys(target[f] + t[f]))
    return merged
