"""Offline fallbacks — the app keeps working (visibly degraded) when no LLM is reachable.

Everything here is deterministic and uses only the uploaded material:
  - extractive_answer : quote the best-matching sentences, with citations
  - cloze_questions   : fill-in-the-blank MCQ/short questions whose answer is a literal span of
                        the source passage (so the key is verifiable by string match)
  - heuristic_topics  : TF-IDF clustering into a topic map (order-based prerequisites)
  - grade_short       : token-overlap / numeric grading for free-text answers
The UI labels every offline result so nobody mistakes it for LLM output.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .ingest import KnowledgeBase, Unit

STOP = set(ENGLISH_STOP_WORDS)
_WORD = re.compile(r"[A-Za-z][A-Za-z\-']+")


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOP and len(w) > 2]


def sentences(text: str) -> list[str]:
    text = re.sub(r"\[Speaker notes\].*", "", text, flags=re.S)
    text = re.sub(r"^#+ .*$", "", text, flags=re.M).replace("**", "")
    out = []
    for block in re.split(r"\n\s*\n|\n(?=\s*[-•*#])", text):
        block = re.sub(r"^[#*\-•\s]+", "", block.strip())
        block = re.sub(r"\s+", " ", block)
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", block):
            s = s.strip(" *")
            if len(s.split()) >= 5:
                out.append(s)
    return out


# ------------------------------------------------------------------ tutor

def extractive_answer(question: str, hits: list[tuple[Unit, float]], max_sentences: int = 3) -> str:
    """Quote the sentences that best overlap the question; cite the unit each came from."""
    q = set(content_words(question))
    scored: list[tuple[float, str, str]] = []
    for u, sim in hits[:4]:
        for s in sentences(u.text):
            overlap = len(q & set(content_words(s)))
            if overlap:
                scored.append((overlap + sim, s, u.id))
    scored.sort(key=lambda x: -x[0])
    picked, seen = [], set()
    for _score, s, uid in scored:
        if s in seen:
            continue
        seen.add(s)
        picked.append(f"{s} [{uid}]")
        if len(picked) >= max_sentences:
            break
    if not picked:
        u = hits[0][0]
        picked = [f"{sentences(u.text)[0] if sentences(u.text) else u.text[:200]} [{u.id}]"]
    return "From your material (offline extractive mode — no language model connected): " + " ".join(picked)


# ------------------------------------------------------------------- quiz

def _doc_freq(units: list[Unit]) -> Counter:
    df: Counter = Counter()
    for u in units:
        df.update(set(content_words(u.text)))
    return df


def domain_terms(kb: KnowledgeBase, per_unit: int = 10) -> list[str]:
    """Most characteristic words of each unit (TF-IDF) — good distractors because they are on-topic."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if not kb.units:
        return []
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{4,}\b")
    try:
        X = vec.fit_transform([u.text for u in kb.units])
    except ValueError:
        return []
    terms = vec.get_feature_names_out()
    pool: list[str] = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        pool += [terms[j] for j in row.argsort()[::-1][:per_unit] if row[j] > 0]
    return list(dict.fromkeys(pool))


def cloze_questions(
    kb: KnowledgeBase,
    units: list[Unit],
    count: int,
    types: list[str],
    seed: int = 0,
    topic_names: list[str] | None = None,
    focus_terms: set[str] | None = None,
) -> list[dict]:
    """Fill-in-the-blank questions. The answer is always a literal word of the source sentence."""
    rng = random.Random(seed)
    df = _doc_freq(kb.units)
    pool_terms = domain_terms(kb) or [w for w, c in df.items() if len(w) >= 5]
    out: list[dict] = []
    order = list(units)
    rng.shuffle(order)
    qn = 0
    for u in order * 2:  # second pass lets a unit yield a second, different sentence
        if len(out) >= count:
            break
        cands = []
        for s in sentences(u.text):
            if not 7 <= len(s.split()) <= 40:
                continue
            if focus_terms and not (focus_terms & set(content_words(s))):
                continue  # keep the question on the requested topic
            terms = [w for w in set(content_words(s)) if len(w) >= 5 and w.isalpha()]
            terms = [w for w in terms if re.search(rf"\b{re.escape(w)}\b", s, flags=re.I)]
            if terms:
                # rarest term in the corpus is the most informative blank
                term = min(sorted(terms), key=lambda w: (df.get(w, 1), w))
                cands.append((s, term))
        used = {q["_blank_sentence"] for q in out if q.get("source_unit_id") == u.id}
        cands = [c for c in cands if c[0] not in used]
        if not cands:
            continue
        s, term = rng.choice(cands)
        blanked = re.sub(rf"\b{re.escape(term)}\b", "_____", s, count=1, flags=re.I)
        # answer as written in the source (keeps original casing)
        m = re.search(rf"\b{re.escape(term)}\b", s, flags=re.I)
        answer = m.group(0) if m else term
        qtype = types[qn % len(types)] if types else "mcq"
        qn += 1
        topic = (u.topics[0] if u.topics else (topic_names[0] if topic_names else "general"))
        for t in u.topics or []:
            if topic_names and t in topic_names:
                topic = t
                break
        n_units = df.get(term, 1)
        q = {
            "id": f"q{len(out) + 1}",
            "type": "mcq" if qtype == "mcq" else "short",
            "difficulty": "easy" if n_units >= 3 else ("medium" if n_units == 2 else "hard"),
            "topic": topic,
            "source_unit_id": u.id,
            "question": f"Complete the statement from your material: “{blanked}”",
            "answer": answer,
            "explanation": f"The source says: “{s}” [{u.id}]",
            "misconception": "",
            "verified": True,
            "verification": "extractive",
            "_blank_sentence": s,
        }
        if q["type"] == "mcq":
            distract = [t for t in pool_terms if t != term.lower() and t not in s.lower()]
            rng.shuffle(distract)
            opts = [answer] + distract[:3]
            if len(opts) < 4:
                continue
            rng.shuffle(opts)
            q["options"] = [f"{'ABCD'[i]}. {o}" for i, o in enumerate(opts)]
            q["answer"] = next(o for o in q["options"] if o[3:] == answer)
        out.append(q)
    return out


def grade_short(response: str, answer: str) -> bool:
    """Offline grading for free-text answers: numeric tolerance, else token overlap."""
    r, a = response.strip().lower(), answer.strip().lower()
    if not r:
        return False
    try:
        rn, an = float(r.replace(",", "")), float(a.replace(",", ""))
        return math.isclose(rn, an, rel_tol=0.02, abs_tol=1e-6)
    except ValueError:
        pass
    if r == a or a in r:
        return True
    rt, at = set(content_words(r)), set(content_words(a))
    if not at:
        return r == a
    overlap = len(rt & at) / len(at)
    return overlap >= 0.6


# ----------------------------------------------------------------- topics

def heuristic_topics(kb: KnowledgeBase, max_topics: int = 8) -> dict:
    """Cluster units with TF-IDF + KMeans. Prerequisite = most similar earlier cluster."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    units = kb.units
    if not units:
        return {"topics": []}
    k = max(1, min(max_topics, round(len(units) / 3)))
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform([u.text for u in units])
    if k == 1 or X.shape[0] < 3:
        labels = np.zeros(len(units), dtype=int)
    else:
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    clusters = []
    for c in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == c]
        centroid = np.asarray(X[idx].mean(axis=0)).ravel()
        top = [t for t in terms[np.argsort(-centroid)[:8]] if " " not in t][:4]
        name = " ".join(w.capitalize() for w in top[:2]) or f"Topic {c + 1}"
        clusters.append({"idx": idx, "centroid": centroid, "name": name, "terms": top, "pos": float(np.mean(idx))})
    clusters.sort(key=lambda c: c["pos"])
    names: set[str] = set()
    for c in clusters:  # unique names
        base, n = c["name"], 2
        while c["name"] in names:
            c["name"] = f"{base} {n}"
            n += 1
        names.add(c["name"])
    topics = []
    for i, c in enumerate(clusters):
        prereqs = []
        best, best_sim = None, 0.0
        for j in range(i):
            a, b = c["centroid"], clusters[j]["centroid"]
            sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            if sim > best_sim:
                best, best_sim = clusters[j]["name"], sim
        if best and best_sim > 0.12:
            prereqs.append(best)
        topics.append(
            {
                "name": c["name"],
                "subtopics": [],
                "key_concepts": c["terms"],
                "prerequisites": prereqs,
                "unit_ids": [units[i].id for i in c["idx"]],
            }
        )
    return {"topics": topics, "mode": "heuristic"}
