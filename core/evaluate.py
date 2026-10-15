"""System evaluation — retrieval, grounding, assessment quality and personalization.

Framework: RAGAS was evaluated and NOT used. It resolves, but needs a funded LLM for every metric and
pulls ~40 packages (langchain, datasets, huggingface_hub 2.x) that conflict with faster-whisper in our
environment. We re-implemented the same metric definitions natively and say so in every report:

  context precision  (ID-based) RAGAS "context precision": mean of precision@k over the ranks that hold a
                     relevant unit; relevance = the unit is in the test case's ground-truth `relevant` list
  context recall     (ID-based) |retrieved ∩ relevant| / |relevant|
  faithfulness       RAGAS-style two steps: (1) LLM splits the answer into atomic claims, (2) LLM checks each
                     claim against the retrieved context; score = supported claims / all claims   [LLM]
  answer relevancy   RAGAS-style: LLM writes 3 questions the answer would address; score = mean TF-IDF cosine
                     between them and the real question (RAGAS uses embeddings; TF-IDF keeps it local) [LLM]
  citation precision share of cited units that are ground-truth relevant (no LLM)
  refusal            off-material queries must be declined: gate-only (no LLM) and end-to-end tutor

Metrics that need an LLM are reported as {"status": "skipped", "reason": ...} when none is reachable —
never estimated. Personalization: core/simulate.py (latent-gap simulated students).
"""
from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import llm
from .quiz import generate as quiz_generate
from .rootcause import build_graph
from .simulate import SimConfig, run_experiment
from .store import Course

ROOT = Path(__file__).resolve().parent.parent
COURSES = {
    "ml": {
        "dir": ROOT / "data" / "sample_courses" / "ml",
        "files": ["ml_notes.md", "lecture_slides.pptx", "lecture_transcript.vtt",
                  "model_selection_chapter.pdf", "lecture2_model_selection.mp4"],
    },
    "os": {"dir": ROOT / "data" / "sample_courses" / "os", "files": ["os_notes.txt"]},
}


# ------------------------------------------------------------ test set / courses

def load_testset(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["cases"] if isinstance(raw, dict) else raw


def build_course(name: str, workdir: Path | None = None, extra_files: list[Path] | None = None) -> Course:
    """Fresh course (own temp data dir) from the bundled sample files, tagged with the curated map."""
    spec = COURSES[name]
    course = Course(workdir or Path(tempfile.mkdtemp(prefix=f"sc_eval_{name}_")))
    course.skipped = []
    for f in spec["files"]:
        try:
            course.add_file(spec["dir"] / f)
        except ValueError as e:  # e.g. no Whisper available for the video: report, do not crash
            course.skipped.append({"file": f, "reason": str(e)[:160]})
    for f in extra_files or []:
        course.add_file(f)
    course.analyze(prefer_seed=True)
    return course


def vision_ok(course: Course) -> bool:
    """True when at least one figure unit carries a real description (a vision model ran at ingestion)."""
    return any(u.kind == "figure" and "not captured" not in u.text for u in course.kb.units)


def usable(case: dict, courses: dict[str, "Course"]) -> bool:
    """Skip cases whose ground truth is missing from the loaded course or that need an unavailable vision model."""
    co = courses.get(case["course"])
    if co is None:
        return False
    if case.get("requires_vision") and not vision_ok(co):
        return False
    return True


def _relevant(case: dict) -> set[tuple[str, str]]:
    return {(f, loc) for f, loc in case.get("relevant", [])}


def _is_rel(unit, rel: set[tuple[str, str]]) -> bool:
    return (unit.source_file, unit.location) in rel


# ------------------------------------------------------------------ retrieval

def context_scores(retrieved: list, rel: set) -> dict:
    """ID-based context precision (RAGAS definition) and recall for one ranked list of units."""
    flags = [_is_rel(u, rel) for u in retrieved]
    hits, ap = 0, 0.0
    for i, f in enumerate(flags, 1):
        if f:
            hits += 1
            ap += hits / i
    found = {(u.source_file, u.location) for u, f in zip(retrieved, flags) if f}
    return {
        "precision": ap / hits if hits else 0.0,
        "recall": len(found & rel) / len(rel) if rel else 0.0,
        "hit": float(hits > 0),
        "rr": next((1 / i for i, f in enumerate(flags, 1) if f), 0.0),
    }


def eval_retrieval(courses: dict[str, Course], cases: list[dict], k: int = 4) -> dict:
    rows = []
    for c in cases:
        if c.get("off_material") or not usable(c, courses):
            continue
        hits = courses[c["course"]].retriever.search(c["question"], k=k)
        rows.append(context_scores([u for u, _ in hits], _relevant(c)))
    n = max(len(rows), 1)
    return {
        "k": k,
        "cases": len(rows),
        "hit_rate": round(sum(r["hit"] for r in rows) / n, 3),
        "mrr": round(sum(r["rr"] for r in rows) / n, 3),
        "context_precision": round(sum(r["precision"] for r in rows) / n, 3),
        "context_recall": round(sum(r["recall"] for r in rows) / n, 3),
    }


# ------------------------------------------------------------------ refusal gate

def _gate_features(course: Course, q: str) -> tuple[float, float]:
    hits = course.retriever.search(q, k=1)
    return (hits[0][1] if hits else 0.0), course.retriever.coverage(q)


def calibrate_gate(courses: dict[str, Course], cases: list[dict]) -> dict:
    """Choose (min_score, min_coverage) on the dev split; report accuracy on the untouched test split."""
    scored = [(c, *_gate_features(courses[c["course"]], c["question"])) for c in cases if usable(c, courses)]

    def acc(split: str, tau: float, kappa: float) -> tuple[float, float, int, int]:
        inm = [(s, cv) for c, s, cv in scored if c["split"] == split and not c.get("off_material")]
        off = [(s, cv) for c, s, cv in scored if c["split"] == split and c.get("off_material")]
        ans = sum(s >= tau and cv >= kappa for s, cv in inm) / max(len(inm), 1)
        ref = sum(not (s >= tau and cv >= kappa) for s, cv in off) / max(len(off), 1)
        return ans, ref, len(inm), len(off)

    grid = [(round(t, 3), k) for t in np.arange(0.02, 0.31, 0.01) for k in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6)]
    bal = {g: sum(acc("dev", *g)[:2]) / 2 for g in grid}
    best = max(bal.values())
    tied = sorted(g for g, v in bal.items() if abs(v - best) < 1e-9)
    tau, kappa = tied[len(tied) // 2]
    ans, ref, n_in, n_off = acc("test", tau, kappa)
    d_ans, d_ref, dn_in, dn_off = acc("dev", tau, kappa)
    bs, br, _, _ = acc("test", 0.01, 0.0)
    # recall-first gate for online mode: the LLM's own NOT_COVERED is the second refusal layer, so the first gate
    # only has to drop the clearly unrelated queries without ever blocking a dev in-material question
    lo = max((t for t, k in grid if k == 0.0 and acc("dev", t, 0.0)[0] >= 1.0), default=0.02)
    lo_ans, lo_ref, _, _ = acc("test", lo, 0.0)
    return {
        "min_score": tau,
        "min_score_online": lo,
        "online_gate_test": {"answered_in_material": round(lo_ans, 3), "refused_off_material": round(lo_ref, 3)},
        "min_coverage": kappa,
        "chosen_on": "dev split: best balanced accuracy over (min_score, min_coverage); middle of the tied optimum",
        "dev": {"answered_in_material": round(d_ans, 3), "refused_off_material": round(d_ref, 3), "n_in": dn_in, "n_off": dn_off},
        "test": {"answered_in_material": round(ans, 3), "refused_off_material": round(ref, 3), "n_in": n_in, "n_off": n_off},
        "no_gate_baseline_test": {"answered_in_material": round(bs, 3), "refused_off_material": round(br, 3)},
    }


# ------------------------------------------------------------ generation (LLM-judged)

CLAIMS_SYSTEM = 'Split the answer into atomic factual claims. Ignore citation markers like [u1234abcd]. Return JSON {"claims": ["..."]} (max 8).'
SUPPORT_SYSTEM = (
    "For each claim decide whether it is directly supported by the context. "
    'Return JSON {"verdicts": [true|false, ...]} in the same order as the claims.'
)
QGEN_SYSTEM = 'Write 3 different questions that the following answer would be a good response to. Return JSON {"questions": ["...", "...", "..."]}.'


def _faithfulness(answer: str, context: str) -> float | None:
    claims = llm.chat_json([{"role": "system", "content": CLAIMS_SYSTEM}, {"role": "user", "content": answer}], max_tokens=500)
    claims = [str(c) for c in (claims.get("claims", []) if isinstance(claims, dict) else [])][:8]
    if not claims:
        return None
    body = "Context:\n" + context + "\n\nClaims:\n" + "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    v = llm.chat_json([{"role": "system", "content": SUPPORT_SYSTEM}, {"role": "user", "content": body}], max_tokens=300)
    verdicts = [bool(x) for x in (v.get("verdicts", []) if isinstance(v, dict) else [])][: len(claims)]
    return sum(verdicts) / len(claims) if verdicts else None


def _answer_relevancy(question: str, answer: str) -> float | None:
    out = llm.chat_json([{"role": "system", "content": QGEN_SYSTEM}, {"role": "user", "content": answer}], max_tokens=250)
    qs = [str(q) for q in (out.get("questions", []) if isinstance(out, dict) else [])][:3]
    if not qs:
        return None
    vec = TfidfVectorizer(stop_words="english").fit([question] + qs)
    sims = cosine_similarity(vec.transform([question]), vec.transform(qs))[0]
    return float(sims.mean())


def eval_generation(courses: dict[str, Course], cases: list[dict]) -> dict:
    from .tutor import answer as tutor_answer

    out: dict = {"mode": "online" if llm.available() else "offline", "model": llm.status()["model"] or None}
    inm = [c for c in cases if not c.get("off_material") and usable(c, courses)]
    off = [c for c in cases if c.get("off_material") and usable(c, courses)]

    # refusal: end-to-end tutor on off-material, plus the no-LLM gate-only number
    refused = 0
    for c in off:
        r = tutor_answer(c["question"], courses[c["course"]].retriever)
        refused += int(r["refused"])
    out["refusal_end_to_end"] = {"cases": len(off), "refused": refused, "rate": round(refused / max(len(off), 1), 3)}

    # in-material: citation precision/hit and answered rate (no LLM needed)
    answered, cit_prec, cit_hit, ctx = 0, [], 0, []
    judged: list[tuple[dict, dict]] = []
    for c in inm:
        co = courses[c["course"]]
        r = tutor_answer(c["question"], co.retriever)
        if r["refused"]:
            continue
        answered += 1
        rel = _relevant(c)
        cited = [(x["source_file"], x["location"]) for x in r["citations"]]
        good = [x for x in cited if x in rel]
        cit_prec.append(len(good) / len(cited) if cited else 0.0)
        cit_hit += int(bool(good))
        judged.append((c, r))
    out["in_material"] = {
        "cases": len(inm), "answered": answered,
        "answer_rate": round(answered / max(len(inm), 1), 3),
        "citation_precision": round(sum(cit_prec) / max(len(cit_prec), 1), 3),
        "citation_hit_rate": round(cit_hit / max(answered, 1), 3),
        "answer_mode": judged[0][1].get("mode") if judged else None,
    }

    if not llm.available() or (judged and judged[0][1].get("mode") != "llm"):
        why = llm.status()["reason"] or "answers were produced by the offline extractive fallback"
        for k in ("faithfulness", "answer_relevancy"):
            out[k] = {"status": "skipped", "reason": f"needs a working LLM ({why})"}
        return out

    faith, rel_scores = [], []
    for c, r in judged:
        co = courses[c["course"]]
        by_id = {u.id: u.text for u in co.kb.units}
        context = "\n\n".join(f"[{x['unit_id']}] {by_id.get(x['unit_id'], x['excerpt'])}" for x in r["citations"])
        try:
            f = _faithfulness(r["answer"], context)
            a = _answer_relevancy(c["question"], r["answer"])
        except Exception:
            continue
        if f is not None:
            faith.append(f)
        if a is not None:
            rel_scores.append(a)
    out["faithfulness"] = {"score": round(float(np.mean(faith)), 3) if faith else None, "n": len(faith), "method": "claim decomposition + per-claim support check"}
    out["answer_relevancy"] = {"score": round(float(np.mean(rel_scores)), 3) if rel_scores else None, "n": len(rel_scores), "method": "3 regenerated questions, TF-IDF cosine"}
    return out


# ------------------------------------------------------------- assessment quality

def eval_assessment(course: Course, rounds: int = 3, per_round: int = 4) -> dict:
    """Generate several assessments for one simulated student and audit them (no answers needed)."""
    names = [t["name"] for t in course.topics if len([u for u in course.kb.units if t["name"] in (u.topics or [])]) >= 2][:3]
    hashes: set[str] = set()
    stems: list[str] = []
    used: dict[str, int] = {}
    qs_all: list[list[dict]] = []
    for r in range(rounds):
        qs = quiz_generate(
            course.kb, names, count=per_round, avoid_hashes=hashes, avoid_stems=stems, used_units=used,
            known_topics=course.topic_names(), topic_terms=course.topic_terms(), seed=r,
        )
        for q in qs:
            hashes.add(q["_hash"])
            stems.append(q["_stem"])
            used[q["source_unit_id"]] = used.get(q["source_unit_id"], 0) + 1
        qs_all.append(qs)
    flat = [q for qs in qs_all for q in qs]
    unit_by_id = {u.id: u for u in course.kb.units}
    prior: list[str] = []
    repeats = 0
    for qs in qs_all:
        for q in qs:
            repeats += int(any(_near(q["_stem"], p) for p in prior))
        prior += [q["_stem"] for q in qs]
    tagged = sum(
        1 for q in flat
        if q["source_unit_id"] in unit_by_id and q["topic"] in course.topic_names()
        and q["topic"] in (unit_by_id[q["source_unit_id"]].topics or [])
    )
    grounded = 0
    for q in flat:
        ans_words = set(re.findall(r"[a-z0-9]+", str(q.get("answer", "")).lower()))
        src_words = set(re.findall(r"[a-z0-9]+", unit_by_id[q["source_unit_id"]].text.lower()))
        grounded += int(bool(ans_words) and len(ans_words & src_words) / len(ans_words) >= 0.5)
    n = max(len(flat), 1)
    return {
        "mode": "online" if llm.available() else "offline",
        "assessments": rounds, "questions": len(flat),
        "verified_rate": round(sum(bool(q.get("verified")) for q in flat) / n, 3),
        "verification_kinds": dict(Counter(q.get("verification", "none") for q in flat)),
        "tag_consistency": round(tagged / n, 3),
        "answer_grounded_in_source": round(grounded / n, 3),
        "repeat_rate_across_assessments": round(repeats / n, 3),
        "types": dict(Counter(q.get("type") for q in flat)),
        "difficulties": dict(Counter(q.get("difficulty") for q in flat)),
    }


def _near(a: str, b: str, thr: float = 0.8) -> bool:
    sa, sb = set(a.split()), set(b.split())
    return bool(sa and sb) and len(sa & sb) / len(sa | sb) >= thr


# --------------------------------------------------------------- personalization

def eval_personalization(sensitivity: bool = True) -> dict:
    out: dict = {}
    for name, spec in COURSES.items():
        graph = build_graph(json.loads((spec["dir"] / "topic_map.json").read_text(encoding="utf-8"))["topics"])
        res = run_experiment(graph, SimConfig())
        if sensitivity:
            res["sensitivity"] = {}
            for label, cfg in (
                ("review_success=0.4", SimConfig(review_success=0.4)),
                ("review_success=1.0", SimConfig(review_success=1.0)),
                ("no_diagnostic_sweep", SimConfig(sweep=False)),
                ("seed=1", SimConfig(seed=1)),
            ):
                r = run_experiment(graph, cfg)
                res["sensitivity"][label] = {
                    p: {m: r["summary"][p][m]["mean"] for m in ("f1_R", "f1_U", "gain")} for p in ("A_fixed", "B_bkt", "C_rootcause")
                }
        out[name] = res
    return out


# -------------------------------------------------------------------------- run

def run_all(testset_path: Path, *, skip_generation: bool = False, extra: dict[str, list[Path]] | None = None) -> dict:
    cases = load_testset(testset_path)
    courses = {n: build_course(n, extra_files=(extra or {}).get(n)) for n in COURSES}
    report: dict = {
        "framework": "native re-implementation of RAGAS metric definitions (RAGAS itself not used; see core/evaluate.py docstring)",
        "llm": llm.status(),
        "testset": {
            "in_material": sum(1 for c in cases if not c.get("off_material")),
            "off_material": sum(1 for c in cases if c.get("off_material")),
            "courses": {n: len(c.kb.units) for n, c in courses.items()},
            "evaluated_in_material": sum(1 for c in cases if not c.get("off_material") and usable(c, courses)),
            "figure_cases_total": sum(1 for c in cases if c.get("requires_vision")),
            "figure_cases_evaluated": sum(1 for c in cases if c.get("requires_vision") and usable(c, courses)),
            "skipped_files": {n: getattr(c, "skipped", []) for n, c in courses.items() if getattr(c, "skipped", [])},
        },
        "retrieval": eval_retrieval(courses, cases),
        "retrieval_gate": calibrate_gate(courses, cases),
    }
    if not skip_generation:
        report["generation"] = eval_generation(courses, cases)
    report["assessment"] = eval_assessment(courses["ml"])
    report["personalization"] = eval_personalization()
    return report
