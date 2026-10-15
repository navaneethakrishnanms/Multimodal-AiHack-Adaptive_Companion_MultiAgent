"""Focused unit tests for quiz grading/novelty, learner model, recommendations, revision and the LLM circuit breaker."""
import json
import time
import urllib.error
from pathlib import Path

import pytest

from core import llm, offline
from core.ingest import KnowledgeBase
from core.learner import LearnerModel, bkt_update
from core.quiz import canonical_topic, generate, grade, is_repeat, stem_norm
from core.recommend import recommend
from core.report import assessment_report, misconception_note
from core.revision import brief, flashcards
from core.rootcause import build_graph

ROOT = Path(__file__).parent.parent
ML = ROOT / "data" / "sample_courses" / "ml"


@pytest.fixture()
def kb():
    k = KnowledgeBase()
    for f in ("ml_notes.md", "lecture_slides.pptx"):
        k.add_file(ML / f)
    topics = json.loads((ML / "topic_map.json").read_text(encoding="utf-8"))["topics"]
    from core.retrieve import assign_units

    assign_units(k, topics)
    k.topics = topics  # test convenience
    return k


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    for k in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    llm.reset()


# ------------------------------------------------------------------ learner

def test_bkt_direction_and_bounds():
    assert bkt_update(0.5, True) > 0.5 > bkt_update(0.5, False)
    assert 0.01 <= bkt_update(0.01, False) <= 0.995 and bkt_update(0.99, True) <= 0.995
    # a correct free-text answer is stronger evidence than a correct MCQ (lower guess rate)
    assert bkt_update(0.25, True, "short") > bkt_update(0.25, True, "mcq")


def test_mastery_moves_only_on_answers():
    m = LearnerModel(None)
    m.touch("A")
    assert m.get("A") == m.prior and m.attempts("A") == 0
    m.record_misconception("A", "confuses X with Y")
    assert m.get("A") == m.prior
    m.update("A", True)
    assert m.get("A") > m.prior and m.attempts("A") == 1


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    m = LearnerModel(p)
    m.update("A", False)
    again = LearnerModel(p)
    assert again.get("A") == m.get("A") and again.attempts("A") == 1


def test_seed_prior_only_before_answers():
    m = LearnerModel(None)
    m.seed_prior("A", 0.5)
    assert m.get("A") == 0.5
    m.update("A", True)
    before = m.get("A")
    m.seed_prior("A", 0.1)
    assert m.get("A") == before


def test_retention_decays_and_plan_prioritises_weak():
    m = LearnerModel(None)
    for _ in range(3):
        m.update("Strong", True)
    m.update("Weak", False)
    now = time.time()
    m.data["Strong"]["last_seen"] = now - 10 * 86400
    assert m.retention("Strong", now) < m.retention("Strong", now - 9 * 86400)
    plan = m.study_plan(["Strong", "Weak", "Never"], days_to_exam=3, minutes_per_day=30)
    assert plan[0]["topics"][0] in ("Weak", "Never"), "weakest/least-known topic comes first"
    assert all(d["minutes_each"] >= 10 for d in plan)


# ------------------------------------------------------------------- quiz

def test_grade_mcq_by_text_letter_and_prefix():
    key = {"type": "mcq", "answer": "B. Overshoot and diverge"}
    assert grade(key, "B. Overshoot and diverge")["correct"]
    assert grade(key, "b")["correct"]
    assert grade(key, "overshoot and diverge")["correct"]
    assert not grade(key, "A. Converge slowly")["correct"]


def test_grade_free_text_offline_numeric_and_overlap():
    assert grade({"type": "numerical", "answer": "3.14"}, "3,14".replace(",", "."))["correct"]
    assert grade({"type": "numerical", "answer": "100"}, "101")["correct"], "2% tolerance"
    assert not grade({"type": "numerical", "answer": "100"}, "150")["correct"]
    assert grade({"type": "short", "answer": "mean squared error"}, "It is the mean squared error")["correct"]
    assert not grade({"type": "short", "answer": "mean squared error"}, "cross entropy")["correct"]
    assert offline.grade_short("", "x") is False


def test_novelty_exact_and_near_duplicates():
    a = "Complete the statement: the learning rate controls the step size of gradient descent"
    assert is_repeat(a, {hash_of(a)}, [])
    assert is_repeat(a + " today", set(), [stem_norm(a)]), "near duplicate (Jaccard ≥ 0.8) is rejected"
    assert not is_repeat("What is L1 regularization?", set(), [stem_norm(a)])


def hash_of(text):
    from core.quiz import _stem_hash

    return _stem_hash(text)


def test_canonical_topic_snaps_name_drift():
    known = ["Gradient Descent", "Learning Rate", "Regularization"]
    assert canonical_topic("gradient descent", known) == "Gradient Descent"
    assert canonical_topic("Learning rate tuning", known) == "Learning Rate"
    assert canonical_topic("Quantum Gravity", known) == "Quantum Gravity", "unknown names are left alone"


def test_generate_rotates_units_and_never_repeats_for_a_student(kb):
    seen_h, seen_s, used = set(), [], {}
    stems = set()
    for r in range(4):
        qs = generate(kb, ["Learning Rate"], count=3, avoid_hashes=seen_h, avoid_stems=seen_s, used_units=used,
                      known_topics=[t["name"] for t in kb.topics], topic_terms=None, seed=r)
        for q in qs:
            assert q["source_unit_id"] in {u.id for u in kb.units} and q["verified"]
            assert q["question"] not in stems
            stems.add(q["question"])
            seen_h.add(q["_hash"])
            seen_s.append(q["_stem"])
            used[q["source_unit_id"]] = used.get(q["source_unit_id"], 0) + 1
    assert len(stems) >= 6


def test_cloze_key_is_literal_span_of_source(kb):
    qs = offline.cloze_questions(kb, kb.units[:6], 6, ["mcq", "short"], seed=1)
    by_id = {u.id: u.text.lower() for u in kb.units}
    for q in qs:
        ans = q["answer"][3:] if q["type"] == "mcq" else q["answer"]
        assert ans.lower() in by_id[q["source_unit_id"]]
        if q["type"] == "mcq":
            assert len(q["options"]) == 4 and len(set(q["options"])) == 4


# ------------------------------------------------- report / recommend / revision

def test_recommendations_explain_themselves_and_find_root_cause(kb):
    graph = build_graph(kb.topics)
    m = LearnerModel(None)
    for _ in range(3):
        for t in ("Supervised Learning", "Linear Regression", "Mean Squared Error"):
            m.update(t, True)
    m.update("Regularization", False)
    sources = lambda t, k: [{"source_file": "ml_notes.md", "location": "para 7"}]  # noqa: E731
    recs = recommend(m, graph, [t["name"] for t in kb.topics], sources)
    root = next(r for r in recs if r["action"] == "rootcause")
    assert root["topic"] == "Regularization" and root["suspect"] == "Overfitting"
    assert all(r["why"] and r["sources"] is not None for r in recs)
    assert recs[0]["priority"] == 1
    # a brand-new student gets the diagnostic
    fresh = recommend(LearnerModel(None), graph, [t["name"] for t in kb.topics], sources)
    assert fresh[0]["action"] == "diagnostic"


def test_report_groups_by_topic_and_flags_suspect(kb):
    graph = build_graph(kb.topics)
    m = LearnerModel(None)
    results = []
    for i, ok in enumerate([False, False, True]):
        m.update("Regularization", ok)
        results.append({"topic": "Regularization", "correct": ok, "quiz_id": "z1", "source": "quiz",
                        "misconception": "" if ok else "thinks L1 shrinks weights smoothly"})
    rep = assessment_report(m, results, graph, lambda t, k: [], "z1")
    assert rep["answered"] == 3 and rep["correct"] == 1 and rep["weak_topics"] == ["Regularization"]
    assert rep["topics"][0]["suspect_prerequisite"]["topic"] in ("Overfitting", "Mean Squared Error")
    assert rep["misconceptions"][0]["count"] == 2


def test_misconception_note_prefers_question_field_then_generic():
    assert "confuses" in misconception_note({"misconception": "confuses L1 and L2"}, "x")
    note = misconception_note({"type": "mcq", "answer": "lasso", "question": "q"}, "ridge")
    assert "ridge" in note and "lasso" in note


def test_flashcards_and_brief_keep_sources(kb):
    cards = flashcards(kb, ["Regularization"], per_topic=3)
    assert cards and all(c["source"]["source_file"] and c["back"] for c in cards)
    b = brief(kb, "Regularization")
    assert b["text"].startswith("Revision brief") and b["sources"]
    assert brief(kb, "No Such Topic")["mode"] == "none"


# ------------------------------------------------------ llm circuit breaker

class _Resp:
    def __init__(self, code, body):
        self.code, self._body = code, body

    def read(self):
        return self._body.encode()


def test_circuit_breaker_trips_on_quota_and_recovers(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    calls = []

    def boom(req, timeout=0):
        calls.append(1)
        raise urllib.error.HTTPError("http://x", 429, "Too Many", {}, _Resp(429, '{"error":{"code":"insufficient_quota"}}'))

    monkeypatch.setattr(llm.urllib.request, "urlopen", boom)
    assert llm.available()
    with pytest.raises(llm.LLMError):
        llm.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1, "no retries on a quota failure"
    assert not llm.available() and "credits" in llm.status()["reason"]
    llm.reset()
    assert llm.available()


def test_env_loader_never_overrides_and_can_be_disabled(tmp_path, monkeypatch):
    envf = tmp_path / ".env"
    envf.write_text("LLM_MODEL=from-file\nOTHER=1\n", encoding="utf-8")
    monkeypatch.setenv("LLM_MODEL", "already-set")
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.delenv("STUDY_NO_ENV", raising=False)
    llm.load_env(envf)
    assert llm.os.environ["LLM_MODEL"] == "already-set" and llm.os.environ["OTHER"] == "1"
    monkeypatch.setenv("STUDY_NO_ENV", "1")
    monkeypatch.delenv("OTHER", raising=False)
    llm.load_env(envf)
    assert "OTHER" not in llm.os.environ


def test_reject_then_regenerate_fills_the_shortfall(kb, monkeypatch):
    from core import quiz as quiz_mod

    uid = kb.units[0].id
    calls = []

    def fake(units, topics, known, need, difficulty, types, stems):
        calls.append(need)
        if len(calls) == 1:
            return [
                {"type": "numerical", "question": "What is the cost of inverting X transpose X for d features?", "answer": "cubic",
                 "source_unit_id": uid, "topic": "Normal Equations", "explanation": "cost grows like d cubed [x]"},
                {"type": "mcq", "question": "Which loss does linear regression usually minimize?",
                 "options": ["A. MSE", "A. MSE", "C. Hinge", "D. Log"], "answer": "A. MSE", "source_unit_id": uid,
                 "topic": "Mean Squared Error", "explanation": "the notes define it"},
                {"type": "short", "question": "What does alpha denote in gradient descent?", "answer": "the learning rate",
                 "source_unit_id": uid, "topic": "Learning Rate", "explanation": "alpha is the learning rate"},
            ]
        return [{"type": "short", "question": "Name the penalty that L1 regularization adds to the loss.",
                 "answer": "lambda times the L1 norm", "source_unit_id": uid, "topic": "Regularization",
                 "explanation": "L1 adds lambda times the norm"}]

    monkeypatch.setattr(quiz_mod, "_llm_questions", fake)
    monkeypatch.setattr(llm, "available", lambda: True)
    monkeypatch.setattr(quiz_mod, "_verify", lambda q, src: dict(q, verified=True, verification="independent-pass"))
    qs = quiz_mod.generate(kb, [], count=2, known_topics=[t["name"] for t in kb.topics])
    assert calls == [2, 1], "second round only asks for what is still missing"
    assert [q["type"] for q in qs] == ["short", "short"]
    assert all(q["verified"] for q in qs)
