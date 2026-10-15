"""Simulator invariants and evaluation helpers."""
import json
from pathlib import Path

import pytest

from core.evaluate import context_scores
from core.ingest import Unit
from core.rootcause import build_graph
from core.simulate import QuestionBank, SimConfig, SimStudent, run_experiment, run_student, root_gaps
from core.rootcause import ancestors, topo_order

ROOT = Path(__file__).parent.parent
ML = build_graph(json.loads((ROOT / "data/sample_courses/ml/topic_map.json").read_text(encoding="utf-8"))["topics"])


def test_cannot_learn_before_prerequisites():
    cfg = SimConfig(review_success=1.0)
    stu = SimStudent(ML, cfg, 0)
    stu.unknown = {"Overfitting", "Regularization"}
    import random

    rng = random.Random(0)
    stu.review("Regularization", rng)
    assert "Regularization" in stu.unknown, "needs Overfitting first"
    stu.review("Overfitting", rng)
    stu.review("Regularization", rng)
    assert not stu.unknown


def test_answer_probability_follows_ancestors():
    stu = SimStudent(ML, SimConfig(slip=0.0, guess=0.0), 0)
    stu.unknown = {"Overfitting"}
    import random

    rng = random.Random(1)
    assert all(stu.answer("Supervised Learning", rng) for _ in range(20))
    assert not any(stu.answer("Regularization", rng) for _ in range(20)), "descendant of an unknown topic is unanswerable"


def test_same_latent_student_across_policies_and_deterministic():
    a = SimStudent(ML, SimConfig(), 3).initial_unknown
    b = SimStudent(ML, SimConfig(), 3).initial_unknown
    assert a == b and 20 <= 100 * len(a) / len(ML) <= 35
    order = topo_order(ML)
    r1 = run_student("C_rootcause", ML, order, 3, SimConfig())
    r2 = run_student("C_rootcause", ML, order, 3, SimConfig())
    assert r1 == r2


def test_probes_come_from_the_same_budget():
    order = topo_order(ML)
    for i in range(10):
        r = run_student("C_rootcause", ML, order, i, SimConfig())
        assert 0 <= r["probe_share"] <= 1
    # A and B never probe
    assert run_student("B_bkt", ML, order, 0, SimConfig())["probe_share"] == 0


def test_question_bank_repeats_only_when_pool_exhausted():
    bank = QuestionBank(3)
    seen = [bank.next("t") for _ in range(4)]
    assert [rep for _, rep in seen] == [False, False, False, True]


def test_root_gaps_hides_descendants():
    anc = {t: ancestors(ML, t) for t in ML}
    assert root_gaps({"Overfitting", "Regularization"}, anc) == {"Overfitting"}


def test_experiment_shape_and_paired_ci():
    res = run_experiment(ML, SimConfig(students=6))
    assert set(res["summary"]) >= {"A_fixed", "B_bkt", "C_rootcause", "B_rootfilter(ablation)"}
    d = res["paired_vs_baselines"]["C_minus_B.f1_R"]
    assert d["ci95"][0] <= d["mean_diff"] <= d["ci95"][1]


def _u(f, loc):
    return Unit(id=f"{f}{loc}", text="x", source_type="text", source_file=f, location=loc)


def test_context_precision_and_recall_definitions():
    rel = {("a", "1"), ("a", "2")}
    # ranked: relevant, irrelevant, relevant -> AP = (1/1 + 2/3) / 2
    s = context_scores([_u("a", "1"), _u("b", "9"), _u("a", "2")], rel)
    assert s["precision"] == pytest.approx((1 + 2 / 3) / 2)
    assert s["recall"] == 1.0 and s["rr"] == 1.0
    miss = context_scores([_u("b", "9")], rel)
    assert miss["precision"] == 0 and miss["recall"] == 0 and miss["hit"] == 0
