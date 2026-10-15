"""RootCause tests on a hand-built 5-node prerequisite graph.

    A Algebra ──► B Derivatives ──┐
        │                         ├──► D Gradient Descent ──► E Learning Rate
        └──────► C Loss Functions ┘
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.learner import LearnerModel, bkt_update
from core.rootcause import (
    RootCause,
    ancestors,
    blame_scores,
    build_graph,
    repair_path,
    should_probe,
    topo_order,
)

TOPICS = [
    {"name": "Algebra", "prerequisites": []},
    {"name": "Derivatives", "prerequisites": ["algebra"]},
    {"name": "Loss Functions", "prerequisites": ["Algebra"]},
    {"name": "Gradient Descent", "prerequisites": ["Derivatives", "loss functions"]},
    {"name": "Learning Rate", "prerequisites": ["gradient descent"]},
]


def graph():
    return build_graph(TOPICS)


def learner_with(**state):
    """state: topic -> (mastery, attempts)"""
    m = LearnerModel(None)
    for t, (p, n) in state.items():
        rec = m._topic(t.replace("_", " "))
        rec["p"], rec["attempts"] = p, n
    return m


def units(topic, k):
    return [{"unit_id": f"{topic}-{i}", "source_file": "notes.md", "location": f"para {i}"} for i in range(k)]


# ---------------------------------------------------------------- graph

def test_build_graph_resolves_names():
    g = graph()
    assert g["Derivatives"] == ["Algebra"]
    assert g["Gradient Descent"] == ["Derivatives", "Loss Functions"]
    assert g["Algebra"] == []


def test_build_graph_drops_unknown_self_and_cycles():
    topics = [
        {"name": "X", "prerequisites": ["X", "Nonexistent Topic", "Y"]},
        {"name": "Y", "prerequisites": ["X"]},  # X<->Y cycle
    ]
    g = build_graph(topics)
    assert g["X"] == ["Y"] or g["Y"] == ["X"]
    assert not (g["X"] and g["Y"]), "exactly one edge of the cycle must survive"
    assert "X" not in g["X"]


def test_ancestors_and_topo_order():
    g = graph()
    assert ancestors(g, "Learning Rate") == {"Gradient Descent", "Derivatives", "Loss Functions", "Algebra"}
    order = topo_order(g)
    for t, prereqs in g.items():
        for p in prereqs:
            assert order.index(p) < order.index(t)


# ---------------------------------------------------------------- blame

def test_blame_is_normalised_and_prefers_never_probed():
    # Derivatives probed & weak (0.2), Loss Functions never probed (prior 0.25)
    m = learner_with(Derivatives=(0.2, 3))
    rows = blame_scores(graph(), "Gradient Descent", m)
    assert pytest.approx(sum(r["blame"] for r in rows), abs=0.01) == 1.0
    # raw: Derivatives 0.8 ; Loss Functions 0.75 * 1.5 = 1.125  -> unprobed wins
    assert rows[0]["topic"] == "Loss Functions" and not rows[0]["probed"]


def test_blame_prefers_weaker_when_both_probed():
    m = learner_with(Derivatives=(0.2, 3), Loss_Functions=(0.7, 3))
    rows = blame_scores(graph(), "Gradient Descent", m)
    assert rows[0]["topic"] == "Derivatives"


def test_blame_formula_values():
    m = learner_with(Derivatives=(0.2, 3))
    rows = {r["topic"]: r for r in blame_scores(graph(), "Gradient Descent", m)}
    assert rows["Derivatives"]["blame"] == pytest.approx(0.8 / (0.8 + 1.125), abs=1e-3)


def test_exclude_gives_next_candidate():
    m = learner_with()
    rows = blame_scores(graph(), "Gradient Descent", m, exclude=["Derivatives"])
    assert [r["topic"] for r in rows] == ["Loss Functions"]


def test_should_probe_false_when_all_strong():
    m = learner_with(Derivatives=(0.95, 4), Loss_Functions=(0.9, 4))
    assert not should_probe(blame_scores(graph(), "Gradient Descent", m))


# ---------------------------------------------------------- diagnose / conclude

def make_rc(m, probe_log=None):
    def make_probe(topic, n):
        if probe_log is not None:
            probe_log.append((topic, n))
        return [{"id": f"{topic}-q{i}"} for i in range(n)]

    return RootCause(graph(), m, units, make_probe)


def test_diagnose_probes_argmax_with_two_questions_and_does_not_touch_mastery():
    m = learner_with(Derivatives=(0.2, 3))
    before = dict(m.mastery())
    log = []
    out = make_rc(m, log).diagnose("Gradient Descent")
    assert out["target"] == "Loss Functions"
    assert log == [("Loss Functions", 2)] and len(out["probe"]) == 2
    assert m.mastery() == before, "blame must never change mastery"
    assert m.attempts("Loss Functions") == 0


def test_no_prerequisites_means_no_probe():
    out = make_rc(learner_with()).diagnose("Algebra")
    assert out["target"] is None and out["probe"] == []


def test_confirmed_weak_only_after_probe_evidence():
    m = learner_with()
    rc = make_rc(m)
    # before any probe answers the prerequisite is just unknown, not "confirmed"
    assert rc.conclude("Gradient Descent", "Derivatives")["confirmed_weak"] is False
    m.update("Derivatives", False, source="probe")
    m.update("Derivatives", False, source="probe")
    res = rc.conclude("Gradient Descent", "Derivatives")
    assert res["confirmed_weak"] is True
    assert res["repair"]["retest_topic"] == "Gradient Descent"


def test_probe_correct_clears_prerequisite_and_suggests_next():
    m = learner_with()
    rc = make_rc(m)
    m.update("Derivatives", True, source="probe")
    m.update("Derivatives", True, source="probe")
    res = rc.conclude("Gradient Descent", "Derivatives")
    assert res["confirmed_weak"] is False
    assert res["next_candidate"] == "Loss Functions"


def test_mastery_moves_exactly_like_bkt():
    m = learner_with()
    expected = bkt_update(bkt_update(0.25, False), False)
    m.update("Derivatives", False, source="probe")
    m.update("Derivatives", False, source="probe")
    assert m.get("Derivatives") == expected


# ---------------------------------------------------------------- repair

def test_repair_path_recurses_one_level_only():
    # Learning Rate wrong -> Gradient Descent confirmed weak; its prereqs:
    #   Derivatives known weak (-> foundation), Loss Functions never tested (-> unverified).
    # Derivatives' own prerequisite Algebra is weak too but must NOT be added (one level max).
    m = learner_with(Gradient_Descent=(0.2, 2), Derivatives=(0.2, 3), Algebra=(0.1, 3))
    path = repair_path(graph(), "Gradient Descent", "Learning Rate", m, units)
    assert [s["topic"] for s in path["steps"]] == ["Derivatives", "Gradient Descent"]
    assert [s["role"] for s in path["steps"]] == ["foundation", "root cause"]
    assert path["unverified_foundations"] == ["Loss Functions"]
    assert path["retest_topic"] == "Learning Rate"
    assert all(len(s["sources"]) == 2 for s in path["steps"]), "top 2 source units per step"


def test_repair_path_when_prereqs_fine():
    m = learner_with(Derivatives=(0.2, 3), Algebra=(0.9, 3))
    path = repair_path(graph(), "Derivatives", "Gradient Descent", m, units)
    assert [s["topic"] for s in path["steps"]] == ["Derivatives"]
