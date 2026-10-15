"""End-to-end API tests with no LLM: ingestion, topic graph, tutor refusal, quiz, grading, RootCause."""


def test_status_reports_offline(client):
    s = client.get("/api/status").json()
    assert s["mode"] == "offline" and s["llm"]["configured"] is False


def test_ingest_and_curated_graph(loaded):
    kb = loaded.get("/api/kb").json()
    assert kb["summary"]["total_units"] >= 13
    g = loaded.get("/api/graph").json()
    names = {n["id"] for n in g["nodes"]}
    assert {"Gradient Descent", "Learning Rate", "Overfitting", "Regularization"} <= names
    assert {"source": "Overfitting", "target": "Regularization"} in g["edges"]
    # every unit is tagged to at least one topic
    topics = loaded.get("/api/kb").json()["topics"]
    assert sum(len(t["unit_ids"]) for t in topics) >= kb["summary"]["total_units"]


def test_tutor_answers_with_citation_and_refuses_off_material(loaded):
    r = loaded.post("/api/chat", json={"question": "Why does a learning rate that is too large make gradient descent diverge?"}).json()
    assert not r["refused"] and r["grounding"] == "source-backed" and r["citations"]
    assert r["mode"] == "offline-extractive"
    off = loaded.post("/api/chat", json={"question": "What is the capital of France?"}).json()
    assert off["refused"] and off["grounding"] == "not-covered" and off["citations"] == []


def test_quiz_questions_are_tagged_verified_and_not_repeated(loaded):
    first = loaded.post("/api/quiz", json={"topics": ["Gradient Descent"], "count": 3, "student": "t1"}).json()
    assert first["count"] >= 2
    for q in first["questions"]:
        assert q["topic"] and q["difficulty"] and q["source"]["location"]
        assert q["verified"] and "answer" not in q and "explanation" not in q
    stems = {q["question"] for q in first["questions"]}
    second = loaded.post("/api/quiz", json={"topics": ["Gradient Descent"], "count": 3, "student": "t1"}).json()
    assert not stems & {q["question"] for q in second["questions"]}, "same student must not see repeated stems"


def test_grade_updates_mastery_and_cites_source(loaded):
    q = loaded.post("/api/quiz", json={"topics": ["Learning Rate"], "count": 2, "types": ["mcq"], "student": "t2"}).json()["questions"][0]
    pick = loaded.post("/api/grade", json={"question_id": q["id"], "response": "definitely-wrong", "student": "t2"}).json()
    assert pick["correct"] is False and pick["mastery"] < pick["mastery_before"]
    assert pick["source"]["location"], "feedback must cite the exact source"
    dash = loaded.get("/api/dashboard", params={"student": "t2"}).json()["report"]
    assert q["topic"] in dash["topics"]


def test_students_are_isolated(loaded):
    q = loaded.post("/api/quiz", json={"topics": ["Learning Rate"], "count": 1, "types": ["mcq"], "student": "a"}).json()["questions"][0]
    loaded.post("/api/grade", json={"question_id": q["id"], "response": "nope", "student": "a"})
    assert loaded.get("/api/dashboard", params={"student": "b"}).json()["report"]["topics"] == {}


def test_rootcause_demo_flow(loaded):
    demo = loaded.post("/api/demo-student").json()
    assert demo["start_topic"] == "Regularization"
    before = loaded.get("/api/dashboard", params={"student": "demo"}).json()["report"]["topics"]
    assert "Overfitting" not in before, "the hidden gap must start untested"

    diag = loaded.post("/api/rootcause/diagnose", json={"topic": "Regularization", "student": "demo"}).json()
    assert diag["target"] == "Overfitting"
    assert diag["blame"][0]["topic"] == "Overfitting" and not diag["blame"][0]["probed"]
    assert abs(sum(b["blame"] for b in diag["blame"]) - 1) < 0.02
    assert len(diag["probe"]) == 2 and all(q["verified"] for q in diag["probe"])
    # blame alone must not have touched mastery
    mid = loaded.get("/api/dashboard", params={"student": "demo"}).json()["report"]["topics"]
    assert "Overfitting" not in mid

    answers = []
    for q in diag["probe"]:
        wrong = loaded.post("/api/demo/pick", json={"question_id": q["id"], "correct": False}).json()["response"]
        answers.append({"question_id": q["id"], "response": wrong})
    res = loaded.post("/api/rootcause/probe-result", json={"probe_id": diag["probe_id"], "responses": answers, "student": "demo"}).json()
    assert res["confirmed_weak"] is True and res["target"] == "Overfitting"
    steps = res["repair"]["steps"]
    assert steps[-1]["topic"] == "Overfitting" and len(steps[-1]["sources"]) == 2
    assert all(s["source_file"] and s["location"] for s in steps[-1]["sources"])
    assert res["retest"]["topic"] == "Regularization" and len(res["retest"]["questions"]) >= 1
    after = loaded.get("/api/dashboard", params={"student": "demo"}).json()["report"]["topics"]
    assert after["Overfitting"]["mastery"] < 0.6


def test_rootcause_clears_prerequisite_when_probe_correct(loaded):
    loaded.post("/api/demo-student")
    diag = loaded.post("/api/rootcause/diagnose", json={"topic": "Regularization", "student": "demo"}).json()
    answers = [
        {"question_id": q["id"], "response": loaded.post("/api/demo/pick", json={"question_id": q["id"], "correct": True}).json()["response"]}
        for q in diag["probe"]
    ]
    res = loaded.post("/api/rootcause/probe-result", json={"probe_id": diag["probe_id"], "responses": answers, "student": "demo"}).json()
    assert res["confirmed_weak"] is False and "repair" not in res


def test_diagnostic_intake_report_recommendations(loaded):
    d = loaded.post("/api/diagnostic", json={"student": "new", "max_questions": 5}).json()
    assert d["count"] >= 3 and len({q["topic"] for q in d["questions"]}) >= 3
    loaded.post("/api/intake", json={"student": "new", "exam_days": 10, "self_rating": {"Overfitting": 1, "Learning Rate": 5}})
    m = loaded.get("/api/dashboard", params={"student": "new"}).json()["report"]["topics"]
    assert m["Learning Rate"]["mastery"] > m["Overfitting"]["mastery"]
    recs = loaded.get("/api/recommendations", params={"student": "new"}).json()["actions"]
    assert recs and all(r["why"] for r in recs)
    for q in d["questions"][:3]:
        loaded.post("/api/grade", json={"question_id": q["id"], "response": "bad", "student": "new"})
    rep = loaded.get("/api/report", params={"student": "new"}).json()
    assert rep["answered"] >= 1 and "weak_topics" in rep
