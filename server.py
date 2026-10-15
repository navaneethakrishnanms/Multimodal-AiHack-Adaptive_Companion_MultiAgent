"""Study Companion — FastAPI server.

Run:
    uvicorn server:app --port 7861        (or python server.py)
Configure the LLM in .env (see config.example.env). Without a working LLM every feature
degrades to a deterministic offline mode and the UI says so.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from core import llm, preview
from core import recommend as recommend_mod, report as report_mod, revision
from core.quiz import generate as quiz_generate, grade as quiz_grade
from core.rootcause import PROBE_QUESTIONS, RootCause, topo_order
from core.store import KeyStore, Course, Students
from core.tutor import answer as tutor_answer

llm.load_env()

ROOT = Path(__file__).parent
DATA = Path(os.environ.get("STUDY_DATA_DIR") or ROOT / "data")
DATA.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Study Companion", version="2.0")

course = Course(DATA)
students = Students(DATA / "students")
keystore = KeyStore(DATA)
SAMPLE = ROOT / "data" / "sample_courses" / "ml"  # bundled course (read-only)


# ------------------------------------------------------------- helpers

def public_question(q: dict, quiz_id: str) -> dict:
    """What the browser may see: no answer, no explanation, but the source it must be checked against."""
    out = {k: v for k, v in q.items() if not k.startswith("_") and k not in ("answer", "explanation")}
    u = course.unit(q.get("source_unit_id", ""))
    out["quiz_id"] = quiz_id
    out["source"] = (
        {"unit_id": u.id, "source_file": u.source_file, "location": u.location, "source_type": u.source_type}
        if u else None
    )
    return out


def make_assessment(
    student_id: str | None,
    topics: list[str],
    count: int,
    *,
    difficulty: str = "mixed",
    types: list[str] | None = None,
    purpose: str = "quiz",
    verified_only: bool = False,
) -> tuple[list[dict], str]:
    """Generate -> verify -> store keys -> remember for novelty. Returns (public questions, quiz_id)."""
    st = students.get(student_id)
    qs = quiz_generate(
        course.kb, topics, count=count, difficulty=difficulty, types=types,
        avoid_hashes=st.hashes, avoid_stems=st.stems, used_units=st.used_units,
        known_topics=course.topic_names(), topic_terms=course.topic_terms(),
    )
    if verified_only:
        qs = [q for q in qs if q.get("verified")]
    qs = qs[:count]
    quiz_id = f"z{uuid.uuid4().hex[:8]}"
    keystore.put_questions(qs, st.id, quiz_id, purpose)
    st.remember(qs)
    return [public_question(q, quiz_id) for q in qs], quiz_id


def source_of(unit_id: str) -> dict | None:
    u = course.unit(unit_id)
    if not u:
        return None
    return {
        "unit_id": u.id, "source_file": u.source_file, "location": u.location,
        "source_type": u.source_type, "excerpt": u.text[:300], "page": u.page,
    }


def rootcause_for(st) -> RootCause:
    def make_probe(topic: str, n: int) -> list[dict]:
        qs, _ = make_assessment(st.id, [topic], n, types=["mcq"], purpose="probe", verified_only=True)
        return qs

    return RootCause(course.graph, st.learner, course.sources, make_probe)


# --------------------------------------------------------------- API

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict:
    s = llm.status()
    return {
        "llm": s,
        "mode": "online" if s["healthy"] else "offline",
        "units": len(course.kb.units),
        "topics": len(course.topics),
        "topic_mode": course.mode,
    }


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:  # sync: transcription/vision can take a while, keep the loop free
    name = Path(file.filename or "upload").name
    dest = course.uploads / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        info = course.add_file(dest)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"file": name, **info, "summary": course.kb.summary()}


@app.get("/api/kb")
def kb_info() -> dict:
    return {
        "summary": course.kb.summary(),
        "topics": course.topics,
        "topic_mode": course.mode,
        "units_sample": [u.to_dict() for u in course.kb.units[:3]],
    }


class AnalyzeIn(BaseModel):
    mode: str = "auto"  # auto: curated map for the bundled sample course, else LLM/heuristic | llm: never use the curated map


@app.post("/api/analyze")
def analyze(inp: AnalyzeIn | None = None) -> dict:
    """Build the topic map + prerequisite graph."""
    if not course.kb.units:
        raise HTTPException(400, "No material uploaded yet")
    out = course.analyze(prefer_seed=(inp or AnalyzeIn()).mode != "llm")
    return {"topics": course.topics, "mode": out.get("mode", ""), "edges": sum(len(v) for v in course.graph.values())}


@app.get("/api/graph")
def graph(student: str | None = None) -> dict:
    st = students.get(student)
    mastery = st.learner.mastery()
    attempts = {t: r["attempts"] for t, r in st.learner.data.items()}
    view = course.graph_view(mastery, attempts)
    view["order"] = topo_order(course.graph)
    return view


class ChatIn(BaseModel):
    question: str
    student: str | None = None
    allow_outside: bool = False
    language: str = "en"


@app.post("/api/chat")
def chat(inp: ChatIn) -> dict:
    st = students.get(inp.student)
    r = tutor_answer(
        inp.question, course.retriever, mastery_for=st.learner.get,
        allow_outside=inp.allow_outside, language=inp.language,
    )
    if r.get("topic") and not r["refused"]:
        st.learner.touch(r["topic"])  # exposure only; mastery moves on answers
    r["llm_mode"] = "online" if llm.available() else "offline"
    return r


class QuizIn(BaseModel):
    topics: list[str] = []
    count: int = 5
    difficulty: str = "mixed"
    types: list[str] = ["mcq", "short"]
    student: str | None = None


@app.post("/api/quiz")
def quiz(inp: QuizIn) -> dict:
    if not course.kb.units:
        raise HTTPException(400, "No material uploaded yet")
    qs, quiz_id = make_assessment(
        inp.student, inp.topics, max(1, min(inp.count, 15)), difficulty=inp.difficulty, types=inp.types
    )
    out = {"questions": qs, "count": len(qs), "quiz_id": quiz_id}
    out["mode"] = "online" if llm.available() else "offline"
    return out


class GradeIn(BaseModel):
    question_id: str
    response: str
    student: str | None = None


def apply_grade(st, qid: str, response: str, source: str = "quiz") -> dict:
    key = keystore.key(qid)
    if not key:
        raise HTTPException(404, "Unknown question id")
    g = quiz_grade(key, response)
    before = st.learner.get(key["topic"])
    after = st.learner.update(key["topic"], g["correct"], key["type"], source=source)
    note = ""
    if not g["correct"]:
        from core.report import misconception_note

        note = misconception_note(key, response)
        st.learner.record_misconception(key["topic"], note)
    st.results.append({
        "qid": qid, "topic": key["topic"], "correct": g["correct"], "t": int(__import__("time").time()),
        "quiz_id": key.get("quiz_id"), "difficulty": key.get("difficulty"), "misconception": note,
        "source": source,
    })
    st.save_history()
    return {
        "question_id": qid,
        "correct": g["correct"],
        "model_answer": key["answer"],
        "explanation": key["explanation"],
        "feedback": g["feedback"],
        "grading": g["mode"],
        "topic": key["topic"],
        "mastery": after,
        "mastery_before": before,
        "source": source_of(key.get("source_unit_id", "")),
        "misconception": note,
        "can_diagnose": (not g["correct"]) and bool(course.graph.get(key["topic"])),
    }


@app.post("/api/grade")
def grade(inp: GradeIn) -> dict:
    return apply_grade(students.get(inp.student), inp.question_id, inp.response)


@app.get("/api/dashboard")
def dashboard(student: str | None = None) -> dict:
    st = students.get(student)
    return {"report": st.learner.report(), "student": {"id": st.id, **st.meta}}


class PlanIn(BaseModel):
    days: int = 7
    student: str | None = None
    minutes_per_day: int = 45


@app.post("/api/plan")
def plan(inp: PlanIn) -> dict:
    st = students.get(inp.student)
    return {"plan": st.learner.study_plan(course.topic_names(), inp.days, inp.minutes_per_day)}


@app.get("/api/source")
def source(file: str, loc: str) -> JSONResponse:
    """Resolve a citation to its exact location text."""
    for u in course.kb.units:
        if u.source_file == file and u.location == loc:
            return JSONResponse(u.to_dict())
    raise HTTPException(404, "Unit not found")


# ------------------------------------------------- report / recommendations / revision

@app.get("/api/report")
def report(student: str | None = None, quiz_id: str | None = None) -> dict:
    st = students.get(student)
    return report_mod.assessment_report(st.learner, st.results, course.graph, course.sources, quiz_id)


@app.get("/api/recommendations")
def recommendations(student: str | None = None) -> dict:
    st = students.get(student)
    return {"actions": recommend_mod.recommend(st.learner, course.graph, course.topic_names(), course.sources)}


@app.get("/api/flashcards")
def flashcards(student: str | None = None, per_topic: int = 3, topics: str = "") -> dict:
    st = students.get(student)
    chosen = [t for t in topics.split("|") if t] or st.learner.weak_topics() or course.topic_names()[:3]
    return {"topics": chosen, "cards": revision.flashcards(course.kb, chosen[:6], max(1, min(per_topic, 6)))}


@app.get("/api/brief")
def brief(topic: str) -> dict:
    return revision.brief(course.kb, topic)


# ---------------------------------------------------------- source viewer

def _safe(name: str) -> str:
    return Path(name).name  # no path traversal


@app.get("/api/figure/{name}")
def figure(name: str) -> FileResponse:
    p = course.dir / "figures" / _safe(name)
    if not p.exists() or p.suffix == ".txt":
        raise HTTPException(404, "No such figure")
    return FileResponse(p)


@app.get("/api/media/{name}")
def media(name: str) -> FileResponse:
    p = course.uploads / _safe(name)
    if not p.exists():
        raise HTTPException(404, "No such file")
    return FileResponse(p)  # Range requests are handled by Starlette, so the player can seek


@app.get("/api/preview/{name}")
def preview_image(name: str, page: int | None = None, slide: int | None = None) -> FileResponse:
    p = course.uploads / _safe(name)
    if not p.exists():
        raise HTTPException(404, "No such file")
    try:
        if p.suffix.lower() == ".pdf" and page:
            return FileResponse(preview.pdf_page_png(p, page, DATA / "previews"))
        if p.suffix.lower() == ".pptx" and slide:
            return FileResponse(preview.slide_png(p, slide, DATA / "previews"))
    except ValueError as e:
        raise HTTPException(404, str(e))
    raise HTTPException(400, "Give ?page= for PDFs or ?slide= for slide decks")


@app.get("/api/source/view")
def source_view(file: str, loc: str) -> dict:
    """Everything the Source Viewer needs to open a citation at its exact page / slide / timestamp."""
    units = course.kb.units
    idx = next((i for i, u in enumerate(units) if u.source_file == file and u.location == loc), None)
    if idx is None:
        raise HTTPException(404, "Unit not found")
    u = units[idx]
    out: dict = {"unit": u.to_dict(), "type": u.source_type, "kind": u.kind, "text": u.text, "topics": u.topics or []}
    if u.source_type == "pdf":
        page = u.page or int(u.location.split(".")[1].split()[0])
        out.update(page=page, image_url=f"/api/preview/{file}?page={page}")
    elif u.source_type == "pptx":
        n = int(u.location.split()[1])
        out.update(slide=n, image_url=f"/api/preview/{file}?slide={n}", image_kind="slide-preview")
    elif u.source_type in ("video", "audio"):
        out.update(media_url=f"/api/media/{file}", start_sec=u.start_sec or 0)
    elif u.source_type == "transcript":
        out.update(start_sec=u.start_sec or 0)
    if u.figure_path:
        out["figure_url"] = f"/api/figure/{u.figure_path}"
        out["figure_described"] = "not captured" not in u.text
    same = [(i, x) for i, x in enumerate(units) if x.source_file == file]
    pos = next(k for k, (i, _x) in enumerate(same) if i == idx)
    out["prev"] = same[pos - 1][1].location if pos > 0 else None
    out["next"] = same[pos + 1][1].location if pos + 1 < len(same) else None
    return out


# ----------------------------------------------------------- students

class StudentIn(BaseModel):
    name: str


@app.get("/api/students")
def list_students() -> dict:
    return {"students": students.list()}


@app.post("/api/students")
def create_student(inp: StudentIn) -> dict:
    st = students.create(inp.name)
    return {"id": st.id, **st.meta}


class IntakeIn(BaseModel):
    student: str | None = None
    name: str | None = None
    exam_days: int = 7
    self_rating: dict[str, int] = {}  # topic -> 1..5


@app.post("/api/intake")
def intake(inp: IntakeIn) -> dict:
    """Cold start: self-ratings seed the prior of topics that have no answers yet."""
    st = students.get(inp.student)
    if inp.name:
        st.meta["name"] = inp.name
    st.meta["exam_days"] = max(1, min(inp.exam_days, 120))
    for topic, rating in inp.self_rating.items():
        r = max(1, min(int(rating), 5))
        st.learner.seed_prior(topic, 0.12 + 0.1 * (r - 1))
    st.save_meta()
    return {"student": {"id": st.id, **st.meta}, "mastery": st.learner.mastery()}


class DiagnosticIn(BaseModel):
    student: str | None = None
    max_questions: int = 8


@app.post("/api/diagnostic")
def diagnostic(inp: DiagnosticIn) -> dict:
    """Diagnostic quiz for a new student: one question per topic, foundations first."""
    if not course.topics:
        raise HTTPException(400, "Build the topic map first")
    order = topo_order(course.graph)
    n = max(1, min(inp.max_questions, len(order)))
    # spread across the graph: take evenly spaced topics along the foundation->advanced order
    picks = [order[round(i * (len(order) - 1) / max(n - 1, 1))] for i in range(n)] if n > 1 else order[:1]
    picks = list(dict.fromkeys(picks))
    qs: list[dict] = []
    quiz_id = ""
    for t in picks:
        got, quiz_id = make_assessment(inp.student, [t], 1, types=["mcq"], purpose="diagnostic")
        qs.extend(got)
    return {"questions": qs, "count": len(qs), "quiz_id": quiz_id, "topics": picks}


# ---------------------------------------------------------- root cause

class DiagnoseIn(BaseModel):
    topic: str
    student: str | None = None
    exclude: list[str] = []
    question_id: str | None = None


@app.post("/api/rootcause/diagnose")
def rootcause_diagnose(inp: DiagnoseIn) -> dict:
    st = students.get(inp.student)
    if inp.topic not in course.graph:
        raise HTTPException(404, f"Unknown topic: {inp.topic}")
    rc = rootcause_for(st)
    out = rc.diagnose(inp.topic, exclude=inp.exclude, n_questions=PROBE_QUESTIONS)
    if out["target"]:
        out["probe_id"] = keystore.put_probe({
            "student": st.id, "topic": inp.topic, "target": out["target"],
            "question_ids": [q["id"] for q in out["probe"]],
        })
        out["probe_verified"] = sum(1 for q in out["probe"] if q.get("verified"))
    else:
        out["self_review"] = course.sources(inp.topic, 2)
    return out


class ProbeAnswer(BaseModel):
    question_id: str
    response: str


class ProbeResultIn(BaseModel):
    probe_id: str
    responses: list[ProbeAnswer]
    student: str | None = None


@app.post("/api/rootcause/probe-result")
def rootcause_probe_result(inp: ProbeResultIn) -> dict:
    probe = keystore.probe(inp.probe_id)
    if not probe:
        raise HTTPException(404, "Unknown probe")
    st = students.get(inp.student or probe["student"])
    target, topic = probe["target"], probe["topic"]
    before = st.learner.get(target)
    graded = []
    for r in inp.responses:
        if r.question_id not in probe["question_ids"]:
            raise HTTPException(400, f"{r.question_id} is not part of this probe")
        # probe answers are the ONLY evidence that moves the blamed prerequisite's mastery
        graded.append(apply_grade(st, r.question_id, r.response, source="probe"))
    rc = rootcause_for(st)
    result = rc.conclude(topic, target)
    result["probe_id"] = inp.probe_id
    result["mastery_before"] = round(before, 3)
    result["answers"] = graded
    if result["confirmed_weak"]:
        for step in result["repair"]["steps"]:
            st.learner.touch(step["topic"])
        retest, quiz_id = make_assessment(st.id, [topic], 2, types=["mcq"], purpose="retest", verified_only=True)
        result["retest"] = {"topic": topic, "quiz_id": quiz_id, "questions": retest}
    keystore.close_probe(inp.probe_id, {k: result[k] for k in ("confirmed_weak", "target_mastery")})
    return result


# ------------------------------------------------------------- sample courses

SAMPLE_COURSES = {
    "ml": ("ml", ["ml_notes.md", "lecture_slides.pptx", "lecture_transcript.vtt",
                  "model_selection_chapter.pdf", "lecture2_model_selection.mp4"]),
    "os": ("os", ["os_notes.txt"]),
}


def load_sample(which: str = "ml") -> dict:
    """Ingest a bundled course (skipping files that cannot be read here) and apply its curated topic map."""
    if which not in SAMPLE_COURSES:
        raise HTTPException(404, "Unknown sample course")
    sub, files = SAMPLE_COURSES[which]
    folder = ROOT / "data" / "sample_courses" / sub
    course.clear()
    skipped = []
    for f in files:
        try:
            course.add_file(folder / f)
        except ValueError as e:
            skipped.append({"file": f, "reason": str(e)[:140]})
    course.analyze(prefer_seed=True)
    return {"summary": course.kb.summary(), "topics": len(course.topics), "skipped": skipped}


class SampleIn(BaseModel):
    course: str = "ml"


@app.post("/api/load-sample")
def load_sample_endpoint(inp: SampleIn) -> dict:
    return load_sample(inp.course)


@app.get("/api/library")
def library() -> dict:
    files: dict[str, dict] = {}
    figures = []
    for u in course.kb.units:
        f = files.setdefault(u.source_file, {"file": u.source_file, "type": u.source_type, "units": 0, "figures": 0})
        f["units"] += 1
        if u.kind == "figure":
            f["figures"] += 1
            figures.append({"file": u.source_file, "location": u.location, "url": f"/api/figure/{u.figure_path}",
                            "described": "not captured" not in u.text, "text": u.text[:200]})
    return {"files": list(files.values()), "figures": figures, "topics": course.topics, "topic_mode": course.mode,
            "edges": sum(len(v) for v in course.graph.values())}


@app.get("/api/evaluation")
def evaluation() -> JSONResponse:
    path = ROOT / "eval_report.json"
    if not path.exists():
        raise HTTPException(404, "Run `python scripts/run_eval.py` first")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------- demo student

@app.post("/api/demo-student")
def demo_student() -> dict:
    """Seed a student whose hidden gap is a prerequisite (Overfitting) of a topic they will miss."""
    import time as _t

    if not course.kb.units or "Overfitting" not in course.topic_names():
        load_sample("ml")
    course.analyze(prefer_seed=True)
    st = students.get("demo")
    st.reset()
    st.meta.update({"name": "Demo student", "exam_days": 7, "demo_gap": "Overfitting"})
    st.save_meta()
    solid = {
        "Learning from Experience": 3, "Supervised Learning": 3, "Linear Regression": 3,
        "Mean Squared Error": 3, "Gradient Descent": 3, "Learning Rate": 2,
    }
    for topic, n in solid.items():
        for _ in range(n):
            st.learner.update(topic, True, "mcq", source="seed")
    st.learner.update("Regularization", False, "mcq", source="seed")
    return {
        "student": {"id": st.id, **st.meta},
        "story": "Strong on the basics, but just missed a Regularization question. The real gap is hidden: Overfitting was never tested.",
        "start_topic": "Regularization",
        "mastery": st.learner.mastery(),
        "seeded_at": int(_t.time()),
    }


class DemoPickIn(BaseModel):
    question_id: str
    correct: bool


@app.post("/api/demo/pick")
def demo_pick(inp: DemoPickIn) -> dict:
    """Demo helper: choose an option for the scripted demo student (only for question keys of student 'demo')."""
    key = keystore.key(inp.question_id)
    if not key or key.get("student") != "demo":
        raise HTTPException(403, "Only available for the demo student")
    opts = key.get("options") or []
    if inp.correct or not opts:
        return {"response": key["answer"]}
    wrong = [o for o in opts if o != key["answer"]]
    return {"response": wrong[0] if wrong else ""}


# --------------------------------------------------------------- misc

@app.post("/api/reset-student")
def reset_student(student: str | None = None) -> dict:
    students.get(student).reset()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 7861)))
