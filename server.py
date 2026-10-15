"""Study Companion — FastAPI server.

Run:
    uvicorn server:app --port 7861        (or python server.py)
Configure the LLM in .env (see config.example.env).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from core.ingest import KnowledgeBase
from core.retrieve import Retriever, build_topics
from core.tutor import answer as tutor_answer
from core.quiz import generate as quiz_generate, to_assessment
from core.learner import LearnerModel

def _load_env() -> None:
    envf = Path(__file__).parent / ".env"
    if not envf.exists():
        return
    for line in envf.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

ROOT = Path(__file__).parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
DATA.mkdir(exist_ok=True)
UPLOADS.mkdir(exist_ok=True)

KB_PATH = DATA / "kb.json"
LEARNER_PATH = DATA / "learner.json"
TOPICS_PATH = DATA / "topics.json"
HISTORY_PATH = DATA / "quiz_history.json"

app = FastAPI(title="Study Companion", version="1.0")

# --------------------------------------------------------------- state
_kb: KnowledgeBase | None = None
_retriever: Retriever | None = None
_learner = LearnerModel(LEARNER_PATH)
_topics: list[dict] = []
_asked_hashes: set[str] = set()


def get_kb() -> KnowledgeBase:
    global _kb, _retriever
    if _kb is None:
        _kb = KnowledgeBase.load(KB_PATH) if KB_PATH.exists() else KnowledgeBase()
        _retriever = Retriever(_kb)
    return _kb


def get_retriever() -> Retriever:
    get_kb()
    assert _retriever is not None
    return _retriever


def load_topics() -> list[dict]:
    global _topics
    if not _topics and TOPICS_PATH.exists():
        _topics = json.loads(TOPICS_PATH.read_text(encoding="utf-8")).get("topics", [])
    return _topics


def load_history() -> None:
    if HISTORY_PATH.exists():
        try:
            _asked_hashes.update(json.loads(HISTORY_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass


load_topics()
load_history()

# --------------------------------------------------------------- API

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    name = Path(file.filename or "upload").name
    dest = UPLOADS / name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    kb = get_kb()
    try:
        added = kb.add_file(dest)
    except ValueError as e:
        raise HTTPException(400, str(e))
    kb.save(KB_PATH)
    global _retriever
    _retriever = Retriever(kb)
    return {"file": name, "units_added": added, "summary": kb.summary()}


@app.get("/api/kb")
def kb_info() -> dict:
    kb = get_kb()
    return {
        "summary": kb.summary(),
        "topics": load_topics(),
        "units_sample": [u.to_dict() for u in kb.units[:3]],
    }


@app.post("/api/analyze")
def analyze() -> dict:
    """Build the topic map (LLM)."""
    global _topics
    kb = get_kb()
    if not kb.units:
        raise HTTPException(400, "No material uploaded yet")
    out = build_topics(kb)
    _topics = out.get("topics", [])
    kb.save(KB_PATH)
    TOPICS_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"topics": _topics}


class ChatIn(BaseModel):
    question: str


@app.post("/api/chat")
def chat(inp: ChatIn) -> dict:
    r = tutor_answer(inp.question, get_retriever())
    return r


class QuizIn(BaseModel):
    topics: list[str] = []
    count: int = 5
    difficulty: str = "mixed"
    types: list[str] = ["mcq", "short"]


@app.post("/api/quiz")
def quiz(inp: QuizIn) -> dict:
    kb = get_kb()
    if not kb.units:
        raise HTTPException(400, "No material uploaded yet")
    qs = quiz_generate(
        kb, inp.topics, count=inp.count, difficulty=inp.difficulty,
        types=inp.types, avoid_hashes=_asked_hashes,
    )
    # store public version, remember hashes
    for q in qs:
        _asked_hashes.add(q.get("_hash", ""))
    HISTORY_PATH.write_text(json.dumps(sorted(_asked_hashes)), encoding="utf-8")
    public = to_assessment(qs)
    # answer keys stay server-side
    keys = {q["id"]: {"answer": q.get("answer"), "topic": q.get("topic", "general"),
                      "type": q.get("type", "mcq"), "explanation": q.get("explanation", ""),
                      "source_unit_id": q.get("source_unit_id", "")} for q in qs}
    (DATA / "quiz_keys.json").write_text(json.dumps(keys, ensure_ascii=False, indent=1), encoding="utf-8")
    # strip answers from what the client sees
    for q in public["questions"]:
        q.pop("answer", None)
        q.pop("explanation", None)
    return public


class GradeIn(BaseModel):
    question_id: str
    response: str


@app.post("/api/grade")
def grade(inp: GradeIn) -> dict:
    keys = json.loads((DATA / "quiz_keys.json").read_text(encoding="utf-8"))
    key = keys.get(inp.question_id)
    if not key:
        raise HTTPException(404, "Unknown question id")
    qtype = key.get("type", "mcq")
    correct_answer = str(key.get("answer", ""))
    response = inp.response.strip()
    if qtype == "mcq":
        correct = response == correct_answer or (
            len(response) == 1 and response in correct_answer
        )
    else:
        # LLM-assisted grading for short/numerical
        try:
            from core import llm
            v = llm.chat_json(
                [
                    {"role": "system", "content": "Grade the student answer. "
                     'Return JSON {"correct": true|false, "feedback": "one short sentence"}'},
                    {"role": "user", "content": f"Question type: {qtype}\nModel answer: {correct_answer}\n"
                     f"Student answer: {response}"},
                ],
                max_tokens=200,
            )
            correct = bool(v.get("correct"))
        except Exception:
            correct = response.lower() == correct_answer.lower()
    p = _learner.update(key.get("topic", "general"), correct, qtype)
    return {
        "correct": correct,
        "model_answer": correct_answer,
        "explanation": key.get("explanation", ""),
        "topic": key.get("topic", "general"),
        "mastery": p,
    }


@app.get("/api/dashboard")
def dashboard() -> dict:
    return {"report": _learner.report()}


class PlanIn(BaseModel):
    days: int = 7


@app.post("/api/plan")
def plan(inp: PlanIn) -> dict:
    topics = load_topics()
    all_names = [t["name"] for t in topics] if topics else sorted({t for u in get_kb().units for t in (u.topics or [])})
    return {"plan": _learner.study_plan(all_names, inp.days)}


@app.get("/api/source")
def source(file: str, loc: str) -> JSONResponse:
    """Resolve a citation to its exact location text (for the citation viewer)."""
    kb = get_kb()
    for u in kb.units:
        if u.source_file == file and u.location == loc:
            return JSONResponse(u.to_dict())
    raise HTTPException(404, "Unit not found")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 7861)))
