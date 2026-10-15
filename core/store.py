"""Persistent application state: the course (knowledge base + topic graph), students, answer keys.

Kept apart from server.py so scripts and tests can reuse it. All state lives under `data/`
(git-ignored): kb.json, topics.json, students/<id>.json, quiz_keys.json, probes.json.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from .ingest import KnowledgeBase
from .learner import LearnerModel
from .retrieve import Retriever, assign_units, build_topics, load_seed_topics, units_for_topic
from .rootcause import build_graph

ROOT = Path(__file__).resolve().parent.parent
LOCK = threading.RLock()


def _read(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


class Course:
    """Knowledge base + topic map + prerequisite graph for one course."""

    def __init__(self, data_dir: Path):
        self.dir = data_dir
        self.kb_path = data_dir / "kb.json"
        self.topics_path = data_dir / "topics.json"
        self.uploads = data_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.kb = KnowledgeBase.load(self.kb_path) if self.kb_path.exists() else KnowledgeBase()
        self.retriever = Retriever(self.kb)
        meta = _read(self.topics_path, {})
        self.topics: list[dict] = meta.get("topics", [])
        self.mode: str = meta.get("mode", "")
        self.graph = build_graph(self.topics)

    # ------------------------------------------------------------ mutation
    def rebuild(self) -> None:
        self.retriever = Retriever(self.kb)
        self.graph = build_graph(self.topics)

    def add_file(self, path: Path) -> dict:
        with LOCK:
            if path.resolve().parent != self.uploads.resolve():
                shutil.copy2(path, self.uploads / path.name)
                path = self.uploads / path.name
            before = len(self.kb.units)
            added = self.kb.add_file(path, self.dir / "figures")
            self.kb.save(self.kb_path)
            self.rebuild()
            if self.topics:  # keep new material tagged against the existing map
                assign_units(self.kb, self.topics)
                self.save_topics()
            return {"units_added": added, "figure_units": sum(
                1 for u in self.kb.units[before:] if u.kind == "figure")}

    def set_topics(self, out: dict) -> None:
        self.topics = out.get("topics", [])
        self.mode = out.get("mode", "")
        self.save_topics()
        self.rebuild()

    def save_topics(self) -> None:
        self.kb.save(self.kb_path)
        _write(self.topics_path, {"topics": self.topics, "mode": self.mode})

    def analyze(self, prefer_seed: bool = False) -> dict:
        """Topic map: curated map for the bundled sample course, LLM/heuristic otherwise."""
        with LOCK:
            seed = next(
                (s for m in sorted((ROOT / "data").glob("sample_courses/*/topic_map.json"))
                 if (s := load_seed_topics(self.kb, m))),
                None,
            )
            from . import llm

            if prefer_seed and seed or (seed and not llm.available()):
                out = seed
            else:
                out = build_topics(self.kb)
            self.set_topics(out)
            return out

    def clear(self) -> None:
        with LOCK:
            self.kb = KnowledgeBase()
            self.topics, self.mode = [], ""
            for p in (self.kb_path, self.topics_path):
                p.unlink(missing_ok=True)
            self.rebuild()

    # --------------------------------------------------------------- views
    def topic_names(self) -> list[str]:
        return [t["name"] for t in self.topics] or sorted({x for u in self.kb.units for x in (u.topics or [])})

    def topic_terms(self) -> dict[str, set[str]]:
        """Content words describing each topic (name + concepts): used to keep offline questions on-topic."""
        from .offline import content_words

        return {t["name"]: set(content_words(" ".join([t["name"]] + t.get("key_concepts", []) + t.get("subtopics", []))))
                for t in self.topics}

    def sources(self, topic: str, k: int = 2) -> list[dict]:
        return units_for_topic(self.kb, self.retriever, self.topics, topic, k)

    def unit(self, unit_id: str):
        return next((u for u in self.kb.units if u.id == unit_id), None)

    def graph_view(self, mastery: dict[str, float] | None = None, attempts: dict[str, int] | None = None) -> dict:
        mastery, attempts = mastery or {}, attempts or {}
        nodes = [
            {
                "id": t["name"],
                "mastery": mastery.get(t["name"]),
                "attempts": attempts.get(t["name"], 0),
                "units": len(t.get("unit_ids", [])),
                "concepts": t.get("key_concepts", [])[:5],
            }
            for t in self.topics
        ]
        edges = [{"source": p, "target": t} for t, ps in self.graph.items() for p in ps]
        return {"nodes": nodes, "edges": edges}


class Student:
    def __init__(self, sid: str, base: Path):
        self.id = sid
        self.dir = base
        self.meta_path = base / f"{sid}.meta.json"
        self.hist_path = base / f"{sid}.history.json"
        self.learner = LearnerModel(base / f"{sid}.json")
        self.meta = _read(self.meta_path, {"id": sid, "name": sid, "exam_days": 7, "created": int(time.time())})
        hist = _read(self.hist_path, {})
        self.stems: list[str] = hist.get("stems", [])
        self.hashes: set[str] = set(hist.get("hashes", []))
        self.used_units: dict[str, int] = hist.get("used_units", {})
        self.results: list[dict] = hist.get("results", [])  # graded answers, newest last

    def save_meta(self) -> None:
        _write(self.meta_path, self.meta)

    def save_history(self) -> None:
        _write(
            self.hist_path,
            {
                "stems": self.stems[-400:],
                "hashes": sorted(self.hashes)[-800:],
                "used_units": self.used_units,
                "results": self.results[-400:],
            },
        )

    def remember(self, questions: list[dict]) -> None:
        for q in questions:
            if q.get("_hash"):
                self.hashes.add(q["_hash"])
            if q.get("_stem"):
                self.stems.append(q["_stem"])
            uid = q.get("source_unit_id")
            if uid:
                self.used_units[uid] = self.used_units.get(uid, 0) + 1
        self.save_history()

    def reset(self) -> None:
        self.learner.reset()
        self.stems, self.hashes, self.used_units, self.results = [], set(), {}, []
        self.save_history()


class Students:
    def __init__(self, base: Path):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Student] = {}

    @staticmethod
    def clean(sid: str | None) -> str:
        sid = re.sub(r"[^a-zA-Z0-9_-]", "", sid or "") or "default"
        return sid[:40]

    def get(self, sid: str | None) -> Student:
        sid = self.clean(sid)
        with LOCK:
            if sid not in self._cache:
                self._cache[sid] = Student(sid, self.base)
                if not self._cache[sid].meta_path.exists():
                    self._cache[sid].save_meta()
            return self._cache[sid]

    def list(self) -> list[dict]:
        ids = {p.name.split(".")[0] for p in self.base.glob("*.meta.json")} | {"default"}
        return [{"id": i, **{k: v for k, v in self.get(i).meta.items() if k != "id"}} for i in sorted(ids)]

    def create(self, name: str) -> Student:
        sid = self.clean(re.sub(r"\s+", "-", name.strip().lower())) or f"s{uuid.uuid4().hex[:6]}"
        st = self.get(sid)
        st.meta["name"] = name.strip() or sid
        st.save_meta()
        return st


class KeyStore:
    """Answer keys + probe records survive restarts and accumulate across quizzes."""

    def __init__(self, base: Path):
        self.keys_path = base / "quiz_keys.json"
        self.probes_path = base / "probes.json"
        self.keys: dict[str, dict] = _read(self.keys_path, {})
        self.probes: dict[str, dict] = _read(self.probes_path, {})

    def put_questions(self, questions: list[dict], student: str, quiz_id: str, purpose: str) -> list[dict]:
        """Assign globally unique ids, store private keys, return the questions (still private)."""
        with LOCK:
            for q in questions:
                q["id"] = f"q{uuid.uuid4().hex[:8]}"
                self.keys[q["id"]] = {
                    "answer": q.get("answer"),
                    "topic": q.get("topic", "general"),
                    "type": q.get("type", "mcq"),
                    "question": q.get("question", ""),
                    "options": q.get("options"),
                    "explanation": q.get("explanation", ""),
                    "misconception": q.get("misconception", ""),
                    "source_unit_id": q.get("source_unit_id", ""),
                    "difficulty": q.get("difficulty", ""),
                    "student": student,
                    "quiz_id": quiz_id,
                    "purpose": purpose,
                    "t": int(time.time()),
                }
            if len(self.keys) > 3000:
                for k in sorted(self.keys, key=lambda k: self.keys[k]["t"])[:500]:
                    del self.keys[k]
            _write(self.keys_path, self.keys)
        return questions

    def key(self, qid: str) -> dict | None:
        return self.keys.get(qid)

    def put_probe(self, probe: dict) -> str:
        with LOCK:
            pid = f"p{uuid.uuid4().hex[:8]}"
            self.probes[pid] = dict(probe, t=int(time.time()))
            _write(self.probes_path, self.probes)
            return pid

    def probe(self, pid: str) -> dict | None:
        return self.probes.get(pid)

    def close_probe(self, pid: str, result: dict) -> None:
        with LOCK:
            if pid in self.probes:
                self.probes[pid]["done"] = True
                self.probes[pid]["result"] = result
                _write(self.probes_path, self.probes)
