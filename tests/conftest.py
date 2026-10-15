"""Shared fixtures: an isolated data dir and no LLM, so tests are offline and deterministic."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
SAMPLE = ROOT / "data" / "sample_course"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """TestClient on a fresh server module with an empty data dir and the LLM disabled."""
    from fastapi.testclient import TestClient

    monkeypatch.setenv("STUDY_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STUDY_NO_ENV", "1")
    for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "VERIFY_MODEL", "VISION_MODEL"):
        monkeypatch.delenv(k, raising=False)
    import server

    server = importlib.reload(server)
    return TestClient(server.app)


@pytest.fixture()
def loaded(client):
    """Client with the three-file sample course uploaded and the curated topic map applied."""
    for name in ("ml_notes.md", "lecture_slides.pptx", "lecture_transcript.vtt"):
        with open(SAMPLE / name, "rb") as f:
            assert client.post("/api/upload", files={"file": (name, f)}).status_code == 200
    assert client.post("/api/analyze").status_code == 200
    return client
