"""Multimodal ingestion (figures, audio/video) and the Source Viewer endpoints."""
import io
from pathlib import Path

import pytest

from core import vision
from core.ingest import _windows, ingest_file, ingest_media, ingest_transcript

ROOT = Path(__file__).parent.parent
ML = ROOT / "data" / "sample_courses" / "ml"
PDF, PPTX, MP4 = ML / "model_selection_chapter.pdf", ML / "lecture_slides.pptx", ML / "lecture2_model_selection.mp4"


def test_figure_units_flagged_when_no_vision(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    units = ingest_file(PDF, tmp_path)
    figs = [u for u in units if u.kind == "figure"]
    assert [f.location for f in figs] == ["p.2 · figure 1", "p.3 · figure 1"]
    assert all(f.figure_path and (tmp_path / f.figure_path).exists() for f in figs)
    assert all("not captured" in f.text for f in figs), "no vision model -> explicit flag, never a guess"
    assert figs[0].page == 2 and "Figure 5.1" in figs[0].text, "nearby caption text is kept for retrieval"


def test_figure_described_by_vision_model(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "describe", lambda path, ctx="": ("A U-shaped validation curve with a minimum near degree 4.", "vision"))
    units = ingest_file(PPTX, tmp_path)
    fig = next(u for u in units if u.kind == "figure")
    assert fig.location == "slide 6 · figure 1" and fig.source_type == "pptx"
    assert "U-shaped validation curve" in fig.text and "not captured" not in fig.text


def test_tiny_images_are_ignored(tmp_path):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "red").save(buf, "PNG")
    assert vision.save_image(buf.getvalue(), "png", tmp_path) is None
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "red").save(buf, "PNG")
    assert vision.save_image(buf.getvalue(), "png", tmp_path).exists()


def test_transcript_unit_label_matches_its_first_cue():
    entries = [(0, "a"), (22, "b"), (48, "c"), (75, "d"), (100, "e"), (130, "f"), (160, "g")]
    units = _windows(entries, "x.vtt", "transcript")
    assert [(u.location, u.text) for u in units] == [("00:00:00", "a b c"), ("00:01:15", "d e f"), ("00:02:40", "g")]
    assert [u.start_sec for u in units] == [0, 75, 160]


def test_vtt_sample_units_start_at_their_cue():
    units = ingest_transcript(ML / "lecture_transcript.vtt")
    assert [u.location for u in units] == ["00:00:00", "00:01:15", "00:02:40", "00:03:45"]
    assert units[1].text.startswith("Reinforcement learning")
    assert units[3].text.startswith("Finally, overfitting")


def test_media_ingestion_with_stub_transcriber(tmp_path, monkeypatch):
    from core import ingest

    monkeypatch.setattr(ingest, "transcribe", lambda p: [(0.0, "Intro."), (31.0, "More."), (65.0, "End.")])
    clip = tmp_path / "lec.mp4"
    clip.write_bytes(b"x")
    units = ingest_media(clip)
    assert [u.location for u in units] == ["00:00:00", "00:00:31", "00:01:05"]
    assert all(u.source_type == "video" and u.source_file == "lec.mp4" for u in units)


def test_real_whisper_transcription_of_sample_video():
    pytest.importorskip("faster_whisper")
    units = ingest_media(MP4)
    text = " ".join(u.text for u in units).lower()
    assert len(units) >= 2 and units[0].start_sec == 0
    assert "validation" in text and "cross-validation" in text.replace("cross validation", "cross-validation")
    assert all(u.source_type == "video" for u in units)


@pytest.fixture()
def media_client(client):
    for f in (PDF, PPTX, MP4):
        with open(f, "rb") as fh:
            r = client.post("/api/upload", files={"file": (f.name, fh)})
            assert r.status_code == 200, r.text
    return client


def test_upload_reports_figures(client):
    with open(PDF, "rb") as fh:
        r = client.post("/api/upload", files={"file": (PDF.name, fh)}).json()
    assert r["figure_units"] == 2 and r["units_added"] >= 5


def test_source_view_pdf_page_and_figure(media_client):
    v = media_client.get("/api/source/view", params={"file": PDF.name, "loc": "p.2 · figure 1"}).json()
    assert v["page"] == 2 and v["figure_url"].startswith("/api/figure/") and v["figure_described"] is False
    img = media_client.get(v["image_url"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/png" and img.content[:4] == b"\x89PNG"
    assert media_client.get(v["figure_url"]).status_code == 200
    assert media_client.get(f"/api/preview/{PDF.name}", params={"page": 99}).status_code == 404


def test_source_view_slide_preview_renders_chart(media_client):
    v = media_client.get("/api/source/view", params={"file": PPTX.name, "loc": "slide 6"}).json()
    assert v["slide"] == 6 and v["image_kind"] == "slide-preview"
    img = media_client.get(v["image_url"])
    assert img.status_code == 200 and len(img.content) > 20_000, "chart picture is drawn into the preview"


def test_source_view_video_timestamp_and_range_seek(media_client):
    v = media_client.get("/api/source/view", params={"file": MP4.name, "loc": "00:00:33"}).json()
    assert v["start_sec"] == 33 and v["media_url"] == f"/api/media/{MP4.name}"
    r = media_client.get(v["media_url"], headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100, "player must be able to seek with Range requests"
    assert media_client.get("/api/media/..%2f..%2fserver.py").status_code in (400, 404)


def test_figure_question_is_flagged_not_hallucinated(media_client):
    media_client.post("/api/analyze")
    r = media_client.post("/api/chat", json={"question": "What does Figure 5.1 show about validation error?"}).json()
    assert not r["refused"]
    assert any("Figure" in c["excerpt"] or c["location"].endswith("figure 1") or c["location"] == "p.2" for c in r["citations"])
