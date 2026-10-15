"""Record the Study Companion demo video (screen capture of the live app).

Prereq: server running on :7861, sample material ingested + topics built.
Produces: demo_raw.webm (via Playwright video capture)
"""
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent.parent / "demo"
OUT.mkdir(exist_ok=True)
VID = OUT / "video"
VID.mkdir(exist_ok=True)

APP = "http://127.0.0.1:7861"


def slow(page, ms=600):
    page.wait_for_timeout(ms)


with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9340")
    ctx = browser.new_context(
        viewport={"width": 1280, "height": 800},
        record_video_dir=str(VID),
        record_video_size={"width": 1280, "height": 800},
    )
    page = ctx.new_page()
    page.on("dialog", lambda d: d.accept())

    # ---------------- 1. landing + material tab
    page.goto(APP, timeout=60000)
    slow(page, 3500)

    # ---------------- 2. upload the three files (visible flow)
    # NOTE: do NOT click #drop — it opens the native file picker and hangs automation.
    page.set_input_files(
        "#file",
        [
            str(OUT.parent / "data/sample_course/ml_notes.md"),
            str(OUT.parent / "data/sample_course/lecture_slides.pptx"),
            str(OUT.parent / "data/sample_course/lecture_transcript.vtt"),
        ],
    )
    slow(page, 8000)

    # ---------------- 3. analyze -> topic map
    page.click("#btnAnalyze")
    slow(page, 100000)  # LLM topic mapping (2 batches, ~70-90s)
    page.evaluate("window.scrollTo(0, 220)")
    slow(page, 4500)

    # ---------------- 4. tutor: grounded question
    page.click("nav button[data-t='tutor']")
    slow(page, 1800)
    page.fill("#q", "Why is the learning rate important in gradient descent?")
    slow(page, 1500)
    page.click("#btnAsk")
    slow(page, 22000)
    # click the first citation to open the excerpt
    try:
        page.locator("#chat .cite a").first.click(timeout=5000)
        slow(page, 5000)
    except Exception:
        pass

    # ---------------- 5. tutor: off-material refusal
    page.fill("#q", "What is the capital of France?")
    slow(page, 1200)
    page.click("#btnAsk")
    slow(page, 18000)

    # ---------------- 6. quiz generation
    page.click("nav button[data-t='quiz']")
    slow(page, 1800)
    # select the Gradient Descent topic chip
    try:
        page.locator("#qTopics .chip", has_text="Gradient Descent").first.click(timeout=5000)
    except Exception:
        pass
    slow(page, 1200)
    page.click("#btnQuiz")
    slow(page, 70000)  # generation + verification
    page.evaluate("window.scrollTo(0, 300)")
    slow(page, 3000)

    # ---------------- 7. answer one question
    try:
        # pick the first MCQ option in the first question card
        card = page.locator(".q").first
        card.locator(".opt").first.click(timeout=5000)
        slow(page, 1500)
        card.locator(".btnSubmit").click(timeout=5000)
        slow(page, 12000)
    except Exception:
        pass
    page.evaluate("window.scrollTo(0, 500)")
    slow(page, 3500)

    # ---------------- 8. dashboard
    page.click("nav button[data-t='dash']")
    slow(page, 4000)
    page.click("#btnPlan")
    slow(page, 6000)

    # ---------------- 9. back to top, finish
    page.evaluate("window.scrollTo(0, 0)")
    slow(page, 4000)

    video = page.video
    ctx.close()  # finalizes the video file
    path = video.path()
    print("VIDEO:", path)
