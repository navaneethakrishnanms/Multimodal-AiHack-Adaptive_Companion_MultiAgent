"""Browser smoke test: walks every screen of the UI, fails on JS errors, saves screenshots.

Usage (server must already be running):  python scripts/ui_smoke.py [--url http://127.0.0.1:7862] [--out shots]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:7862")
ap.add_argument("--out", default="shots")
ap.add_argument("--dark", action="store_true")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(exist_ok=True, parents=True)
errors: list[str] = []


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1366, "height": 860}, color_scheme="dark" if args.dark else "light")
        page = ctx.new_page()
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None)
        page.goto(args.url, wait_until="networkidle")

        def shot(name: str) -> None:
            page.wait_for_timeout(500)
            page.screenshot(path=str(OUT / f"{name}.png"))

        # multi-file upload regression: a live FileList used to be emptied after the first file
        ml = Path(__file__).resolve().parent.parent / "data" / "sample_courses" / "ml"
        page.set_input_files("#file", [str(ml / "ml_notes.md"), str(ml / "lecture_transcript.vtt"), str(ml / "model_selection_chapter.pdf")])
        page.wait_for_function("document.querySelectorAll('#upList .chip.ok').length >= 3", timeout=120000)
        shot("0_multi_upload")
        page.click("#btnSample")
        page.wait_for_function("document.querySelector('#analyzeStat').textContent.includes('✓')", timeout=240000)
        shot("1_library")

        page.click("nav button[data-tab=tutor]")
        page.fill("#q", "Why does a learning rate that is too large make gradient descent diverge?")
        page.click("#btnAsk")
        page.wait_for_selector("#chat .msg.bot .banner", timeout=60000)
        page.wait_for_selector("#svBody img.page, #svBody .excerpt", timeout=20000)
        shot("2_tutor_grounded")
        page.fill("#q", "What is the capital of France?")
        page.click("#btnAsk")
        page.wait_for_selector("#chat .msg.bot.warn", timeout=60000)
        shot("3_tutor_refusal")

        page.click("#btnDemo")
        page.wait_for_selector("#qTopics .chip.on", timeout=120000)
        page.click("#btnQuiz")
        page.wait_for_selector("#quizBox .q", timeout=120000)
        shot("4_practice")
        page.click("#quizBox .q [data-act=demo-pick][data-correct='0']")
        page.click("#quizBox .q [data-act=submit]")
        page.wait_for_selector("#quizBox .fb", timeout=30000)
        shot("5_feedback")

        page.click("nav button[data-tab=rootcause]")
        page.select_option("#rcTopic", "Regularization")
        page.click("#btnDiagnose")
        page.wait_for_selector("#rcProbe .q", timeout=120000)
        page.wait_for_timeout(800)
        shot("6_rootcause_probe")
        page.click("[data-act=demo-probe][data-correct='0']")
        page.click("[data-act=probe-submit]")
        page.wait_for_selector("#rcResult .step", timeout=120000)
        page.wait_for_timeout(600)
        shot("7_rootcause_repair")

        page.click("nav button[data-tab=progress]")
        page.wait_for_selector("#mastery .trow", timeout=30000)
        page.click("#btnPlan")
        page.wait_for_selector("#planBox .tile", timeout=30000)
        shot("8_progress")

        page.click("nav button[data-tab=evaluation]")
        page.wait_for_selector("#evChart svg", timeout=30000)
        shot("9_evaluation")

        page.set_viewport_size({"width": 390, "height": 800})
        page.click("nav button[data-tab=practice]")
        shot("10_mobile")
        b.close()
    for e in errors:
        print("ERROR", e)
    print("screens ok" if not errors else f"{len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
