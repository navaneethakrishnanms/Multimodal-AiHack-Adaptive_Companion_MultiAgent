"""Upload the demo video to YouTube as unlisted (Studio automation).

Usage: python ytup.py <video.mp4> "<title>"
"""
import sys
import time

from playwright.sync_api import sync_playwright

VIDEO = sys.argv[1]
TITLE = sys.argv[2] if len(sys.argv) > 2 else "Study Companion — demo"
DESCRIPTION = """Study Companion — a source-grounded, adaptive AI tutor.
Multimodal AI Hackathon 2026, Track D: Personalized Tutoring & Adaptive Learning.

Turns your own course material (PDF, slides, notes, lecture transcripts) into a
source-cited knowledge base, then runs grounded tutoring with explicit refusals,
verified adaptive quizzes, and a Bayesian Knowledge Tracing learner model.

Code: https://github.com/maebahesioru/multimodal-study-companion"""

with sync_playwright() as p:
    browser = p.chromium.connect_over_cdp("http://127.0.0.1:9340")
    ctx = browser.contexts[0]
    pg = ctx.new_page()
    pg.on("dialog", lambda d: d.accept())
    pg.set_viewport_size({"width": 1440, "height": 900})

    pg.goto("https://studio.youtube.com", timeout=90000)
    pg.wait_for_timeout(15000)
    print("URL:", pg.url[:100], flush=True)

    # create -> upload
    pg.evaluate("""() => {
        const b = [...document.querySelectorAll('a, button, ytcp-button')]
            .find(x => /作成|Create/i.test(x.innerText || x.getAttribute('aria-label') || ''));
        if (b) b.click();
    }""")
    pg.wait_for_timeout(4000)
    # click the upload item
    pg.evaluate("""() => {
        const items = [...document.querySelectorAll('ytcp-button, tp-yt-paper-item, a, li')]
            .filter(x => /アップロード|Upload/i.test(x.innerText || ''));
        if (items.length) items[0].click();
    }""")
    pg.wait_for_timeout(8000)

    # file input
    fi = None
    for i in range(20):
        fi = pg.query_selector("input[type=file]")
        if fi:
            break
        pg.wait_for_timeout(1500)
    if not fi:
        print("NO FILE INPUT FOUND", flush=True)
        sys.exit(1)
    fi.set_input_files(VIDEO)
    print("file set", flush=True)
    pg.wait_for_timeout(20000)  # upload + processing begins

    # fill the details form (title/description) — Studio uses contenteditable
    try:
        pg.evaluate(
            """([title, desc]) => {
                const boxes = [...document.querySelectorAll('#textbox[contenteditable], ytcp-social-suggestions-textbox #textbox')];
                if (boxes[0]) { boxes[0].focus(); document.execCommand('selectAll'); document.execCommand('insertText', false, title); }
            }""",
            [TITLE],
        )
        print("title set", flush=True)
    except Exception as e:
        print("title err:", str(e)[:80], flush=True)
    pg.wait_for_timeout(2000)
    # scroll + description
    pg.evaluate(
        """(desc) => {
            const boxes = [...document.querySelectorAll('#textbox[contenteditable]')];
            const d = boxes.find(b => (b.id || '') === 'textbox' && boxes.indexOf(b) > 0) || boxes[1];
            if (d) { d.focus(); document.execCommand('selectAll'); document.execCommand('insertText', false, desc); }
        }""",
        DESCRIPTION,
    )
    print("desc set", flush=True)
    pg.wait_for_timeout(3000)

    # 3 steps: 詳細 -> 公開範囲 -> 確認 (click 次へ x3)
    for step in range(3):
        pg.evaluate("""() => {
            const b = [...document.querySelectorAll('ytcp-button')]
                .find(x => /次へ|Next/i.test(x.innerText || ''));
            if (b) b.click();
        }""")
        pg.wait_for_timeout(6000)
        print(f"next {step+1}", flush=True)

    # visibility = unlisted (限定公開)
    pg.evaluate("""() => {
        const radios = [...document.querySelectorAll('tp-yt-paper-radio-button, ytcp-radio-button')];
        const unlisted = radios.find(r => /限定公開|Unlisted/i.test(r.innerText || ''));
        if (unlisted) unlisted.click();
    }""")
    print("unlisted selected", flush=True)
    pg.wait_for_timeout(3000)

    # final save/publish
    pg.evaluate("""() => {
        const b = [...document.querySelectorAll('ytcp-button')]
            .find(x => /保存|公開|Save|Publish/i.test(x.innerText || ''));
        if (b) b.click();
    }""")
    print("save clicked", flush=True)
    pg.wait_for_timeout(20000)

    # find the video URL
    url = pg.evaluate("""() => {
        const a = [...document.querySelectorAll('a')]
            .find(x => /youtu\\.be|watch\\?v=/.test(x.getAttribute('href') || ''));
        return a ? a.getAttribute('href') : null;
    }""")
    print("VIDEO URL:", url, flush=True)
    time.sleep(5)
