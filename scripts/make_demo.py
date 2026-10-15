"""Record the demo video: scripted browser run + synthesized narration + captions -> demo/final.mp4.

    python scripts/make_demo.py              # starts its own server on a temp data dir (uses .env if the LLM works)
    python scripts/make_demo.py --offline    # force offline mode (no LLM calls)
    python scripts/make_demo.py --dry-run    # only write docs/DEMO_SCRIPT.md and narration audio

Needs: playwright (+ Edge or Chromium), edge-tts (falls back to gTTS, then Windows speech), imageio-ffmpeg, av.
Does NOT upload anywhere. Outputs: demo/final.mp4, demo/captions.srt, demo/chapters.txt, demo/run_report.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from demo_script import scenes, write_doc  # noqa: E402

OUT = ROOT / "demo"
PORT = int(os.environ.get("DEMO_PORT", 7870))
URL = f"http://127.0.0.1:{PORT}"
SAMPLE = ROOT / "data" / "sample_courses" / "ml"
FILES = ["ml_notes.md", "lecture_slides.pptx", "lecture_transcript.vtt", "model_selection_chapter.pdf", "lecture2_model_selection.mp4"]
VOICE = "en-US-AriaNeural"

CARD_CSS = """body{margin:0;height:100vh;display:grid;place-items:center;background:#f6f3ec;color:#1c2430;font-family:Georgia,'Times New Roman',serif}
.c{max-width:1250px;padding:40px}h1{font-size:64px;margin:0 0 18px;letter-spacing:-.02em}p{font:28px/1.5 system-ui,Segoe UI,sans-serif;color:#556070;margin:10px 0}
.b{display:inline-block;background:#0f6b5e;color:#fff;border-radius:12px;padding:6px 16px;font:600 20px system-ui;margin-bottom:22px}
ul{font:28px/1.7 system-ui;color:#1c2430;margin-top:22px} .row{display:flex;gap:14px;align-items:center;justify-content:center;flex-wrap:wrap}
.n{background:#fffdf8;border:2px solid #0f6b5e;border-radius:14px;padding:14px 20px;font:600 22px system-ui;text-align:center;min-width:210px}
.n small{display:block;font:400 16px system-ui;color:#556070;margin-top:4px}.a{font:700 34px system-ui;color:#0f6b5e}
.lane{margin:26px 0}.lane h3{font:700 18px system-ui;letter-spacing:.08em;text-transform:uppercase;color:#0f6b5e;margin:0 0 10px}"""

TITLE = f"""<style>{CARD_CSS}</style><div class=c><span class=b>Multimodal AI Hackathon 2026 · Track D</span><h1>Study Companion</h1>
<p>A source-grounded tutor that doesn’t just say <b>what</b> you got wrong — it finds <b>why</b>.</p>
<ul><li>One knowledge base from slides, textbooks and lecture video</li><li>Answers that cite the exact page, slide or timestamp — or decline</li><li>Verified quizzes, a learner model, and prerequisite root-cause diagnosis</li></ul></div>"""

ARCH = f"""<style>{CARD_CSS}</style><div class=c style="max-width:1700px"><h1 style="font-size:46px">Architecture</h1>
<div class=lane><h3>Knowledge base</h3><div class=row><div class=n>Upload<small>PDF · PPTX · MP4/MP3 · VTT · MD</small></div><span class=a>→</span>
<div class=n>Ingest<small>pages, slides, Whisper, figures + vision</small></div><span class=a>→</span><div class=n>Source-linked units<small>file + page/slide/timestamp</small></div><span class=a>→</span>
<div class=n>Topic map<small>concepts + prerequisite DAG</small></div></div></div>
<div class=lane><h3>Grounded tutor</h3><div class=row><div class=n>Question</div><span class=a>→</span><div class=n>TF-IDF retrieval<small>+ calibrated refusal gate</small></div><span class=a>→</span>
<div class=n>Tutor LLM<small>only retrieved excerpts, must cite</small></div><span class=a>→</span><div class=n>Source Viewer<small>page · slide · video time</small></div></div></div>
<div class=lane><h3>Assessment and learner model</h3><div class=row><div class=n>Quiz generator</div><span class=a>→</span><div class=n>Independent verifier<small>cross-model option</small></div><span class=a>→</span>
<div class=n>Student answer</div><span class=a>→</span><div class=n>BKT mastery<small>per topic, per student</small></div><span class=a>→</span><div class=n style="background:#dcefea">RootCause<small>blame → probe → repair → re-test</small></div></div></div>
<p>Every LLM feature has an offline fallback · every metric comes from <code>scripts/run_eval.py</code></p></div>"""

OUTRO = f"""<style>{CARD_CSS}</style><div class=c><h1>Study Companion</h1><p>Open source starting point and the LearnFlow AI project are credited in the README.</p>
<p>Code, docs and evaluation: see the repository.</p></div>"""

CURSOR_JS = """(() => { const d = document.createElement('div'); d.id='fakecursor';
d.style.cssText='position:fixed;z-index:99999;width:22px;height:22px;border-radius:50%;background:rgba(15,107,94,.35);border:3px solid #0f6b5e;pointer-events:none;transform:translate(-50%,-50%);left:-50px;top:-50px;transition:transform .08s';
addEventListener('DOMContentLoaded',()=>document.body.appendChild(d));
addEventListener('mousemove',e=>{d.style.left=e.clientX+'px';d.style.top=e.clientY+'px'},true);
addEventListener('mousedown',()=>d.style.transform='translate(-50%,-50%) scale(.6)',true);
addEventListener('mouseup',()=>d.style.transform='translate(-50%,-50%) scale(1)',true);})();"""


# ------------------------------------------------------------------ helpers

def log(*a) -> None:
    print("[demo]", *a, flush=True)


def http(path: str, body: dict | None = None, timeout: int = 600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start_server(data_dir: Path, offline: bool) -> subprocess.Popen:
    env = dict(os.environ, PORT=str(PORT), STUDY_DATA_DIR=str(data_dir), KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8")
    if offline:
        env["STUDY_NO_ENV"] = "1"
        for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "VERIFY_MODEL", "VISION_MODEL"):
            env.pop(k, None)
    proc = subprocess.Popen([sys.executable, str(ROOT / "server.py")], env=env, cwd=ROOT,
                            stdout=open(OUT / "server.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            http("/api/status")
            return proc
        except Exception:
            time.sleep(1)
    proc.kill()
    raise SystemExit("server did not start; see demo/server.log")


async def _edge(text: str, path: Path) -> None:
    import edge_tts

    await edge_tts.Communicate(text, VOICE, rate="+0%").save(str(path))


def tts(text: str, path: Path) -> Path:
    """edge-tts -> gTTS -> Windows speech. Returns the audio path actually written."""
    try:
        asyncio.run(_edge(text, path))
        return path
    except Exception as e:
        log("edge-tts failed:", str(e)[:80])
    try:
        from gtts import gTTS

        gTTS(text).save(str(path))
        return path
    except Exception as e:
        log("gTTS failed:", str(e)[:80])
    wav = path.with_suffix(".wav")
    clean = text.replace("'", "")
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.SetOutputToWaveFile('{wav}'); $s.Speak('{clean}'); $s.Dispose()"], check=True)
    return wav


def media_info(path: Path) -> dict:
    import av

    with av.open(str(path)) as c:
        v = next((s for s in c.streams if s.type == "video"), None)
        a = next((s for s in c.streams if s.type == "audio"), None)
        return {"duration": float(c.duration) / 1e6 if c.duration else 0.0, "has_audio": a is not None,
                "size": (v.codec_context.width, v.codec_context.height) if v else None}


def captions(sc: list[dict], starts: list[float], durs: list[float], out: Path) -> None:
    def ts(t: float) -> str:
        h, r = divmod(t, 3600)
        m, s = divmod(r, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int((s % 1) * 1000):03d}"

    lines, n = [], 1
    for s, t0, d in zip(sc, starts, durs):
        parts = re.split(r"(?<=[.!?])\s+", s["text"])
        chunks = []
        for p in parts:
            while len(p) > 95:
                cut = p.rfind(",", 0, 95)
                cut = cut if cut > 40 else p.rfind(" ", 0, 95)
                chunks.append(p[: cut + 1].strip())
                p = p[cut + 1:].strip()
            if p:
                chunks.append(p)
        total = sum(len(c) for c in chunks) or 1
        t = t0 + 0.35
        for c in chunks:
            span = d * len(c) / total
            lines.append(f"{n}\n{ts(t)} --> {ts(t + span - 0.05)}\n{c}\n")
            n += 1
            t += span
    out.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ scenes (browser)

class Recorder:
    def __init__(self, page):
        self.page = page

    def wait(self, ms: int) -> None:
        self.page.wait_for_timeout(ms)

    def glide(self, sel: str, click: bool = True, nth: int = 0) -> None:
        loc = self.page.locator(sel).nth(nth)
        loc.scroll_into_view_if_needed()
        box = loc.bounding_box()
        if box:
            self.page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=22)
            self.wait(250)
        if click:
            loc.click()

    def type(self, sel: str, text: str) -> None:
        self.glide(sel)
        self.page.keyboard.type(text, delay=22)
        self.wait(300)


def run_scenes(page, sc: list[dict], audio_dur: list[float], ctx: dict, t0: float) -> tuple[list[float], list[dict]]:
    r = Recorder(page)
    starts: list[float] = []
    status: list[dict] = []

    def card(html: str) -> None:
        page.goto("about:blank")
        page.set_content(html)

    def goto_app() -> None:
        if not page.url.startswith(URL):
            page.goto(URL, wait_until="networkidle")
            page.evaluate("document.documentElement.style.zoom = 1.18")

    def ask(q: str, banner: str) -> None:
        r.type("#q", q)
        r.glide("#btnAsk")
        page.wait_for_selector(f"#chat .msg.bot.{banner}", timeout=120000)
        r.wait(900)

    def s_intro():
        card(TITLE)

    def s_library():
        goto_app()
        r.wait(1200)
        page.set_input_files("#file", [str(SAMPLE / f) for f in FILES])
        page.wait_for_function(f"document.querySelectorAll('#upList .chip.ok').length >= {len(FILES)}", timeout=240000)
        r.wait(1500)
        r.glide("#btnAnalyze")
        page.wait_for_function("document.querySelector('#analyzeStat').textContent.includes('✓')", timeout=240000)
        r.wait(1500)
        page.mouse.wheel(0, 520)
        r.wait(2500)
        page.mouse.wheel(0, 520)
        r.wait(2500)

    def s_tutor():
        r.glide("nav.tabs button[data-tab=tutor]")
        r.wait(600)
        ask("Why does a learning rate that is too large make gradient descent diverge?", "ok")
        chips = page.locator("#chat .msg.bot:last-child .chip.src")
        slide = chips.filter(has_text="lecture_slides.pptx")
        (slide.first if slide.count() else chips.first).scroll_into_view_if_needed()
        (slide.first if slide.count() else chips.first).click()  # prefer the slide citation
        page.wait_for_selector("#svBody img.page", timeout=30000)
        r.wait(2500)
        ask("What does Figure 5.1 show about validation error?", "ok")
        pdf_chip = page.locator("#chat .msg.bot:last-child .chip.src", has_text="model_selection_chapter.pdf").first
        pdf_chip.scroll_into_view_if_needed()
        pdf_chip.click()
        page.wait_for_selector("#svBody img.page", timeout=30000)
        r.wait(3500)
        ask("What is k-fold cross-validation?", "ok")
        vid = page.locator("#chat .msg.bot:last-child .chip.src", has_text="lecture2_model_selection.mp4").first
        if vid.count():
            vid.click()
            page.wait_for_selector("#svMedia", timeout=30000)
            page.evaluate("const m=document.querySelector('#svMedia'); m.muted=true; m.play().catch(()=>{})")
        r.wait(4500)

    def s_refusal():
        ask("What is the capital of France?", "warn")
        r.wait(3500)

    def s_quiz():
        r.glide("nav.tabs button[data-tab=practice]")
        r.glide("#btnDemo")
        page.wait_for_selector("#qTopics .chip.on", timeout=180000)
        r.wait(1500)
        r.glide("#btnQuiz")
        page.wait_for_selector("#quizBox .q", timeout=180000)
        r.wait(1500)
        r.glide("#quizBox .q .chip.src", click=False)
        r.wait(2500)

    def s_rootcause():
        r.glide("#quizBox .q [data-act=demo-pick][data-correct='0']")
        r.wait(600)
        r.glide("#quizBox .q [data-act=submit]")
        page.wait_for_selector("#quizBox .fb", timeout=60000)
        r.wait(3000)
        r.glide("#quizBox [data-act=rc-from]")
        page.wait_for_selector("#rcProbe .q", timeout=180000)
        r.wait(6000)
        r.glide("[data-act=demo-probe][data-correct='0']")
        r.wait(900)
        r.glide("[data-act=probe-submit]")
        page.wait_for_selector("#rcResult .step", timeout=180000)
        r.wait(3500)
        page.mouse.wheel(0, 420)
        r.wait(1500)
        r.glide("#rcResult .step .chip.src", nth=0)
        r.wait(4500)

    def s_retest():
        r.glide("#rcResult .q [data-act=demo-pick][data-correct='1']", nth=0)
        r.wait(600)
        r.glide("#rcResult .q [data-act=submit]", nth=0)
        page.wait_for_selector("#rcResult .fb", timeout=60000)
        r.wait(4000)

    def s_progress():
        r.glide("nav.tabs button[data-tab=progress]")
        page.wait_for_selector("#mastery .trow", timeout=60000)
        r.wait(5000)
        r.glide("#btnPlan")
        page.wait_for_selector("#planBox .tile", timeout=60000)
        r.wait(2500)
        r.glide("#btnCards")
        page.wait_for_selector("#cardsBox .flip", timeout=120000)
        r.glide("#cardsBox .flip", nth=0)
        r.wait(3000)

    def s_evaluation():
        r.glide("nav.tabs button[data-tab=evaluation]")
        page.wait_for_selector("#evChart svg", timeout=60000)
        r.wait(6000)
        page.mouse.wheel(0, 700)
        r.wait(4000)
        page.select_option("#evMetric", "gain")
        r.wait(3500)

    def s_architecture():
        card(ARCH)

    def s_outro():
        card(OUTRO)

    actions = [s_intro, s_library, s_tutor, s_refusal, s_quiz, s_rootcause, s_retest, s_progress, s_evaluation, s_architecture, s_outro]
    for s, act, dur in zip(sc, actions, audio_dur):
        start = time.time()
        starts.append(start - t0)
        ok, err = True, ""
        try:
            act()
        except Exception as e:  # keep recording: a failed scene is reported, not fatal
            ok, err = False, f"{type(e).__name__}: {str(e)[:200]}"
            log("scene failed:", s["id"], err)
        elapsed = time.time() - start
        if elapsed < dur + 0.8:
            page.wait_for_timeout(int((dur + 0.8 - elapsed) * 1000))
        status.append({"scene": s["id"], "ok": ok, "error": err, "seconds": round(time.time() - start, 1), "narration_seconds": round(dur, 1)})
    return starts, status


# ------------------------------------------------------------------ main

def main() -> int:
    OUT.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="sc_demo_"))
    offline = "--offline" in sys.argv
    dry = "--dry-run" in sys.argv
    proc = start_server(work / "data", offline)
    try:
        st = http("/api/status")
        log("server mode:", st["mode"], st["llm"].get("model"))
        # prep: load the sample course + demo student to read the real blame numbers for the narration
        lib = http("/api/load-sample", {"course": "ml"})
        http("/api/demo-student", {})
        d = http("/api/rootcause/diagnose", {"topic": "Regularization", "student": "demo"})
        ctx = {"mode": st["mode"], "model": st["llm"].get("model"), "blame": d["blame"], "topics": lib["topics"],
               "edges": len(http("/api/graph")["edges"]), "files": len(FILES),
               "vision": st["mode"] == "online"}  # figures get described only when a vision-capable model is reachable
        sc = scenes(ctx)
        write_doc(ctx)

        # narration
        audio, durs = [], []
        for i, s in enumerate(sc):
            p = tts(s["text"], work / f"n{i:02d}.mp3")
            audio.append(p)
            durs.append(media_info(p)["duration"])
        log("narration seconds:", round(sum(durs), 1), "words:", sum(len(s["text"].split()) for s in sc))
        if dry:
            return 0

        # fresh server state for the recording: the script uploads the files itself
        proc.kill()
        proc = start_server(work / "rec_data", offline)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge")  # Edge plays H.264 sample video; bundled Chromium cannot
            except Exception:
                browser = p.chromium.launch()
            vid_dir = work / "video"
            ctx_b = browser.new_context(viewport={"width": 1920, "height": 1080}, record_video_dir=str(vid_dir),
                                        record_video_size={"width": 1920, "height": 1080})
            ctx_b.add_init_script(CURSOR_JS)
            page = ctx_b.new_page()
            t0 = time.time()
            starts, status = run_scenes(page, sc, durs, ctx, t0)
            video_path = Path(page.video.path())
            ctx_b.close()
            browser.close()

        # captions + audio mix + merge
        srt = work / "captions.srt"
        captions(sc, starts, durs, srt)
        import imageio_ffmpeg

        ff = imageio_ffmpeg.get_ffmpeg_exe()
        inputs = ["-i", str(video_path)]
        for a in audio:
            inputs += ["-i", str(a)]
        delays = ";".join(f"[{i + 1}:a]adelay={int((starts[i] + 0.4) * 1000)}|{int((starts[i] + 0.4) * 1000)}[a{i}]" for i in range(len(audio)))
        mix = "".join(f"[a{i}]" for i in range(len(audio))) + f"amix=inputs={len(audio)}:normalize=0[aout]"
        final = OUT / "final.mp4"
        cmd = [ff, "-y", "-loglevel", "error", *inputs, "-filter_complex",
               f"{delays};{mix};[0:v]scale=1920:1080,subtitles=captions.srt:force_style='FontName=Arial,FontSize=22,Outline=2,Shadow=0,MarginV=36,PrimaryColour=&H00FFFFFF&,OutlineColour=&H00302010&'[v]",
               "-map", "[v]", "-map", "[aout]", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(final)]
        subprocess.run(cmd, check=True, cwd=work)
        (OUT / "captions.srt").write_text(srt.read_text(encoding="utf-8"), encoding="utf-8")
        chapters = "\n".join(f"{int(t // 60)}:{int(t % 60):02d} {s['title']}" for s, t in zip(sc, starts))
        (OUT / "chapters.txt").write_text(chapters, encoding="utf-8")
        info = media_info(final)
        report = {"mode": ctx["mode"], "model": ctx["model"], "scenes": status, "final": info, "narration_seconds": round(sum(durs), 1)}
        (OUT / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        failed = [s["scene"] for s in status if not s["ok"]]
        log(f"final.mp4: {info['duration']:.1f}s, audio={info['has_audio']}, size={info['size']}, failed scenes={failed or 'none'}")
        ok = 180 <= info["duration"] <= 600 and info["has_audio"] and info["size"] == (1920, 1080)
        return 0 if ok and not failed else 2
    finally:
        proc.kill()


if __name__ == "__main__":
    sys.exit(main())
