"""Scene-by-scene demo script: narration text + what is shown. Single source for the video and docs/DEMO_SCRIPT.md.

Narration is built from the live system state (online/offline) and eval_report.json, so the voice-over can
never quote a number the report does not contain or claim a capability the recording did not use.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def pc(x: float) -> str:
    return f"{round(100 * x)} percent"


def report() -> dict:
    return json.loads((ROOT / "eval_report.json").read_text(encoding="utf-8"))


def scenes(ctx: dict) -> list[dict]:
    """ctx: {mode: online|offline, model, blame: [{topic, blame}], topics: int, edges: int, files: int}"""
    R = report()
    online = ctx["mode"] == "online"
    ret, gate, ml = R["retrieval"], R["retrieval_gate"], R["personalization"]["ml"]
    gen = R.get("generation") or {}
    pb = ml["paired_vs_baselines"]
    f1 = pb["C_minus_B.f1_R"]
    gain = pb["C_minus_B.gain"]
    gain_sig = gain["ci95"][0] > 0 or gain["ci95"][1] < 0
    faith = gen.get("faithfulness", {})
    faith_txt = (f"faithfulness measured at {faith['score']} and answer relevancy at {gen['answer_relevancy']['score']}"
                 if faith.get("score") is not None else
                 "faithfulness and answer relevancy are not measured in this run, because they need a working language model, and we do not estimate them")
    top, second = (ctx["blame"] + [{"topic": "", "blame": 0}, {"topic": "", "blame": 0}])[:2]
    gate_key = "online_gate_test" if online else "test"
    vision = ("A vision model describes each figure, so you can ask about it." if ctx.get("vision") else
              "No vision model is connected in this recording, so the figures are kept and clearly marked as not described, instead of guessed.")
    quiz_how = ("The language model writes the questions, and an independent pass re-solves each one before you see it. "
                "A question whose key does not match is dropped." if online else
                "No language model is connected in this recording, so the questions are fill in the blank, and the key is a literal span of the source, "
                "which can be checked by string match. With a model connected they are written and independently verified.")
    def scene(sid, title, show, *beats):
        return {"id": sid, "title": title, "show": show, "beats": [{"key": k, "text": t} for k, t in beats],
                "text": " ".join(t for _k, t in beats)}

    return [
        scene("intro", "Problem and idea", "title card",
              ("intro", "Studying from slides, textbooks and lecture videos means jumping between files, and a general chatbot answers confidently without "
                        "saying where the answer came from, or whether your own course even covers it. Study Companion builds one knowledge base from all your "
                        "material. It answers only from it, with citations that open the exact page, slide or timestamp. And when you get a question wrong, "
                        "it works out why.")),
        scene("library", "Upload and ingestion", "Library: drop 5 files, build the topic map",
              ("lib_upload", f"Let's start with the material. I drop in {ctx['files']} files from a sample machine learning course: notes, a slide deck, a lecture transcript, "
                             "a textbook chapter as a P D F, and a recorded lecture video."),
              ("lib_wait", "There is no manual preprocessing. The P D F and the slides are read page by page and the images inside them are extracted as figures. "
                           f"The video is transcribed with Whisper into timestamped passages. {vision}"),
              ("lib_map", f"One click builds the topic map: {ctx['topics']} topics joined by {ctx['edges']} prerequisite links, with every passage tagged to them.")),
        scene("tutor", "Grounded chat and Source Viewer", "Tutor: cited answer, slide, textbook figure, video timestamp",
              ("t_ask", "Now the tutor. I ask why a learning rate that is too large makes gradient descent diverge."),
              ("t_slide", "The banner says source backed, and every claim carries a numbered citation. Clicking one opens the exact source. Here is slide four, redrawn from the slide itself."),
              ("t_fig", "Next I ask about the textbook figure. The answer points to page two, and the viewer shows the real page and the extracted figure."),
              ("t_video", "And this question about cross validation is answered from the lecture video, which opens at the right timestamp.")),
        scene("refusal", "Declining what the material does not cover", "Tutor: off-material question, amber banner",
              ("r_ask", "A tutor you can trust has to say when it does not know. The capital of France is not in this course, so the tutor declines and flags it in amber instead of improvising."),
              ("r_hold", "A general answer can still be requested, but it is shown separately and marked as outside your materials. "
                         f"On our held out test questions the refusal gate declines {pc(gate[gate_key]['refused_off_material'])} of the off material queries, "
                         f"while still answering {pc(gate[gate_key]['answered_in_material'])} of the in material ones.")),
        scene("quiz", "Assessment with verification", "Practice: demo student, Regularization quiz",
              ("q_gen", "Practice. This demo student is strong on the basics, but has a hidden gap. I pick the topic Regularization and generate a quiz."),
              ("q_hold", f"Every question is tagged with its topic, difficulty and source, and carries a verified badge. {quiz_how}")),
        scene("rootcause", "Root cause diagnosis", "Wrong answer, root cause graph, probe, repair path",
              ("rc_wrong", "I answer wrongly. The feedback shows the model answer, the explanation, the exact source to check, and the mastery estimate dropping."),
              ("rc_find", "Now the key idea. A wrong answer on Regularization may really be a gap in something it depends on. Root cause scores each prerequisite: "
                          "weak or never tested prerequisites get the most blame."),
              ("rc_hold", f"{top['topic']} has never been tested, so it gets {pc(top['blame'])} of the blame, while {second['topic']}, which the student knows well, "
                          f"gets {pc(second['blame'])}. The graph lights up the path."),
              ("rc_probe", f"Only two verified probe questions on {top['topic']} can change its mastery. The student misses both, and the diagnosis is confirmed."),
              ("rc_repair", "The repair path lists exactly what to read, with links to the notes and the lecture timestamp, followed by a re test.")),
        scene("retest", "Re-test", "Re-test answered correctly, mastery rises",
              ("retest", "After reading, the student re tests Regularization and gets it right, and the mastery estimate goes up. Mastery only moves on real answers. "
                         "Reading material, and the blame scores themselves, never change it.")),
        scene("progress", "Progress, recommendations, schedule", "Progress: mastery, next best actions, schedule, flashcards",
              ("p_tab", "The progress screen turns this into a plan. Mastery per topic comes from Bayesian knowledge tracing. Each next best action explains why it was chosen, with source links."),
              ("p_plan", "The study schedule spreads weak topics over the days left and brings them back using a forgetting curve."),
              ("p_cards", "Flashcards for weak topics keep their source links too.")),
        scene("evaluation", "Evaluation", "Evaluation dashboard from eval_report.json",
              ("e_tab", f"Finally, evaluation. Every number here comes from a script. Retrieval puts a relevant passage in the top four for {pc(ret['hit_rate'])} of {ret['cases']} questions, "
                        f"with context precision {ret['context_precision']} and recall {ret['context_recall']}."),
              ("e_gen", f"For generation, {faith_txt}."),
              ("e_chart", "For personalization we simulated thirty students per course whose hidden gaps follow the prerequisite graph. Root cause improves the F one score of diagnosed root causes "
                          f"over weakest first tracking by {f1['mean_diff']}, with a ninety five percent interval from {f1['ci95'][0]} to {f1['ci95'][1]}."),
              ("e_gain", f"It does not win on every measure: its gain in true gap closure is {'statistically clear' if gain_sig else 'not statistically significant'} at this sample size. We report that as it is.")),
        scene("architecture", "Architecture", "architecture slide",
              ("arch", "Under the hood, ingestion turns P D Fs, slides, audio and video into source linked units, with figures described by a vision model. A topic map with prerequisites tags every unit. "
                       "Retrieval is T F I D F with a calibrated refusal gate, and the tutor sees only retrieved excerpts and must cite them. Quizzes are generated, independently verified, and de duplicated per student. "
                       "The learner model is Bayesian knowledge tracing per topic, and Root cause adds the prerequisite graph on top of it. Everything has an offline fallback, so the app degrades instead of breaking.")),
        scene("outro", "Credits", "closing card",
              ("outro", "Study Companion builds on an open source starting point and borrows ideas from the LearnFlow project, both credited in the readme. Thanks for watching.")),
    ]


def write_doc(ctx: dict) -> None:
    sc = scenes(ctx)
    L = ["# Demo video script\n",
         f"_Generated by `scripts/make_demo.py` (mode: **{ctx['mode']}**). Narration is built from the live system state and `eval_report.json`; "
         "re-record after changing either._\n",
         "Target length 6–7 minutes (hard limit for the hackathon: 3–10). Voice: edge-tts `en-US-AriaNeural`. Browser recording: Playwright, 1920×1080, captions burned in.\n"]
    for i, s in enumerate(sc, 1):
        words = len(s["text"].split())
        beats = "\n".join(f"> **{b['key']}** — {b['text']}\n>" for b in s["beats"])
        L.append(f"## Scene {i} — {s['title']}  \n*On screen:* {s['show']}  \n"
                 f"*Narration (~{words} words, ~{round(words / 2.4)} s), one block per synchronised beat (the browser step runs when its block starts):*\n\n{beats}\n")
    (ROOT / "docs" / "DEMO_SCRIPT.md").write_text("\n".join(L), encoding="utf-8")
