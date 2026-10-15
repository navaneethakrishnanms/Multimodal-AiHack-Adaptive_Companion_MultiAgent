# Understanding the codebase (Phase 1)

Judged from the code, not from the original README. Status key: **done** / **partial** / **missing**.
The Track D brief was supplied in the task (there is no `docs/TrackD.pdf` in the repo).

## Requirement map

| Req | Where | Status | What the code actually does |
|---|---|---|---|
| 1a ingest lecture video / textbook / slides | `core/ingest.py` `ingest_pdf/pptx/transcript` | **partial** | PDF, PPTX, MD/TXT, VTT/SRT work. There is **no audio/video ingestion** — a "lecture video" only works if the user already has a transcript file. |
| 1b topics, concepts, prerequisites, origin links | `core/retrieve.py::build_topics`, `Unit` | **partial** | Every `Unit` keeps `source_file` + `location` (page / slide / timestamp). Prerequisites are free-text names from the LLM, never validated against real topic names, no cycle check, no graph object. |
| 1c tag every unit to topics | `build_topics` | **partial** | Tags come from LLM batches of 12. Units the LLM forgets stay untagged (`topics=[]`) silently. Names drift between batches ("Gradient Descent" / "Gradient descent"), so one concept becomes several topics (visible in the old `eval_report.json` mastery keys). |
| 1d images / diagrams / figures | `Unit.image_count` | **missing** | Only an integer count of images per page/slide. No pixels extracted, no captioning, no figure units. |
| 2a cited answers that open the exact source | `core/tutor.py`, `/api/source` | **partial** | Answers cite `[unit_id]`; citations carry file + location + 300-char excerpt. UI shows text only: it cannot open a page image, slide image or seek to a timestamp. |
| 2b decline off-material queries | `tutor.answer` | **done (weak)** | Refusal = retrieval returns nothing (TF-IDF cosine > 0.01, a very low bar) or the LLM emits `NOT_COVERED`. No explicit "outside knowledge" separation. |
| 3a quizzes with tags | `core/quiz.py::generate` | **partial** | MCQ/short/numerical tagged with topic, source unit, difficulty. `_pick_units` always takes the first units of a topic, so quizzes keep sampling the same passages. Difficulty is the LLM's own claim, not calibrated. |
| 3b verification + no repeats | `quiz.generate` verify loop, `_stem_hash` | **partial** | Second LLM call re-solves each question (same model unless configured otherwise). Novelty = exact normalised-stem hash only, global (not per learner); paraphrases slip through. |
| 3c cited feedback + report | `/api/grade`, `LearnerModel.report` | **partial** | Feedback is the generated explanation. No source link in the grade response, **no post-assessment report**, the generated `misconception` field is never used. |
| 4a mastery updated after quizzes **and conversation** | `core/learner.py` | **partial** | Standard BKT per topic after each graded answer. Chat does not touch the model. One global learner (no student ids). |
| 4b new students | `learner.DEFAULT` | **missing** | Neutral prior 0.25 only. No diagnostic quiz / intake flow. |
| 5a standard eval framework | `core/evaluate.py` | **partial** | Native LLM-judge prompts (RAGAS-style names). No RAGAS/DeepEval, no context precision/recall. |
| 5b metrics on team test set | `tests/testset.json` | **partial** | 10 in-material + 4 off-material questions. Too small; context precision/recall missing. |
| 5c personalization evaluation | `eval_personalization` | **weak** | Students answer correctly with a fixed probability **regardless of what they know**, so "mastery gain" only reflects the dice. `hash(pname)` is salted per process (not reproducible); `Path("/tmp/...")` breaks on Windows; no baseline policy to compare against. |
| 6a course flow map | — | **missing** | Prerequisites exist only as text. |
| 6b revision material | — | **missing** | |
| 6c study schedule | `LearnerModel.study_plan` | **partial** | Ebbinghaus-style priority and day buckets; no per-day time budget or exam-date model. |
| 6d Indian-language / mixed language | — | **missing** | |
| 6e audio tutoring | — | **missing** | |

## Data flow

```
upload (server.py /api/upload) ──► ingest_file ──► KnowledgeBase.units ──► data/kb.json
                                                        │
/api/analyze ──► build_topics (LLM, batches) ──► topics.json + unit.topics tags
                                                        │
/api/chat ──► Retriever.search (TF-IDF) ──► optional LLM rerank ──► tutor.answer (LLM) ──► text + citations
/api/quiz ──► quiz.generate (LLM gen + LLM verify + hash dedupe) ──► quiz_keys.json (overwritten each time)
/api/grade ──► MCQ string match | LLM grade ──► LearnerModel.update (BKT) ──► data/learner.json
/api/dashboard, /api/plan ──► LearnerModel.report / study_plan
```

## Weakest points (ordered by rubric impact)

1. **Personalization (20%)**: the BKT model is sound but nothing *uses* it to decide what to ask next, and nothing exploits the topic graph. Wrong answers are never diagnosed (is the student weak on T, or on something T depends on?). The simulated-student evaluation cannot show improvement.
2. **Multimodal (20%)**: no figure/diagram extraction, no audio/video ingestion, and the source viewer shows text only.
3. **Evaluation (15%)**: not a standard framework, no context precision/recall, tiny test set, and personalization numbers are dice rolls.
4. **Robustness**: with no LLM configured `eval_generation` and topic mapping raise; `quiz_keys.json` is overwritten by each quiz so earlier quizzes cannot be graded; topic-name drift fragments the learner model.
5. **UX (15%)**: four plain tabs, no source viewer, no graph, no skeleton loaders, dark only.
6. **Tests**: there is no test suite at all.

## Plan

Phase 2 adds `core/rootcause.py` (prerequisite-blame diagnosis) on top of a validated prerequisite graph. Phase 3 replaces the dice-roll students with latent-knowledge students and compares three policies. Phases 4–6 add multimodal ingestion, the UI and the demo. All LLM-dependent features get an offline fallback so the app and tests run without a key.
