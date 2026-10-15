# Architecture

Study Companion is a Python/FastAPI service with a single-file browser UI. State lives under `data/` (git-ignored): one course (knowledge base + topic graph), per-student learner files, answer keys and probe records.

```
                        ┌────────────────────────── core/store.py ──────────────────────────┐
 upload ─► core/ingest ─┤ KnowledgeBase: units {id, text, source_file, location, page,       │
 PDF/PPTX/MP4/MP3/      │   start_sec, kind(text|figure), figure_path, topics[]}             │
 VTT/SRT/MD/TXT         │ Course: retriever (TF-IDF), topics[], prerequisite DAG             │
        │               └────────────────────────────────────────────────────────────────────┘
        ├─ figures: core/vision.py  (PyMuPDF / python-pptx images → vision LLM description, cached)
        └─ speech : faster-whisper  (GPU float16 → CPU int8 fallback, cached per file)

 question ─► Retriever.search ─► refusal gate ─► (LLM rerank) ─► core/tutor ─► answer + [unit_id] citations
                                      │ below gate                                    │
                                      └──► NOT_COVERED (+ optional flagged outside answer)
                                                                                      ▼
                                                                  Source Viewer: /api/source/view
                                                                  (page PNG · slide preview · figure · video#t=)

 scope ─► core/quiz.generate ─► structural checks ─► independent verify (VERIFY_MODEL optional) ─► novelty filter
              │                                                       │
              └─ offline: cloze questions with literal-span keys ◄────┘
 answer ─► core/quiz.grade ─► core/learner (BKT per student/topic) ─► report / recommend / plan / rootcause
```

## Modules
| Module | Responsibility |
|---|---|
| `core/ingest.py` | Turn each file type into units with an exact origin. Transcript/ASR windows are closed *before* appending a cue so a unit's timestamp equals its first cue. |
| `core/vision.py` | Save figures (content-addressed), skip icons, describe with a vision model, cache descriptions; otherwise the unit says "visual content not captured". |
| `core/retrieve.py` | TF-IDF (1–2 grams, stop-words removed), refusal gate, topic map building (LLM two-pass, heuristic fallback, curated sample maps), unit tagging. |
| `core/tutor.py` | Grounded answer from retrieved excerpts only; personalised hint from mastery; optional answer language; extractive fallback. |
| `core/quiz.py` | Generation, verification, novelty (hash + token Jaccard), rotating source units, topic canonicalisation, grading. |
| `core/learner.py` | BKT, retention (forgetting curve), study plan, misconception notes. |
| `core/rootcause.py` | Prerequisite blame, probe selection, repair path, re-test. Pure functions, shared with the simulator. |
| `core/recommend.py`, `core/report.py`, `core/revision.py` | Next best actions with reasons, post-assessment report, flashcards/briefs. |
| `core/evaluate.py`, `core/simulate.py` | Metrics and the latent-gap student simulator. |
| `core/offline.py` | Deterministic fallbacks used whenever no LLM is reachable. |
| `core/llm.py` | OpenAI-compatible client; circuit breaker after credential/quota failures so the app drops to offline mode at once. |

## Grounding method
1. **Retrieve** the top units by TF-IDF similarity. A two-feature **gate** (best similarity, optionally query-term coverage of the course vocabulary) decides whether the material covers the question at all. Thresholds are calibrated on a dev split of `tests/testset.json` and written to `core/gate.json` by `scripts/run_eval.py --write-gate`; with an LLM online a recall-first threshold is used because the model's own `NOT_COVERED` is a second layer.
2. **Generate** with only the retrieved excerpts in context. The prompt requires a `[unit_id]` marker after each claim and the exact refusal string when the excerpts do not contain the answer.
3. **Cite**: markers become numbered citations; each opens its unit in the Source Viewer. If the model forgets markers the UI still attaches the excerpts it was shown.
4. **Separate** outside knowledge: only on an explicit request, in a dashed "outside your materials" box.
5. **Figures** are units like any other. Without a vision model their text says the visual was not captured, and the tutor is instructed to say so.

## Learner model
Per student and topic, BKT with question-type-aware guess/slip (MCQ 0.25/0.10, free text 0.02/0.08) and learning rate 0.12. Priors start at 0.25 and can be seeded by an intake form (self-rating) or measured by a diagnostic quiz spread over the prerequisite graph. Mastery changes **only** through graded answers (quiz, probe, re-test); chatting and reading update `last_seen` for the forgetting curve (`retention = exp(-days / max(1.5, 6·mastery))`) but never mastery. Wrong answers record a misconception note on the topic.

## RootCause
For a missed topic T with prerequisites P:
1. `blame(p) ∝ (1 − mastery(p)) × (1 + 0.5 · [p never answered])`, normalised over P.
2. Probe the argmax with two verified MCQs on p (generated and verified by the normal quiz pipeline). If every prerequisite is already strong from real answers, skip the probe and review T itself.
3. The probe answers go through the ordinary BKT update (`source="probe"`) — the only thing that can lower p's mastery.
4. If p is confirmed weak (answered and mastery < 0.6): repair path = known-weak prerequisites of p (one level of recursion) then p, each with its top-2 source units, then a re-test of T. Otherwise the next suspect is offered.

The same functions run inside `core/simulate.py`, so the evaluated policy is the shipped policy.

## API surface (all JSON)
`/api/status`, `/api/upload`, `/api/library`, `/api/load-sample`, `/api/analyze`, `/api/graph`, `/api/chat`, `/api/quiz`, `/api/diagnostic`, `/api/grade`, `/api/report`, `/api/recommendations`, `/api/dashboard`, `/api/plan`, `/api/flashcards`, `/api/brief`, `/api/intake`, `/api/students`, `/api/rootcause/diagnose`, `/api/rootcause/probe-result`, `/api/source/view`, `/api/preview/{file}`, `/api/figure/{name}`, `/api/media/{file}`, `/api/evaluation`, `/api/demo-student`. Original contracts (`/api/upload`, `/api/kb`, `/api/analyze`, `/api/chat`, `/api/quiz`, `/api/grade`, `/api/dashboard`, `/api/plan`, `/api/source`) keep their request shapes; new fields are optional.

## Failure behaviour
No LLM / bad key / no credits → `core/llm.py` trips a 2-minute circuit breaker and every feature uses its offline fallback (extractive answers, cloze questions, heuristic or curated topic maps, token-overlap grading). The UI shows the mode. No Whisper → audio/video upload returns a clear error and the transcript path still works. No vision model → figures are flagged, never guessed.
