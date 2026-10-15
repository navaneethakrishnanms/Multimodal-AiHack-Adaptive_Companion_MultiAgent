# Audit against the six judging criteria

Evidence is listed as repository locations; numbers live in `eval_report.json` / `docs/EVALUATION.md` and are not repeated here.

## 1. Knowledge Base & Grounding (20%)
| Aspect | Evidence |
|---|---|
| Multimodal extraction | `core/ingest.py` (PDF/PPTX/MD/VTT/SRT + Whisper audio/video), `core/vision.py` (figures → citable units, vision descriptions, cache), tests `tests/test_multimodal.py` incl. a real Whisper run on the sample video |
| Source-link preservation | every `Unit` has file + page/slide/timestamp (+ `start_sec`, figure path); transcript windowing fixed so labels equal the first cue (`test_transcript_unit_label_matches_its_first_cue`) |
| Citation accuracy | numbered citations bound to `[unit_id]` markers, Source Viewer renders the exact page/slide/figure/video time (`/api/source/view`); metric: citation precision in `eval_report.json → generation` |
| Correct refusal | calibrated two-feature gate + model-level `NOT_COVERED`; held-out numbers in `retrieval_gate`; flagged outside-knowledge only on request |
| Gaps | figure descriptions and generation faithfulness need a funded LLM; reported "not measured" without one. Sample corpora are small. |

## 2. Personalization Effectiveness (20%)
| Aspect | Evidence |
|---|---|
| Learner model | `core/learner.py` BKT per student/topic, evidence-only updates, intake + diagnostic (`/api/intake`, `/api/diagnostic`) |
| Differentiator | `core/rootcause.py` + `/api/rootcause/*` + UI graph; unit tests on a hand-built 5-node graph (`tests/test_rootcause.py`) |
| Measured gains | `core/simulate.py` (latent-gap students, 3 policies, paired CIs, ablation, sensitivity) → `docs/EVALUATION.md`; claims in docs are limited to what the CIs support |
| Responsiveness | tutor tone adapts to topic mastery (`core/tutor.py`), next-best-action reasons (`core/recommend.py`), forgetting-curve schedule |
| Gaps | simulated, not real students; RootCause does not win every metric (documented) |

## 3. Assessment Quality (15%)
Generation with topic/source/difficulty tags (`core/quiz.py`), structural screening (duplicate options, key not in options, non-numeric numerical keys), independent verification with optional `VERIFY_MODEL`, reject→regenerate rounds, per-student novelty (hash + near-duplicate), cited feedback, post-assessment report with misconceptions (`core/report.py`). Audit metrics in `eval_report.json → assessment` (offline mode is trivially verifiable; the LLM path is unit-tested with stubs and measured when a model is connected).

## 4. User Experience & Demo Video (15%)
Six-screen UI (`static/index.html`): Library with ingestion progress and figure gallery, Tutor with Source Viewer and green/amber banners, Practice with chips and cited feedback, Root cause graph with animated path and repair card, Progress with schedule/flashcards/recommendations, Evaluation with charts. Light/dark, skeleton loaders, responsive, keyboard-operable chips and citations. `scripts/ui_smoke.py` drives every screen and fails on JS errors. Demo: `scripts/make_demo.py`, script in `docs/DEMO_SCRIPT.md`. Optional features: flow map, flashcards, spoken briefs, schedule, Hindi, voice in/out.

## 5. System Evaluation (15%)
`core/evaluate.py` + `scripts/run_eval.py`: ID-based context precision/recall and hit/MRR, claim-level faithfulness and answer relevancy (LLM-judged, skipped honestly when unavailable), citation precision, refusal gate with dev/test split, assessment audit, simulated students. Test set with ground-truth locations in `tests/testset.json`, with a history note recording that the first gate calibration was weak and how the set was extended. `docs/EVALUATION.md` is rendered from the JSON, with limitations.

## 6. Technical Implementation (15%)
Modular `core/` package, pure functions shared between product and simulator, per-feature offline fallbacks, LLM circuit breaker, cache for Whisper/vision, Dockerfile/compose, `.env`-based secrets never committed, a full pytest suite (offline, deterministic, no network or keys needed).

## Cheap fixes applied during the audit
- reject→regenerate round and numerical-key screening added to quiz generation (+ test)
- `busy()` button-label bug found by the UI smoke run
- stale numbers removed from README (all figures link to generated docs)
- README credit slot for the original author (the original README names none) — **to be filled by the team**
