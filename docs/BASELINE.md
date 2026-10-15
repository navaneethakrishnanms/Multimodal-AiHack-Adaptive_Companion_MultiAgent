# Baseline (before our changes)

Measured on 2026-10-15 in the `torch_gpu` conda env, sample course (3 files), **no LLM key configured**.

## Reproduced by us (no LLM needed)

| Item | Result |
|---|---|
| Ingestion of `data/sample_course/*` | 13 units (5 notes paragraphs, 5 slides, 3 transcript windows) — works |
| Retrieval, 10 labelled questions (`scripts/run_eval.py` code path) | hit rate@6 = 1.0, MRR = 0.717 |
| Unit tags / topic map | none — `/api/analyze` returns `{"topics": []}` silently without an LLM |

## Failures observed without a key

| Call | Behaviour |
|---|---|
| `POST /api/chat` | HTTP 500 (unhandled `LLMError`) |
| `POST /api/quiz` | HTTP 500 |
| `scripts/run_eval.py` | crashes in `eval_generation` (`LLMError`), so no report is written |

## Retrieval weakness found
TF-IDF is built without stop-word removal. Off-material queries match on function words:
"What is the capital of France?" scores 0.183 against notes para 1, above the in-material noise
floor, so a similarity threshold cannot be used as a refusal gate. (Fixed in Phase 3.)

## Numbers shipped by the original author (not reproduced by us)
`eval_report.json` in the original repo (LLM-judged; needs a key to reproduce):
faithfulness 0.80, answer relevancy 1.00, refusal accuracy 1.00 (4 off-material questions),
and simulated-student "mastery gain" 0.169 / 0.076 / 0.431 for weak / medium / strong profiles.
The personalization numbers come from students that answer correctly with a fixed probability
regardless of what they know, so they say nothing about whether personalization works.
Those figures are kept here for reference only and are **not** reused in our reporting.
