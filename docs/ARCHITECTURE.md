# Architecture & Design — Study Companion

This document covers the three things Track D asks to be documented: the **architecture**,
the **grounding method**, and the **learner-model approach**.

## 1. System architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Browser (static/index.html — single page, no build step)          │
│   Material · Tutor · Quiz · Dashboard                              │
└───────────────▲────────────────────────────────────────────────────┘
                │ REST (JSON)
┌───────────────┴────────────────────────────────────────────────────┐
│  server.py — FastAPI                                                │
│   /api/upload  /api/analyze  /api/chat  /api/quiz  /api/grade       │
│   /api/dashboard  /api/plan  /api/source                            │
└───┬──────────┬───────────┬────────────┬────────────┬───────────────┘
    │          │           │            │            │
 ingest.py  retrieve.py  tutor.py     quiz.py     learner.py
    │          │           │            │            │
    └──────────┴─────┬─────┴────────────┴────────────┘
                     │
              core/llm.py  (OpenAI-compatible client)
```

Design choices:

- **No external vector DB / embedding service.** Retrieval is TF-IDF (scikit-learn) with an
  optional LLM re-ranker. This keeps the system runnable anywhere (no API keys besides the LLM,
  works in any language) and is fast on course-sized corpora. The interface (`Retriever.search`)
  is small enough to swap in embeddings without touching the rest.
- **Stateless core, thin server.** Every core module is a pure function over a knowledge base
  object; the server owns persistence (JSON files) so the system is easy to test and audit.
- **The LLM client is provider-agnostic** (any OpenAI-compatible endpoint; configurable extra
  headers), with two robustness behaviors learned in production: automatic retry when an output
  is truncated at the token limit, and automatic fallback when a provider rejects strict
  JSON mode.

## 2. Grounding method

The grounding contract has three layers:

1. **Source-linked units.** Ingestion never stores bare text: every unit carries
   `{file, type, page/slide/timestamp, image_count}`. Nothing downstream can lose the origin.
2. **Retrieval-scoped generation.** The tutor prompt receives *only* the top-k retrieved
   excerpts (with ids and locations) and must cite each claim as `[unit_id]`. The model never
   sees the rest of the course, so it cannot blend unrelated context.
3. **Explicit refusal.** The prompt defines a refusal token (`NOT_COVERED`) for questions the
   excerpts do not answer. The API surfaces `refused: true` and the UI flags the message —
   the student can see the difference between "your material says" and "I don't know".

The citation viewer closes the loop: clicking a citation opens the exact excerpt
(`/api/source?file=...&loc=...`), so verification costs one click.

## 3. Quiz generation & verification

- Generation produces structured questions tagged to `{topic, source_unit, difficulty, type}`.
- **Structural validation** drops broken items (MCQ without exactly 4 options; answer not
  among options; missing source unit).
- **Independent verification**: a second LLM call solves each question from its source excerpt
  without seeing the proposed answer. `mismatch` verdicts drop the item (or repair short/numerical
  keys with the independently solved answer); `unclear` verdicts keep the item but mark
  `verified: false` so the dashboard can show honesty.
- **Novelty**: question stems are normalized and hashed; the hash set persists across
  assessments, so later quizzes avoid repeating earlier questions. The repetition rate is
  reported in the evaluation.

## 4. Learner model

Per-topic mastery uses **Bayesian Knowledge Tracing** (single-skill BKT):

```
P(Ln | correct)   = P(Ln-1)(1-slip) / [P(Ln-1)(1-slip) + (1-P(Ln-1))guess]
P(Ln | incorrect) = P(Ln-1)slip / [P(Ln-1)slip + (1-P(Ln-1))(1-guess)]
P(Ln+1)           = P(Ln) + (1-P(Ln)) * learn
```

- Priors are question-type aware: `guess=0.25, slip=0.10` for 4-option MCQ; `guess=0.02,
  slip=0.08` for short/numerical (guessing is rare, careless slips less so).
- `learn=0.12` — each graded opportunity also implies learning happened.
- New students start at `P=0.25` (neutral); the first diagnostic quiz moves the estimate.
- The **study plan** combines mastery with an Ebbinghaus-style retention estimate
  (`exp(-days_since / f(p))`) to weight weak + forgotten topics first over a 7-day horizon.

Why BKT: it is simple, interpretable (the dashboard shows exactly one number per topic),
updates in O(1) after every answer, and has well-understood extensions (IRT, DKT) if the
prototype needs to scale.

## 5. Evaluation design

`tests/testset.json` contains two kinds of cases:

- 10 questions with **known source locations** (used for retrieval hit-rate and MRR), and
- 4 **off-material questions** (used for refusal accuracy).

Generation quality is judged by the LLM with two prompts (faithfulness to cited excerpts,
answer relevancy), but the report always includes raw counts so results can be audited rather
than trusted blindly. Personalization is evaluated with **simulated students** (three accuracy
profiles) running multiple sessions of generated quizzes, reporting mastery gain and question
repetition rate.

See `docs/EVALUATION.md` for the latest measured results.
