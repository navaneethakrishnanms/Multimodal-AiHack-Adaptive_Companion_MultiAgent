# Study Companion — a source-grounded, adaptive AI tutor

> Multimodal AI Hackathon 2026 — Track D: Personalized Tutoring & Adaptive Learning

Study Companion unifies lecture videos, textbooks and slide decks into a **source-cited
knowledge base**, then uses it to run **adaptive assessments** and **personalized tutoring**.
Every answer and every quiz question is grounded in your own material, with a citation
that opens the exact page, slide or video timestamp — and the tutor **refuses** questions
the material does not cover instead of hallucinating.

---

## What it does (mapped to Track D requirements)

| Track D requirement | How Study Companion implements it |
|---|---|
| 1a. Multimodal ingestion without manual preprocessing | Drop in PDF / PPTX / TXT / MD / VTT / SRT — parsed automatically (text, slides, speaker notes, timestamped transcripts) |
| 1b. Source-linked units | Every passage keeps `file + page / slide / timestamp`; citations resolve to that exact location |
| 1c. Topics, subtopics, key concepts | One-click **Analyze** builds a topic map with subtopics, key concepts and prerequisites; every passage is tagged |
| 1d. Images/diagrams | Image counts per page/slide are tracked and surfaced; visual content that is not in the text is explicitly flagged |
| 2a. Cited answers | The tutor answers **only** from retrieved excerpts, citing `[unit_id]` for each claim; the UI opens the source excerpt |
| 2b. Decline / flag | Off-material questions return `NOT_COVERED` and are visually flagged — no outside knowledge is blended in |
| 3a. Quiz generation with tags | MCQ / short / numerical questions tagged to topic, source unit and difficulty, on a student-chosen scope |
| 3b. Verification + novelty | A **second independent LLM pass solves every question** and confirms the key (mismatches are dropped or repaired); asked questions are hashed so later quizzes avoid repetition |
| 3c. Cited feedback + reports | Per-answer feedback cites the source; the dashboard reports weak topics and mastery |
| 4a. Learner model | **Bayesian Knowledge Tracing** per topic (guess/slip priors per question type), updated after every graded answer |
| 4b. New students | Mastery starts at a neutral prior; a diagnostic quiz or tutor conversation seeds the model |
| 5a. Evaluation framework | RAGAS-style metrics implemented natively (faithfulness, answer relevancy, retrieval hit rate/MRR, refusal accuracy) |
| 5b. Metrics on a team-built test set | `tests/testset.json` — 10 questions with known source locations + 4 off-material queries |
| 5c. Personalization evaluation | Simulated student profiles (weak / medium / strong) run across sessions; mastery gains and question repetition rate reported |
| 6. Optional extras | 7-day study plan weighted by mastery + forgetting curves; topic map; per-topic mastery dashboard |

## Architecture

```
 upload ──► ingest (pdf/pptx/txt/vtt) ──► knowledge base (JSON, source-linked units)
                                              │
                        analyze ──► topic map (LLM) ──► unit tags
                                              │
 question ──► TF-IDF retrieval ──► (optional LLM re-rank) ──► grounded answer + citations
                                              │
 quiz scope ──► generation (LLM) ──► independent verification (LLM) ──► adaptive quiz
                                              │
 answer ──► grading ──► Bayesian Knowledge Tracing ──► mastery dashboard + study plan
```

- **Retrieval**: scikit-learn TF-IDF (local, no embedding service needed; works in any language).
  An optional LLM re-ranker improves the ordering of the top candidates.
- **Grounding**: the generation prompt receives *only* retrieved excerpts and must cite them;
  a refusal token (`NOT_COVERED`) is part of the contract and is surfaced to the UI.
- **Verification**: quiz answers are validated by an independent solve from the source excerpt
  (cross-model validation); structurally broken MCQs are dropped; novelty is enforced with stem hashing.
- **Learner model**: standard BKT update with question-type-aware guess/slip priors, plus an
  Ebbinghaus-style retention estimate for the study plan.

## Setup

```bash
# 1. Python 3.11+ (tested on 3.12)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure the LLM (any OpenAI-compatible endpoint)
cp config.example.env .env
#   edit .env:  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
#   optional:   LLM_EXTRA_HEADERS={"x-your-header": "value"}

# 3. Run
uvicorn server:app --port 7861
#   open http://127.0.0.1:7861
```

A sample course ("Introduction to Machine Learning": notes + slides + lecture transcript) ships in
`data/sample_course/` — regenerate with `python scripts/make_sample.py`.

## Using it

1. **Material** — drop your files (PDF/PPTX/TXT/MD/VTT/SRT) and click *Analyze* to build the topic map.
2. **Tutor** — ask questions; answers carry citations; click a citation to open the exact excerpt.
   Off-material questions are declined.
3. **Quiz** — pick topics, count and difficulty; every question is verified before it appears;
   submit answers to get cited feedback and mastery updates.
4. **Dashboard** — per-topic mastery (BKT), weak topics, and a 7-day study plan weighted by
   forgetting curves.

## Evaluation

```bash
.venv/bin/python scripts/run_eval.py            # full suite
.venv/bin/python scripts/run_eval.py --skip-personalization
```

Reports: `eval_report.json` (machine-readable) — see `docs/EVALUATION.md` for the latest results
and an honest discussion of limitations.

## Repository layout

```
core/ingest.py     multimodal ingestion → source-linked units
core/retrieve.py   TF-IDF retrieval + LLM re-rank + topic mapping
core/tutor.py      grounded Q&A with citations and refusal
core/quiz.py       quiz generation + independent verification + novelty
core/learner.py    Bayesian Knowledge Tracing + study plan
core/evaluate.py   RAGAS-style metrics + simulated students
server.py          FastAPI app (REST API + static UI)
static/index.html  single-page UI
scripts/           sample material generator, evaluation runner
tests/testset.json labeled evaluation set
```

## Notes & limitations (honest disclosure)

- Video ingestion accepts **transcripts** (VTT/SRT) — the pipeline is designed for transcripts
  exported from lecture videos; speech-to-text is intentionally left to the user's tool of choice
  to keep the system dependency-light.
- Diagram *content* is not OCR'd in this prototype; the system flags where visuals exist and
  tells the student the visual itself was not captured, rather than pretending to understand it.
- The learner model is intentionally simple (single-skill BKT per topic); it is a solid base for
  IRT-style extensions.
- Metrics depend on the LLM judge; the evaluation reports raw counts so results can be audited.
