# Devpost submission text — Study Companion (Track D)

## Inspiration

Students study from lecture videos, textbooks and slides scattered across platforms —
and generic AI chatbots answer from anywhere *except* the course material, or from
material the student hasn't reached yet. We built a companion that stays grounded in the
student's own material and models what the student actually knows.

## What it does

**Study Companion** turns your own course material into a source-grounded tutor and an
adaptive assessment engine.

- **Multimodal ingestion** — drop in PDF textbooks, PPTX slide decks, notes (TXT/MD) and
  lecture transcripts (VTT/SRT). No manual preprocessing. Every passage keeps its exact
  origin: page number, slide number, or video timestamp.
- **Source-grounded tutor** — answers cite the exact excerpt `[unit_id]`; one click opens
  the source. Questions the material doesn't cover are **declined explicitly**
  (`NOT_COVERED`) — outside knowledge is never blended in silently.
- **Adaptive assessment** — quizzes (MCQ / short answer / numerical) on a chosen scope,
  every question tagged to topic, source and difficulty. Each question is **independently
  verified** by a second model pass that solves it from the source; broken or mismatched
  questions are dropped or repaired. Repeated questions are blocked across assessments.
- **Learner model** — per-topic mastery via **Bayesian Knowledge Tracing** (question-type
  aware guess/slip priors), updated after every graded answer, with a dashboard showing
  weak topics and a 7-day study plan weighted by forgetting curves.

## How we built it

- **Python + FastAPI** backend with a dependency-light core; a single-page vanilla-JS UI.
- **Retrieval**: scikit-learn TF-IDF + optional LLM re-ranking — no external vector DB,
  runs anywhere, works in any language.
- **Grounding contract**: retrieval-scoped prompts + mandatory citations + explicit refusal
  token; the UI flags refusals and opens source excerpts on click.
- **Quiz pipeline**: generate → structural validation → independent solve-verification →
  novelty hashing.
- **Evaluation**: a RAGAS-style harness implemented natively (retrieval hit-rate/MRR,
  faithfulness, answer relevancy, refusal accuracy) on a labeled test set, plus simulated
  student profiles that measure mastery gains and question repetition across sessions.
- **LLM**: any OpenAI-compatible endpoint (we ran it on a DeepSeek-class model), with
  retry/fallback logic for truncated outputs and provider JSON-mode quirks.

## Challenges we ran into

- Getting an LLM to *consistently refuse* off-material questions took a strict prompt
  contract plus UI-level flags — early versions blended outside knowledge subtly.
- Independent question verification caught real answer-key errors that pure generation
  missed; we made mismatches visible (`verified` tags) rather than hiding them.
- Reasoning-style models burn output budget on hidden reasoning; we added truncation
  detection with automatic retry at higher budgets.

## What we learned

Grounding is a *system property*, not a prompt: source-linked ingestion, retrieval-scoped
generation, explicit refusal, and verifiable citations together produce a tutor students
can trust. And an honest evaluation (including off-material refusal and simulated
students) is more valuable than a demo that only shows happy paths.

## What's next

- Speech-to-text for raw lecture videos (transcript-first by design today)
- OCR for diagrams and figures in slides/textbooks
- IRT/DKT extensions of the learner model; multi-language tutoring

## Built with

python, fastapi, scikit-learn, bayesian-knowledge-tracing, llm, rag, tf-idf
