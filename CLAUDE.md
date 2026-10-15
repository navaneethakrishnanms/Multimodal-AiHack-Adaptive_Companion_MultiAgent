# CLAUDE.md — working rules for this repo

Project: Track D entry (Multimodal AI Hackathon 2026) — source-grounded study companion.
Built on top of an existing open-source repo (credit the original author in README).

## Hard rules
- **Never invent or hand-edit metrics.** Every number in docs, UI or the Devpost text must come
  from a script run and be present in `eval_report.json`. Report negative results honestly.
- Credit the original repo author in README. Write our own docs; do not copy the original's text.
- Minimal, readable changes in the existing style. No heavy new dependencies.
- Every feature needs a graceful fallback when a model, key or optional package is missing.
- Never print, log or commit API keys. `.env` stays git-ignored.
- Do NOT upload to YouTube, submit to Devpost, create accounts or enter credentials.
  Stop and tell the user when those are needed.
- Python env for installs/runs: conda env `torch_gpu`
  (`C:\Users\Admin\anaconda3\envs\torch_gpu\python.exe`).

## Workflow
- Work through the phases in order; commit after each ("phase N: ...").
- If a phase fails after 2 honest attempts, log it in `docs/BLOCKERS.md` and move on.
- Only ask the user for: an API key, a login, or an irreversible decision.

## Layout
- `core/` ingestion, retrieval, tutor, quiz, learner model, evaluation (+ `rootcause.py`)
- `server.py` FastAPI app; `static/index.html` single-file UI (no build step)
- `scripts/` eval + demo tooling; `tests/` unit tests + `testset.json`
- `docs/` architecture, evaluation, understanding, handoff
