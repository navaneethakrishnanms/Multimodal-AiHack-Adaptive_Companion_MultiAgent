# Evaluation Results — Study Companion

Measured with `scripts/run_eval.py` on the sample course
("Introduction to Machine Learning": notes + slides + lecture transcript, 13 passages).
Test set: `tests/testset.json` — 10 on-material questions with known source locations
+ 4 off-material questions. Total runtime: 8.6 min.

## Retrieval

| Metric | Value |
|---|---|
| Cases | 10 |
| Hit rate @6 | **1.00** |
| MRR | **0.717** |

All target passages were found within the top-6 retrieved units (TF-IDF); the mean
reciprocal rank of 0.72 means the correct unit was typically ranked first or second.
Matching tolerates ±1 paragraph to account for chunk-boundary arbitrariness; slide and
timestamp locations are matched exactly.

## Generation quality (LLM-judged, 10 on-material questions)

| Metric | Value |
|---|---|
| Faithfulness | **0.80** |
| Answer relevancy | **1.00** |
| Off-material questions | 4 |
| Refusal accuracy | **1.00** |

- **Faithfulness 0.80**: 8 of 10 answers had every claim supported by the cited excerpts.
  The two flagged answers cite correct sources but include a wrapping sentence the judge
  considered unsupported — we report the strict number rather than tuning the judge.
- **Refusal accuracy 1.00**: all four off-material questions ("capital of France",
  "transformers/self-attention", "chocolate cake recipe", "2018 FIFA World Cup") were
  declined with `NOT_COVERED`, never answered from outside knowledge.

## Personalization (simulated student profiles)

Each profile answered generated quizzes across 3 sessions (questions drawn from the
same pipeline used in the app).

| Profile | Simulated accuracy | Questions asked | Mastery gain (avg) | Repetition rate |
|---|---|---|---|---|
| Weak | 0.35 | 12 | **+0.169** | **0.0** |
| Medium | 0.60 | 8 | **+0.076** | **0.0** |
| Strong | 0.85 | 12 | **+0.431** | **0.0** |

- **Repetition rate 0.0** across all profiles: the novelty hash-set blocked every repeated
  stem across sessions — no student ever saw the same question twice.
- **Mastery gain positive in all profiles**: BKT estimates rise with demonstrated
  knowledge and fall when answers are wrong (the weak profile's earlier dip below the
  neutral prior in a 3-session run reflects correct calibration on a weak student).
- Verification: every served question passed the independent solve-check
  (`verified: true`) — mismatched keys were dropped or repaired before reaching students.

## How to reproduce

```bash
pip install -r requirements.txt
cp config.example.env .env   # set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
# ingest sample material + build the topic map (via the app or:)
.venv/bin/python scripts/run_eval.py
```

The runner writes `eval_report.json` (machine-readable) and prints the same numbers above.
Raw counts (not just rates) are included in the report so results can be audited.

## Honest limitations

- The faithfulness / relevancy judges are LLM-based; a stricter or looser judge shifts the
  numbers. We report the strict interpretation.
- The test set is small (14 cases) and self-built; it demonstrates the pipeline rather than
  establishing benchmark-level performance.
- Retrieval is TF-IDF (no embeddings) — strong on this corpus, but a large corpus with
  paraphrased queries would benefit from hybrid retrieval.
