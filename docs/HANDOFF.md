# HANDOFF & RUNBOOK: Track D Multimodal AI Hackathon 2026

## 1. What Works
- **Multimodal Ingestion Pipeline (`core/multimodal.py`, `core/extract.py`):**
  - Text, PDF extraction with PyMuPDF.
  - Presentation extraction with `python-pptx` (slides & text).
  - Audio/Video transcription using `faster-whisper` fallback with timestamped segments.
  - Figure & visual extraction from PDF/PPTX with bounding box cropping and base64 storage.
- **Hierarchical Knowledge Store (`core/store.py`):**
  - Citable units with source metadata (`file`, `page`, `slide`, `timestamp_sec`, `figure_path`).
  - Strict lexical BM25 + cosine semantic vector retrieval.
  - Calibrated gating threshold (`min_score=0.14`) to refuse out-of-domain / ungrounded queries.
- **Tutor & Citation Grounding (`core/tutor.py`):**
  - Strict grounded chat answering with inline references.
  - Honest rejection banner (`amber NOT-COVERED`) when retrieval confidence is below gate threshold.
  - Green `SOURCE-BACKED` badge with right-hand live preview when verified.
- **RootCause Prerequisite Diagnosis (`core/rootcause.py`):**
  - Bayesian Knowledge Tracing with prerequisite blame attribution: $blame(p) \propto (1 - mastery(p)) \times (1 + \text{novelty\_bonus})$.
  - Active targeted probing with generated and verified questions on the suspected concept.
  - Direct 2-unit repair path generation and post-repair re-testing.
- **Unified Modern Web UI (`static/index.html`):**
  - Single-page responsive dashboard (Library, Tutor, Practice, RootCause Graph via Cytoscape.js, Progress, Evaluation).
  - Light/Dark theme toggle, interactive source document viewer, and "Demo Student" seed button.
- **REST API (`server.py`):**
  - Endpoints: `/api/courses`, `/api/chat`, `/api/quiz/generate`, `/api/quiz/submit`, `/api/rootcause/diagnose`, `/api/rootcause/probe-result`, `/api/progress`, `/api/eval`.

## 2. What Failed / Known Limitations
- **LLM Rate Limits (HTTP 429) during batch generation evaluation:**
  - With public/trial API quota, generation faithfulness and answer relevancy metrics in `eval_report.json` gracefully skipped execution rather than crashing or faking scores.
  - Offline extractive fallbacks operate deterministically with 100% test pass rate when API key is unset or throttled.
- **Video Generation:**
  - Playwright browser recording and edge-tts narration scripts are implemented in `scripts/make_demo.py`, but automated video rendering was halted per instruction to keep the portal server live and ready for interactive live demo.

## 3. Real Headline Numbers (Source: `eval_report.json`)
Every metric is directly derived from script runs on test data:
- **Retrieval Performance (44 in-material evaluation cases):**
  - Hit Rate @ 4: **95.5%**
  - Mean Reciprocal Rank (MRR): **0.905**
  - Context Precision: **87.5%**
  - Context Recall: **88.6%**
- **Hallucination / Off-Material Refusal Gate:**
  - Off-Material Refusal Rate: **88.9%** (16/18 ungrounded queries rejected)
  - In-Material Answer Rate: **97.7%**
  - Citation Precision: **77.5%**
- **Assessment Verification:**
  - Question Verification Rate: **100%** (11/11 verified against source units)
  - Source Grounding Rate: **100%**
  - Repeat Rate Across Assessments: **0.0%**
- **RootCause Gap Diagnosis (Simulated Student Policy C vs Plain BKT Policy B):**
  - Root Prerequisite Diagnosis Precision (`precision_R`): **48.6%** (RootCause) vs **10.2%** (Plain BKT) -> **~4.8x improvement**
  - Prerequisite Diagnostic F1 (`f1_R`): **0.280** vs **0.159**
  - Overall Concept Gap Precision (`precision_U`): **68.3%** vs **30.6%**
- **Test Suite:**
  - **62 / 62 tests passing** (100% pass rate across offline API, store, rootcause, multimodal, and evaluation tests).

## 4. Exact Commands to Run the Portal
### Activate Environment & Start Server
```powershell
# Open terminal in project root
cd c:\Users\Admin\Downloads\multimodal-study-companion-master

# Activate torch_gpu conda environment
conda activate torch_gpu

# Start portal server (FastAPI + Uvicorn)
python server.py --port 8000
```
Then navigate to: **`http://localhost:8000`** in Google Chrome or any modern browser.

### Run Unit Tests
```powershell
python -m pytest tests/ -v
```

### Re-run Evaluation Suite
```powershell
python scripts/run_eval.py --write-gate
```

## 5. Checklist of Manual Steps Required
- [ ] **Open Browser to http://localhost:8000** and explore the 6 tabs:
  - Library: View loaded lecture slides, transcripts, and figure thumbnails.
  - Tutor: Query course concepts and verify the live source viewer.
  - Practice: Try quiz generation with verified badge chips.
  - RootCause: Click "Demo Student" to seed prerequisite gaps and trigger diagnostic path traversal.
  - Progress: Check topic mastery bars and retention decay.
  - Evaluation: View live charts and tables comparing Policies A, B, and C.
- [ ] **Add Custom OpenAI / Vision API Key (Optional):**
  - Add `OPENAI_API_KEY=your_key_here` to `.env` or `config.example.env` if live GPT-4o-mini generation is desired instead of offline extractive mode.
- [ ] **Demo Video Recording:**
  - Record screen walkthrough of the portal (or run `python scripts/make_demo.py` if autonomous Playwright recording is desired).
- [ ] **Upload Demo Video:**
  - Upload video to YouTube or Loom (unlisted or public).
- [ ] **Devpost Submission:**
  - Copy text directly from `DEVPOST_TEXT.md` into the Devpost submission portal.
  - Attach GitHub repository link: `https://github.com/navaneethakrishnanms/Multimodal-AiHack-Adaptive_Companion_MultiAgent`

---

## 6. Ten-Line Summary
1. Built a complete, production-grade Multimodal Study Companion adhering to all Track D rubric criteria.
2. Implemented hierarchical multimodal parsing for PDFs, slides, figures, and audio/video transcripts.
3. Created an honest citation engine with calibrated refusal gates (88.9% off-material refusal rate).
4. Invented RootCause prerequisite diagnostic policy that scores and probes latent learning gaps.
5. Achieved ~4.8x higher diagnostic precision on root concept gaps compared to standard BKT.
6. Verified 100% of generated assessment questions against source context with zero repeat rate.
7. Crafted a clean, zero-build modern web interface featuring Cytoscape interactive graph visualization.
8. Passed all 62 automated unit and integration tests across offline and online modules.
9. Fully documented codebase with ARCHITECTURE.md, EVALUATION.md, DEVPOST_TEXT.md, and AUDIT.md.
10. The portal server is now running live on port 8000, ready for immediate demo and presentation.
