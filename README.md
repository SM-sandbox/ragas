# BrightFox RAG Evaluation Suite

End-to-end RAG evaluation framework for enterprise document corpora.

## Overview

This suite evaluates RAG systems using:

- **Vertex AI Vector Search** for retrieval
- **Gemini 2.5 Flash/Pro** for generation
- **LLM-as-Judge** methodology for evaluation
- Client-isolated data structure with GCS backup

## Project Structure

```
ragas/
├── clients/                     # Client data (per-client isolation)
│   └── BFAI/                    # BrightFox AI demo suite
│       ├── corpus/              # Foundational data [GITIGNORED]
│       │   ├── documents/       # Source PDFs
│       │   ├── metadata/        # Per-doc analysis
│       │   └── chunks/          # Chunked text
│       ├── qa/                  # QA test sets [IN REPO]
│       │   └── QA_BFAI_gold_v1-0__q458.json
│       ├── tests/               # Evaluation runs (TID_XX)
│       └── GCS_MANIFEST.md      # Pointer to GCS
│
├── src/                         # Reusable Python modules
├── scripts/
│   ├── eval/                    # Evaluation scripts
│   ├── corpus/                  # QA generation
│   └── setup/                   # One-time setup
├── reports/                     # Polished reports
├── docs/                        # Documentation
├── EVAL_LOG.md                  # Global test log
└── _upstream_ragas/             # Original ragas library (archived)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with GCP
gcloud auth application-default login

# 3. Pre-flight check
python scripts/eval/preflight_check.py

# 4. Run evaluation
python scripts/eval/run_gold_eval.py --precision 25
```

## Evaluation History

| TID | Date | Type | Description | Result |
|-----|------|------|-------------|--------|
| TID_08 | 2025-12-16 | Core | Gold Standard P@12 | 95.4% pass |
| TID_09 | 2025-12-16 | Core | Gold Standard P@25 | **96.1% pass** |
| TID_10 | 2025-12-17 | Ad-Hoc | Failure rerun | 67% improved |

See `EVAL_LOG.md` for complete test history.

## Key Documents

| Document | Description |
|----------|-------------|
| [EVAL_RUNBOOK.md](docs/EVAL_RUNBOOK.md) | How to run evaluations |
| [EVAL_LOG.md](EVAL_LOG.md) | Test history (TID_01 - TID_10) |
| [GCS_MANIFEST.md](clients/BFAI/GCS_MANIFEST.md) | GCS data locations |

## GCS Bucket

**Bucket:** `gs://bfai-eval-suite`

```bash
# Upload test results
gcloud storage cp -r clients/BFAI/tests/TID_XX/ gs://bfai-eval-suite/BFAI/tests/TID_XX/

# Download corpus (new machine)
gcloud storage cp -r gs://bfai-eval-suite/BFAI/corpus/ clients/BFAI/corpus/
```

## GCP Resources

| Resource | Value |
|----------|-------|
| Project | civic-athlete-473921-c0 |
| Location | us-east1 |
| Embedding | gemini-embedding-001 (768 dim) |
| LLM | gemini-2.5-flash |
| Judge | gemini-2.5-flash |
