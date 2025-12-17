# BrightFox RAG Evaluation Suite

Welcome to the BrightFox RAG Evaluation documentation. This project provides a comprehensive framework for evaluating Retrieval-Augmented Generation (RAG) systems.

## Quick Links

| Document | Description |
|----------|-------------|
| [EVAL_RUNBOOK.md](EVAL_RUNBOOK.md) | **Start here** - How to run evaluations |
| [EVAL_LOG.md](../EVAL_LOG.md) | Log of all evaluation runs (TID_01 - TID_10) |
| [GCS_MANIFEST.md](../clients/BFAI/GCS_MANIFEST.md) | BFAI data locations in GCS |
| [Architecture](architecture.md) | System design and data flow |
| [Metrics](metrics.md) | What each evaluation metric means |

## Evaluation History

We have completed **10 evaluation tests** to date:

| TID | Date | Type | Description | Key Result |
|-----|------|------|-------------|------------|
| TID_01 | 2024-12-14 | Ad-Hoc | Embedding model comparison | gemini-RETRIEVAL_QUERY wins |
| TID_02 | 2024-12-14 | Ad-Hoc | Embedding dimension test | 768 sufficient |
| TID_03 | 2024-12-14 | Ad-Hoc | Azure vs GCP comparison | GCP +13% pass rate |
| TID_04 | 2025-12-15 | Ad-Hoc | Temperature sweep | 0.0 optimal |
| TID_05 | 2025-12-15 | Ad-Hoc | Context size sweep | P@25 optimal |
| TID_06 | 2025-12-15 | Ad-Hoc | Gemini Pro low reasoning | Comparable to Flash |
| TID_07 | 2025-12-15 | Ad-Hoc | E2E consistency test | High consistency |
| TID_08 | 2025-12-16 | Core | Gold Standard P@12 | 95.4% pass |
| TID_09 | 2025-12-16 | Core | Gold Standard P@25 | **96.1% pass** |
| TID_10 | 2025-12-17 | Ad-Hoc | Failure rerun enhanced | 67% improved |

**GCS Bucket:** `gs://bfai-eval-suite/BFAI/`

## Current Gold Corpus

**File:** `clients/BFAI/qa/QA_BFAI_gold_v1-0__q458.json`

| Metric | Value |
|--------|-------|
| Questions | 458 |
| Single-hop | 222 (48%) |
| Multi-hop | 236 (52%) |
| Pass rate (P@25) | 96.1% |

## Project Structure

```
bfai_eval_suite/
├── clients/                 # Client data (per-client isolation)
│   └── BFAI/                # BrightFox AI demo suite
│       ├── corpus/          # Foundational data [GITIGNORED]
│       │   ├── documents/   # Source PDFs
│       │   ├── metadata/    # Per-doc analysis
│       │   └── chunks/      # Chunked text
│       ├── qa/              # QA test sets [IN REPO]
│       │   └── QA_BFAI_gold_v1-0__q458.json
│       ├── tests/           # Evaluation runs
│       │   └── TID_XX/data/ # [GITIGNORED]
│       └── GCS_MANIFEST.md  # Pointer to GCS
├── src/                     # Reusable code
├── scripts/                 # Runnable scripts
│   ├── eval/                # Evaluation scripts
│   ├── corpus/              # QA generation
│   └── setup/               # One-time setup
├── reports/                 # Polished reports
├── docs/                    # This documentation
└── EVAL_LOG.md              # Global test log
```

## Getting Started

```bash
# Pre-flight check
python scripts/eval/preflight_check.py

# Run evaluation
python scripts/eval/run_gold_eval.py --precision 25

# After run: upload to GCS and update EVAL_LOG.md
# See EVAL_RUNBOOK.md Section 9: Post-Run Checklist
```
