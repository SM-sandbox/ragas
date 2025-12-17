# Evaluation Cookbook

A practical guide to running experiments, managing data, and understanding the evaluation framework.

## Quick Start

### 1. Run an Experiment

```bash
cd bfai_eval_suite
source venv/bin/activate

# Run a single experiment
python scripts/remaining_experiments.py
```

### 2. Generate a Report

```bash
python scripts/generate_sample_report.py
```

### 3. View Documentation

```bash
mkdocs serve
# Open http://localhost:8000
```

---

## Repository Structure

```
bfai_eval_suite/
├── core/                    # Core framework modules
│   ├── models.py           # Model registry integration
│   ├── preflight.py        # Pre-flight validation checks
│   ├── metrics.py          # Retrieval metrics (precision, recall, MRR)
│   ├── report.py           # Report generation and formatting
│   └── archive.py          # GCS archiving for experiment data
│
├── scripts/                 # Runnable experiment scripts
│   ├── remaining_experiments.py    # Main experiment runner
│   ├── generate_sample_report.py   # Report generator
│   ├── temperature_context_sweep.py # Parameter sweeps
│   └── retrieval_metrics_evaluator.py
│
├── corpus/                  # Test data
│   ├── qa_corpus_200.json  # 224 Q&A pairs with ground truth
│   └── document_inventory.md
│
├── experiments/             # Experiment runs (large files → GCS)
│   └── YYYY-MM-DD_experiment_name/
│       ├── retrieval_cache.json    # Cached retrieval results
│       ├── *_checkpoint.jsonl      # Per-question results
│       ├── *_results.json          # Aggregated results
│       └── archive_manifest.json   # GCS archive reference
│
├── reports/                 # Generated reports (committed to git)
│   ├── Temperature_Context_Sweep_Summary.md
│   ├── context_100_NEW_FORMAT.md
│   └── Generation_Consistency_Report.md
│
├── docs/                    # MkDocs documentation source
│   ├── index.md
│   ├── architecture.md
│   ├── cookbook.md          # This file
│   └── ...
│
├── tests/                   # Unit tests
│   └── unit/
│       ├── test_models.py
│       ├── test_preflight.py
│       └── test_metrics.py
│
└── data/                    # Document chunks (gitignored → GCS)
    └── chunks/*.jsonl
```

---

## Data Storage Strategy

### What's in Git (Small Files)
- **Reports** (`reports/*.md`) - Human-readable experiment summaries
- **Corpus** (`corpus/qa_corpus_200.json`) - 224 Q&A test pairs
- **Code** (`core/`, `scripts/`, `tests/`)
- **Docs** (`docs/`)
- **Results JSON** (`experiments/*_results.json`) - Aggregated metrics only

### What's in GCS (Large Files)
- **Retrieval cache** (`retrieval_cache.json`) - ~80MB per experiment
- **Checkpoints** (`*_checkpoint.jsonl`) - Per-question raw results
- **Document chunks** (`data/chunks/*.jsonl`)

### GCS Archive Location
```
gs://brightfoxai-documents/BRIGHTFOXAI/EVAL_ARCHIVE/
└── {experiment_name}_{timestamp}/
    ├── retrieval_cache.json
    ├── *_checkpoint.jsonl
    └── manifest.json
```

### Archiving Experiments

```python
from core.archive import archive_experiment_to_gcs

# Archive an experiment directory
manifest = archive_experiment_to_gcs(
    experiment_dir=Path("experiments/2025-12-15_temp_context_sweep"),
    experiment_name="temp_context_sweep",
    metadata={"model": "gemini-2.5-flash", "corpus": "qa_corpus_200"}
)
print(f"Archived to: {manifest.gcs_path}")
```

### Restoring from Archive

```python
from core.archive import download_archived_experiment

download_archived_experiment(
    experiment_name="temp_context_sweep_20251216_112419",
    local_dir=Path("experiments/restored")
)
```

---

## Running Experiments

### Experiment Configuration

Experiments are configured in `scripts/remaining_experiments.py`:

```python
runner.run_experiment(
    experiment_name="context_100",      # Name for checkpoint files
    context_size=100,                   # Number of chunks to include
    model="gemini-2.5-flash",          # Generation model
    temperature=0.0,                    # Temperature (0.0 = deterministic)
    thinking_budget=None               # For reasoning models only
)
```

### Checkpoint-Based Resumption

Experiments automatically resume from checkpoints:

1. **Checkpoint file**: `{experiment_name}_checkpoint.jsonl`
2. Each line = one completed question
3. On restart, skips already-completed questions
4. Safe to interrupt and resume

### Pre-flight Checks

Before running, validate your environment:

```python
from core.preflight import run_preflight_checks, PreflightConfig

config = PreflightConfig(
    job_id="bfai__eval66a_g1_1536_tt",
    corpus_path="corpus/qa_corpus_200.json",
    model="gemini-2.5-flash"
)
result = run_preflight_checks(config)
if not result.passed:
    print("Pre-flight failed:", result.summary)
```

---

## Report Generation

### Standard Report Format

Reports include:
- **Configuration** - Model, temperature, context size
- **Pre-flight Results** - Validation status
- **Retrieval Metrics** - Precision@k, Recall@k, MRR@k
- **LLM Judge Results** - Pass/partial/fail rates, scores
- **Timing Breakdown** - Generation, judge, total time
- **Token Usage** - Prompt, completion, thinking tokens

### Generating Reports

```python
from core.report import create_report_from_results, save_report

report = create_report_from_results(
    results_file="experiments/.../context_100_results.json",
    experiment_name="context_100"
)
save_report(report, "reports/context_100_Report.md")
```

---

## Metrics Reference

### Retrieval Metrics

| Metric | Description |
|--------|-------------|
| **Recall@k** | % of relevant docs in top-k retrieved |
| **Precision@k** | % of top-k that are relevant |
| **MRR@k** | Mean Reciprocal Rank of first relevant doc |
| **Hit Rate@k** | % of queries with ≥1 relevant in top-k |

### Judge Metrics

| Metric | Scale | Description |
|--------|-------|-------------|
| **Correctness** | 1-5 | Factual accuracy |
| **Completeness** | 1-5 | Covers all aspects |
| **Faithfulness** | 1-5 | Grounded in context |
| **Relevance** | 1-5 | Addresses the question |
| **Clarity** | 1-5 | Well-structured response |
| **Overall Score** | 1-5 | Weighted average |

### Verdicts

| Verdict | Criteria |
|---------|----------|
| **pass** | Overall score ≥ 4.0 |
| **partial** | Overall score 3.0-3.9 |
| **fail** | Overall score < 3.0 |

---

## Common Tasks

### Add a New Model

1. Add to `core/models.py` or use orchestrator's `approved_models.py`
2. Update experiment script with new model name
3. Run experiment with `model="new-model-name"`

### Compare Two Models

```bash
# Run both experiments
python -c "
from scripts.remaining_experiments import RemainingExperiments
runner = RemainingExperiments()
runner.run_experiment('model_a', model='gemini-2.5-flash')
runner.run_experiment('model_b', model='gemini-2.5-pro')
"

# Compare results
python scripts/generate_comparison_report.py model_a model_b
```

### Re-run Failed Questions

```python
# Remove failed entries from checkpoint
import json
checkpoint = "experiments/.../experiment_checkpoint.jsonl"
entries = [json.loads(l) for l in open(checkpoint)]
good = [e for e in entries if '[Generation Error' not in e.get('answer', '')]
with open(checkpoint, 'w') as f:
    for e in good:
        f.write(json.dumps(e) + '\n')

# Re-run - will pick up where it left off
python scripts/remaining_experiments.py
```

---

## Troubleshooting

### TLS Certificate Errors

**Symptom:** `503 Could not find a suitable TLS CA certificate bundle`

**Cause:** Virtual environment path changed or missing certifi package.

**Fix:**
```bash
pip install --upgrade certifi
# Or recreate venv
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

### Rate Limiting (429 Errors)

**Symptom:** `429 Resource Exhausted` or `Quota exceeded`

**Cause:** Too many concurrent API calls.

**Fix:** The experiment runner has built-in retry with exponential backoff. If persistent, reduce concurrency or wait.

### Missing Retrieval Cache

**Symptom:** `FileNotFoundError: retrieval_cache.json`

**Cause:** Large files gitignored, need to restore from GCS.

**Fix:**
```python
from core.archive import download_archived_experiment
download_archived_experiment("experiment_name_timestamp", Path("experiments/restored"))
```
