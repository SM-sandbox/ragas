# Runs and Experiments System

This document describes the 3-tier evaluation system for organizing RAG evaluation outputs.

## Overview

All evaluation outputs are organized into three categories based on their purpose:

| Type | Code | Directory | Purpose |
|------|------|-----------|---------|
| **Checkpoint** | `C###` | `checkpoints/` | CICD/daily health checks - locked baseline config |
| **Run** | `R###` | `runs/` | Full evaluation with variations |
| **Experiment** | `E###` | `experiments/` | Quick exploratory tests |

## When to Use Each Type

### Checkpoints (C###)

Use for **production health checks** and **CICD gating**.

- **Config:** Fixed, locked - same every time
- **Corpus:** Full 458 questions
- **Question:** "Is the system still working as expected?"
- **Example:** Daily regression check, pre-deployment validation

### Runs (R###)

Use for **hypothesis testing** with full corpus.

- **Config:** Variable - testing different settings
- **Corpus:** Full 458 questions
- **Question:** "What happens if I change X?"
- **Example:** Testing precision@12 vs precision@25, comparing models

### Experiments (E###)

Use for **quick exploratory tests**.

- **Config:** Variable
- **Corpus:** Partial (30, 100 questions)
- **Question:** "Let me quickly check something"
- **Example:** Debugging a failure, testing a prompt change

## Naming Conventions

### General Format

```text
{TYPE}{NNN}__{YYYY-MM-DD}__{mode}__{description}/
```

### Components

| Component | Description | Values |
|-----------|-------------|--------|
| `TYPE` | Category code | `C`, `R`, `E` |
| `NNN` | Sequential 3-digit number | `001`, `002`, ... |
| `YYYY-MM-DD` | Date of run | `2025-12-19` |
| `mode` | Execution environment | `L` (local), `C` (cloud) |
| `description` | Brief config summary | `p25-flash`, `p12-test` |

### Examples

```text
C001__2025-12-19__C__p25-flash/      # Checkpoint: cloud, precision@25
R001__2025-12-18__L__p12-test/       # Run: local, testing precision@12
E001__2025-12-17__L__quick30-debug/  # Experiment: local, 30 questions
```

## Directory Structure

```text
ragas/
├── clients_qa_gold/              # Gold standard data (per client)
│   └── BFAI/
│       ├── corpus/               # Document corpus for indexing
│       ├── qa/                   # Gold QA pairs
│       │   └── QA_BFAI_gold_v1-0__q458.json
│       ├── baselines/            # Gold baseline snapshots
│       │   ├── baseline_gold__BFAI__v1__2025-12-17__q458.json
│       │   ├── REGISTRY.md
│       │   └── registry.json
│       └── tests/                # Client-specific test data
│
├── clients_eval_data/            # Evaluation outputs (per client)
│   └── BFAI/
│       ├── checkpoints/          # CICD/daily health checks (C###)
│       │   ├── C001__2025-12-19__C__p25-flash/
│       │   │   ├── checkpoint.json
│       │   │   └── results.json
│       │   ├── REGISTRY.md
│       │   └── registry.json
│       │
│       ├── runs/                 # Full evaluation runs (R###)
│       │   ├── R001__2025-12-18__L__p25-flash-2.5/
│       │   │   ├── results.jsonl
│       │   │   ├── run_summary.json
│       │   │   └── comparison.md
│       │   ├── REGISTRY.md
│       │   └── registry.json
│       │
│       └── experiments/          # Quick experiments (E###)
│           ├── E001__2025-12-17__L__gemini3-eval/
│           ├── REGISTRY.md
│           └── registry.json
│
├── eval_runners/                 # Entry point scripts (shared)
│   ├── adhoc/                    # Ad-hoc evaluation runner
│   ├── baseline/                 # Baseline evaluation runner
│   └── cicd/                     # CICD evaluation runner
```

## Registry Files

Each directory contains two registry files:

### REGISTRY.md (Human-readable)

A markdown table listing all entries with key metrics.

### registry.json (Programmatic)

A JSON file for programmatic access:

```json
{
  "schema_version": "1.0",
  "type": "checkpoints",
  "entries": [
    {
      "id": "C001",
      "date": "2025-12-19",
      "mode": "cloud",
      "config_summary": "p25-flash",
      "questions": 458,
      "pass_rate": 0.926,
      "folder": "C001__2025-12-19__C__p25-flash"
    }
  ]
}
```

## Baselines

Baselines are the **gold standard snapshots** that runs are compared against.

### Naming Convention

```text
baseline_gold__{client}__{version}__{date}__q{count}.json
```

### Example

```text
baseline_gold__BFAI__v2__2025-12-18__q458.json
```

### When to Update Baseline

1. Run a full evaluation with significant improvements
2. Verify metrics meet quality thresholds
3. Use `--update-baseline` flag to save
4. Update registry files

## Adding New Entries

### 1. Run the Evaluation

```bash
# Checkpoint (CICD)
python evaluations/cicd/run_cicd_eval.py

# Run (full eval with variations)
python evaluations/baseline/run_baseline.py --precision 12

# Experiment (quick test)
python lib/core/evaluator.py --cloud --quick 30
```

### 2. Move/Rename Output

Rename the output folder to match the naming convention.

### 3. Update Registry

Add entry to both `REGISTRY.md` and `registry.json`.

## Related Documentation

- `docs/EVAL_SYSTEM.md` - Full evaluation system architecture
- `docs/EVAL_RUNBOOK.md` - Step-by-step evaluation procedures
- `testing-manifest.md` - Testing specifications
