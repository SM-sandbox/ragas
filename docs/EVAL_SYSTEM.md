# RAG Pipeline Evaluation System

**Complete Documentation for Idempotent, Baseline-Driven Evaluation**

This document describes the complete evaluation system for RAG pipelines. It is designed to be:
- **Idempotent**: Same inputs → same outputs (temperature=0.0)
- **Baseline-driven**: Compare every run against a known baseline
- **Comprehensive**: Capture all metrics, tokens, latency, cost
- **Automated**: Single script to run, compare, and report

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Directory Structure](#2-directory-structure)
3. [Data Schemas](#3-data-schemas)
4. [Naming Conventions](#4-naming-conventions)
5. [Scripts Reference](#5-scripts-reference)
6. [Configuration](#6-configuration)
7. [Workflow](#7-workflow)
8. [Test Suite](#8-test-suite)
9. [Replication Guide](#9-replication-guide)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        EVAL RUNNER                               │
│  eval_runner.py - Interactive CLI with configurable defaults    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CORE EVAL                                 │
│  core_eval.py - Orchestrates run, saves outputs, compares       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  RUN GOLD EVAL  │ │ BASELINE MGR    │ │ COST CALCULATOR │
│  run_gold_eval  │ │ baseline_manager│ │ cost_calculator │
│  .py            │ │ .py             │ │ .py             │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   ORCHESTRATOR  │ │   BASELINES/    │ │   PRICING       │
│   (gRAG_v3)     │ │   *.json        │ │   TABLES        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Components

| Component | File | Purpose |
|-----------|------|---------|
| **Eval Runner** | `eval_runner.py` | Interactive CLI, shows defaults, accepts overrides |
| **Core Eval** | `core_eval.py` | Main orchestrator, saves JSONL, compares to baseline |
| **Run Gold Eval** | `run_gold_eval.py` | Executes RAG pipeline, captures all metrics |
| **Baseline Manager** | `baseline_manager.py` | Load/save/compare baselines, version management |
| **Cost Calculator** | `cost_calculator.py` | Token → cost calculation with model pricing |
| **Report Generator** | `generate_report.py` | Generate markdown comparison reports from results |

---

## 2. Directory Structure

```
ragas/
├── scripts/eval/
│   ├── eval_runner.py        # Interactive CLI entry point
│   ├── core_eval.py          # Main evaluation orchestrator
│   ├── run_gold_eval.py      # RAG pipeline execution
│   ├── baseline_manager.py   # Baseline CRUD operations
│   ├── cost_calculator.py    # Token cost calculation
│   └── generate_report.py    # Markdown report generator
│
├── baselines/
│   └── baseline_{CLIENT}_v{VERSION}__{DATE}__q{COUNT}.json
│
├── runs/
│   └── {DATE}__{MODEL}__p{PRECISION}__{RUN_ID}/
│       ├── run_summary.json  # Aggregated metrics
│       ├── results.jsonl     # Per-question results
│       └── comparison.md     # Baseline comparison report
│
├── configs/
│   └── eval_config.yaml      # Default configuration
│
├── clients/{CLIENT}/qa/
│   └── QA_{CLIENT}_gold_v{VERSION}__q{COUNT}.json  # Test corpus
│
├── tests/eval/
│   ├── test_cost_calculator.py
│   ├── test_baseline_manager.py
│   ├── test_core_eval.py
│   ├── test_run_gold_eval.py
│   ├── test_integration.py
│   └── test_e2e.py
│
└── docs/
    └── EVAL_SYSTEM.md        # This file
```

---

## 3. Data Schemas

### 3.1 Baseline Schema

**File**: `baselines/baseline_{CLIENT}_v{VERSION}__{DATE}__q{COUNT}.json`

```json
{
  "schema_version": "1.1",
  "baseline_version": "1",
  "created_date": "2025-12-17",
  "client": "BFAI",
  
  "environment": {
    "type": "local",
    "orchestrator_url": null
  },
  
  "index": {
    "job_id": "bfai__eval66a_g1_1536_tt",
    "deployed_index_id": "idx_bfai_eval66a_g1_1536_tt",
    "last_build": "2025-12-15",
    "chunks_indexed": 6059,
    "document_count": 65,
    "embedding_model": "gemini-embedding-001",
    "embedding_dimension": 1536
  },
  
  "corpus": {
    "file": "QA_BFAI_gold_v1-0__q458.json",
    "question_count": 458,
    "distribution": {
      "single_hop": 229,
      "multi_hop": 229,
      "easy": 161,
      "medium": 161,
      "hard": 136
    }
  },
  
  "config": {
    "generator_model": "gemini-2.5-flash",
    "judge_model": "gemini-2.0-flash",
    "reasoning_effort": "low",
    "temperature": 0.0,
    "precision_k": 25,
    "recall_k": 100,
    "enable_hybrid": true,
    "enable_reranking": true,
    "force_json": false,
    "workers": 5
  },
  
  "metrics": {
    "pass_rate": 0.856,
    "partial_rate": 0.105,
    "fail_rate": 0.039,
    "acceptable_rate": 0.961,
    "recall_at_100": 0.991,
    "mrr": 0.717,
    "overall_score_avg": 4.71,
    "scores": {
      "correctness_avg": 4.64,
      "completeness_avg": 4.70,
      "faithfulness_avg": 4.91,
      "relevance_avg": 4.95,
      "clarity_avg": 4.98
    }
  },
  
  "latency": {
    "total_avg_s": 8.3,
    "total_min_s": 3.7,
    "total_max_s": 34.2,
    "by_phase": {
      "retrieval_avg_s": 0.252,
      "rerank_avg_s": 0.196,
      "generation_avg_s": 7.742,
      "judge_avg_s": 1.342
    },
    "by_difficulty": {
      "easy_avg_s": 6.9,
      "medium_avg_s": 8.3,
      "hard_avg_s": 10.0
    }
  },
  
  "tokens": {
    "prompt_total": 1832000,
    "completion_total": 137400,
    "thinking_total": 0,
    "cached_total": 0,
    "total": 1969400,
    "avg_per_question": {
      "prompt": 4000,
      "completion": 300,
      "thinking": 0
    }
  },
  
  "cost": {
    "input_cost": 0.1374,
    "output_cost": 0.04122,
    "thinking_cost": 0.0,
    "cached_cost": 0.0,
    "total_cost_usd": 0.17862,
    "cost_per_question_usd": 0.00039
  },
  
  "answer_stats": {
    "avg_length_chars": 533,
    "min_length_chars": 150,
    "max_length_chars": 1200
  },
  
  "execution": {
    "run_id": "baseline_gold_standard_2025-12-17",
    "run_timestamp": "2025-12-17T00:00:00Z",
    "run_duration_seconds": 3800,
    "questions_per_second": 0.12,
    "workers": 5,
    "hostname": "local",
    "python_version": "3.11",
    "orchestrator_version": "v3.0"
  },
  
  "quality": {
    "avg_logprobs": null,
    "fallback_rate": 0.0,
    "finish_reason_distribution": {
      "STOP": 458,
      "MAX_TOKENS": 0,
      "SAFETY": 0
    }
  },
  
  "retry_stats": {
    "total_questions": 458,
    "succeeded_first_try": 458,
    "succeeded_after_retry": 0,
    "failed_all_retries": 0,
    "total_retry_attempts": 458,
    "avg_attempts": 1.0
  },
  
  "errors": {
    "total_errors": 0,
    "by_phase": {
      "retrieval": 0,
      "rerank": 0,
      "generation": 0,
      "judge": 0
    },
    "error_messages": []
  },
  
  "skipped": {
    "count": 0,
    "reasons": {
      "missing_ground_truth": 0,
      "invalid_question": 0,
      "timeout": 0
    },
    "question_ids": []
  },
  
  "breakdown_by_type": {
    "single_hop": {
      "total": 229,
      "pass": 196,
      "partial": 24,
      "fail": 9,
      "pass_rate": 0.856
    },
    "multi_hop": {
      "total": 229,
      "pass": 196,
      "partial": 24,
      "fail": 9,
      "pass_rate": 0.856
    }
  },
  
  "breakdown_by_difficulty": {
    "easy": {
      "total": 161,
      "pass": 145,
      "partial": 13,
      "fail": 3,
      "pass_rate": 0.901
    },
    "medium": {
      "total": 161,
      "pass": 138,
      "partial": 17,
      "fail": 6,
      "pass_rate": 0.857
    },
    "hard": {
      "total": 136,
      "pass": 109,
      "partial": 18,
      "fail": 9,
      "pass_rate": 0.801
    }
  },
  
  "notes": "Description of this baseline"
}
```

### 3.2 Per-Question Result Schema (JSONL)

**File**: `runs/{RUN_FOLDER}/results.jsonl`

Each line is a JSON object:

```json
{
  "question_id": "sh_easy_001",
  "question_type": "single_hop",
  "difficulty": "easy",
  "recall_hit": true,
  "mrr": 1.0,
  
  "judgment": {
    "correctness": 5,
    "completeness": 5,
    "faithfulness": 5,
    "relevance": 5,
    "clarity": 5,
    "overall_score": 5,
    "verdict": "pass"
  },
  
  "time": 7.826,
  "timing": {
    "retrieval": 0.25,
    "rerank": 0.15,
    "generation": 5.8,
    "judge": 1.3,
    "total": 7.5
  },
  
  "tokens": {
    "prompt": 5000,
    "completion": 300,
    "thinking": 0,
    "total": 5300,
    "cached": 0
  },
  
  "llm_metadata": {
    "model": "gemini-2.5-flash",
    "model_version": "gemini-2.5-flash-preview-05-20",
    "finish_reason": "STOP",
    "reasoning_effort": "low",
    "used_fallback": false,
    "avg_logprobs": null,
    "response_id": "abc123",
    "temperature": 0.0,
    "has_citations": true
  },
  
  "retry_info": {
    "attempts": 1,
    "recovered": false,
    "error": null
  },
  
  "answer_length": 450,
  "retrieval_candidates": 100
}
```

### 3.3 Run Summary Schema

**File**: `runs/{RUN_FOLDER}/run_summary.json`

Same structure as baseline, but without `baseline_version` and with `run_id`.

### 3.4 Corpus Schema

**File**: `clients/{CLIENT}/qa/QA_{CLIENT}_gold_v{VERSION}__q{COUNT}.json`

```json
{
  "metadata": {
    "version": "1.0",
    "client": "BFAI",
    "question_count": 458,
    "created_date": "2025-12-01"
  },
  "questions": [
    {
      "question_id": "sh_easy_001",
      "question": "What is the capital of France?",
      "question_type": "single_hop",
      "difficulty": "easy",
      "ground_truth_answer": "Paris is the capital of France.",
      "source_filenames": ["geography.pdf"],
      "expected_chunks": ["chunk_123", "chunk_456"]
    }
  ]
}
```

---

## 4. Naming Conventions

### 4.1 Baseline Files

```
baseline_{CLIENT}_v{VERSION}__{DATE}__q{COUNT}.json
```

| Component | Format | Example |
|-----------|--------|---------|
| CLIENT | UPPERCASE | `BFAI` |
| VERSION | Integer | `1`, `2`, `10` |
| DATE | YYYY-MM-DD | `2025-12-17` |
| COUNT | Integer | `458` |

**Example**: `baseline_BFAI_v1__2025-12-17__q458.json`

### 4.2 Run Folders

```
{DATE}__{MODEL}__p{PRECISION}__{RUN_ID_SUFFIX}
```

| Component | Format | Example |
|-----------|--------|---------|
| DATE | YYYY-MM-DD | `2025-12-18` |
| MODEL | Model name (slashes replaced with _) | `gemini-2.5-flash` |
| PRECISION | p{K} | `p25` |
| RUN_ID_SUFFIX | Last 8 chars of run_id | `f978a2b8` |

**Example**: `2025-12-18__gemini-2.5-flash__p25__f978a2b8`

### 4.3 Run IDs

```
run_{YYYYMMDD}_{HHMMSS}_{UUID8}
```

**Example**: `run_20251218_102744_f978a2b8`

### 4.4 Corpus Files

```
QA_{CLIENT}_gold_v{MAJOR}-{MINOR}__q{COUNT}.json
```

**Example**: `QA_BFAI_gold_v1-0__q458.json`

---

## 5. Scripts Reference

### 5.1 eval_runner.py

**Purpose**: Interactive CLI entry point with configurable defaults.

**Location**: `scripts/eval/eval_runner.py`

**Usage**:
```bash
# Interactive mode - shows defaults, prompts for changes
python scripts/eval/eval_runner.py

# Run with defaults (no prompts)
python scripts/eval/eval_runner.py --run

# Dry run - show config without running
python scripts/eval/eval_runner.py --dry-run

# Quick test (N questions)
python scripts/eval/eval_runner.py --quick 10

# Test mode (30 questions, 5 per bucket)
python scripts/eval/eval_runner.py --test

# Override settings
python scripts/eval/eval_runner.py --workers 3 --precision 12 --model gemini-3-flash-preview

# Save as new baseline
python scripts/eval/eval_runner.py --run --update-baseline

# List all baselines
python scripts/eval/eval_runner.py --list-baselines
```

**Defaults** (edit in script):
```python
DEFAULTS = {
    "client": "BFAI",
    "corpus": "QA_BFAI_gold_v1-0__q458.json",
    "job_id": "bfai__eval66a_g1_1536_tt",
    "generator_model": "gemini-2.5-flash",
    "judge_model": "gemini-2.0-flash",
    "reasoning_effort": "low",
    "precision_k": 25,
    "recall_k": 100,
    "enable_hybrid": True,
    "enable_reranking": True,
    "workers": 5,
    "temperature": 0.0,  # MUST be 0.0 for idempotency
    "quick": 0,
    "test_mode": False,
    "update_baseline": False,
}
```

### 5.2 core_eval.py

**Purpose**: Main evaluation orchestrator.

**Location**: `scripts/eval/core_eval.py`

**Key Functions**:
- `generate_run_id()` - Creates unique run identifier
- `get_run_folder()` - Generates run folder path
- `save_jsonl()` - Saves results as JSONL
- `run_evaluation()` - Main entry point

**Flow**:
1. Generate run ID
2. Load baseline for comparison
3. Load corpus
4. Run evaluation via `GoldEvaluator`
5. Calculate cost
6. Save run_summary.json and results.jsonl
7. Compare to baseline
8. Save comparison.md
9. Optionally save as new baseline

### 5.3 run_gold_eval.py

**Purpose**: Executes RAG pipeline and captures all metrics.

**Location**: `scripts/eval/run_gold_eval.py`

**Key Class**: `GoldEvaluator`

**Captured Metrics Per Question**:
- `question_id`, `question_type`, `difficulty`
- `recall_hit`, `mrr`
- `judgment` (5 dimensions + verdict)
- `timing` (retrieval, rerank, generation, judge, total)
- `tokens` (prompt, completion, thinking, cached, total)
- `llm_metadata` (model, finish_reason, temperature, etc.)
- `answer_length`, `retrieval_candidates`

**Aggregated Metrics**:
- Pass/partial/fail rates
- Recall@K, MRR
- Score averages (correctness, completeness, faithfulness, relevance, clarity)
- Latency by phase
- Token totals
- Cost calculation
- Answer length stats
- Finish reason distribution

### 5.4 baseline_manager.py

**Purpose**: Baseline CRUD operations and comparison.

**Location**: `scripts/eval/baseline_manager.py`

**Key Functions**:
- `get_baseline_path(client, version, date, count)` - Generate path
- `parse_baseline_filename(filename)` - Extract metadata from filename
- `list_baselines(client=None)` - List all baselines
- `get_latest_baseline(client)` - Load latest baseline for client
- `load_baseline(path)` - Load specific baseline
- `save_baseline(data, client, version=None, date=None)` - Save new baseline
- `compare_to_baseline(current, baseline)` - Generate comparison
- `format_comparison_report(comparison)` - Markdown report

**Comparison Logic**:
- Calculates deltas for all metrics
- Flags regressions (negative delta > threshold)
- Flags improvements (positive delta > threshold)
- Thresholds: pass_rate 2%, fail_rate 2%, mrr 2%, overall_score 0.1

### 5.5 cost_calculator.py

**Purpose**: Token → cost calculation.

**Location**: `scripts/eval/cost_calculator.py`

**Pricing Table** (per 1M tokens):
```python
PRICING = {
    "gemini-2.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "thinking": 0.30,
        "cached": 0.01875,  # 75% discount
    },
    "gemini-3-flash-preview": {
        "input": 0.10,
        "output": 0.40,
        "thinking": 0.40,
        "cached": 0.025,
    },
    "gemini-2.0-flash": {
        "input": 0.075,
        "output": 0.30,
        "thinking": 0.0,
        "cached": 0.01875,
    },
}
```

**Key Functions**:
- `get_pricing(model)` - Get pricing for model (with fallback)
- `calculate_cost(prompt_tokens, completion_tokens, ...)` - Single generation cost
- `calculate_run_cost(total_prompt_tokens, ..., question_count)` - Full run cost

---

## 6. Configuration

### 6.1 eval_config.yaml

**Location**: `configs/eval_config.yaml`

```yaml
# Eval Configuration
environment:
  type: local
  orchestrator_url: null

index:
  job_id: bfai__eval66a_g1_1536_tt

models:
  generator: gemini-2.5-flash
  judge: gemini-2.0-flash
  reasoning_effort: low
  temperature: 0.0

retrieval:
  recall_k: 100
  precision_k: 25
  enable_hybrid: true
  enable_reranking: true

corpus:
  client: BFAI
  file: QA_BFAI_gold_v1-0__q458.json

execution:
  workers: 5
  checkpoint_interval: 10

output:
  runs_dir: runs/
  baselines_dir: baselines/
  reports_dir: reports/

pricing:
  gemini-2.5-flash:
    input_per_1m: 0.075
    output_per_1m: 0.30
    thinking_per_1m: 0.30
    cached_per_1m: 0.01875
```

---

## 7. Workflow

### 7.1 First-Time Setup

1. **Create corpus file**:
   ```
   clients/{CLIENT}/qa/QA_{CLIENT}_gold_v1-0__q{COUNT}.json
   ```

2. **Create initial baseline** (from existing report or first run):
   ```
   baselines/baseline_{CLIENT}_v1__{DATE}__q{COUNT}.json
   ```

3. **Update DEFAULTS in eval_runner.py**:
   ```python
   DEFAULTS = {
       "client": "YOUR_CLIENT",
       "corpus": "QA_YOUR_CLIENT_gold_v1-0__q100.json",
       ...
   }
   ```

### 7.2 Running an Evaluation

```bash
# Option 1: Interactive
python scripts/eval/eval_runner.py

# Option 2: Quick test first
python scripts/eval/eval_runner.py --quick 10

# Option 3: Full run with defaults
python scripts/eval/eval_runner.py --run
```

### 7.3 Updating the Baseline

```bash
# Run and save as new baseline
python scripts/eval/eval_runner.py --run --update-baseline
```

This creates `baseline_{CLIENT}_v{N+1}__{TODAY}__q{COUNT}.json`

### 7.4 Comparing Runs

Every run automatically compares to the latest baseline and generates:
- Console output with deltas
- `comparison.md` in run folder

### 7.5 Reviewing Results

```bash
# List all baselines
python scripts/eval/eval_runner.py --list-baselines

# View run summary
cat runs/{RUN_FOLDER}/run_summary.json | jq .

# View per-question results
head -5 runs/{RUN_FOLDER}/results.jsonl | jq .

# View comparison
cat runs/{RUN_FOLDER}/comparison.md
```

---

## 8. Test Suite

### 8.1 Test Files

| File | Tests | Purpose |
|------|-------|---------|
| `test_cost_calculator.py` | 23 | Pricing, cost calculation, edge cases |
| `test_baseline_manager.py` | 38 | Baseline CRUD, comparison, versioning |
| `test_core_eval.py` | 22 | Run ID, folder paths, JSONL |
| `test_run_gold_eval.py` | 40 | JSON extraction, corpus loading, retry logic, breakdowns |
| `test_generate_report.py` | 39 | Report generation, MRR matrix, index/topic type, run context |
| `test_integration.py` | 11 | Component interactions |
| `test_e2e.py` | 26 | Real API calls (skipped by default) |

**Total: 199 tests**

### 8.2 Running Tests

```bash
# Unit + Integration (fast, no API calls)
python3 -m pytest tests/eval/ -v

# Specific file
python3 -m pytest tests/eval/test_cost_calculator.py -v

# End-to-End (slow, costs money)
python3 -m pytest tests/eval/ -v -m e2e
```

### 8.3 Test Categories

**Unit Tests**:
- Function-level testing
- Edge cases (zero values, negative numbers, missing keys)
- Idempotency (same input → same output)

**Integration Tests**:
- Full baseline workflow (save → load → compare → report)
- Multiple baseline versioning
- Metrics/token/latency aggregation
- Schema validation

**End-to-End Tests**:
- Real API calls with 3 questions
- Validates all output fields
- Cost calculation verification
- Output file validation

---

## 9. Replication Guide

### 9.1 To Replicate for a New Pipeline

1. **Copy the scripts**:
   ```
   scripts/eval/
   ├── eval_runner.py
   ├── core_eval.py
   ├── run_gold_eval.py      # Adapt for your pipeline
   ├── baseline_manager.py
   └── cost_calculator.py
   ```

2. **Create directories**:
   ```
   baselines/
   runs/
   configs/
   clients/{YOUR_CLIENT}/qa/
   tests/eval/
   ```

3. **Create your corpus**:
   ```json
   {
     "questions": [
       {
         "question_id": "unique_id",
         "question": "Your question text",
         "question_type": "single_hop|multi_hop",
         "difficulty": "easy|medium|hard",
         "ground_truth_answer": "Expected answer"
       }
     ]
   }
   ```

4. **Create initial baseline**:
   - Copy the baseline schema
   - Fill in your metrics (or zeros to start)
   - Save as `baseline_{CLIENT}_v1__{DATE}__q{COUNT}.json`

5. **Adapt run_gold_eval.py**:
   - Replace orchestrator imports with your pipeline
   - Ensure you capture the same metrics
   - Keep the output schema identical

6. **Update DEFAULTS in eval_runner.py**:
   ```python
   DEFAULTS = {
       "client": "YOUR_CLIENT",
       "corpus": "your_corpus.json",
       "job_id": "your_job_id",
       "generator_model": "your_model",
       ...
   }
   ```

7. **Copy and adapt tests**:
   - Update paths and expected values
   - Run tests to verify

### 9.2 Key Invariants

**MUST maintain for idempotency**:
- `temperature: 0.0` always
- Same corpus file
- Same model versions
- Same retrieval settings

**MUST capture for comparison**:
- All 5 judgment dimensions + verdict
- Timing by phase
- Token counts (prompt, completion, thinking, cached)
- LLM metadata (model, finish_reason, etc.)

**MUST follow for baseline management**:
- Filename convention: `baseline_{CLIENT}_v{N}__{DATE}__q{COUNT}.json`
- Auto-increment version on save
- Compare to latest baseline by default

---

## Appendix A: Metric Definitions

| Metric | Definition | Range |
|--------|------------|-------|
| `pass_rate` | % of questions with verdict="pass" | 0-1 |
| `partial_rate` | % of questions with verdict="partial" | 0-1 |
| `fail_rate` | % of questions with verdict="fail" | 0-1 |
| `acceptable_rate` | pass_rate + partial_rate | 0-1 |
| `recall_at_100` | % of questions where ground truth chunk in top 100 | 0-1 |
| `mrr` | Mean Reciprocal Rank of ground truth chunk | 0-1 |
| `correctness` | Factual accuracy of answer | 1-5 |
| `completeness` | Coverage of all aspects | 1-5 |
| `faithfulness` | Grounded in retrieved context | 1-5 |
| `relevance` | Addresses the question | 1-5 |
| `clarity` | Well-structured and clear | 1-5 |
| `overall_score` | Holistic quality score | 1-5 |

## Appendix B: Comparison Thresholds

| Metric | Threshold | Regression if |
|--------|-----------|---------------|
| `pass_rate` | 2% | delta < -0.02 |
| `fail_rate` | 2% | delta > +0.02 |
| `acceptable_rate` | 2% | delta < -0.02 |
| `recall_at_100` | 1% | delta < -0.01 |
| `mrr` | 2% | delta < -0.02 |
| `overall_score_avg` | 0.1 | delta < -0.1 |

## Appendix C: Cost Calculation

```
input_cost = (prompt_tokens / 1,000,000) × input_price
output_cost = (completion_tokens / 1,000,000) × output_price
thinking_cost = (thinking_tokens / 1,000,000) × thinking_price
cached_cost = (cached_tokens / 1,000,000) × cached_price

total_cost = input_cost + output_cost + thinking_cost + cached_cost
cost_per_question = total_cost / question_count
```

---

## Appendix D: Quick Reference Runbook

### Daily Core Eval (Automated, Same Every Time)

```bash
# Just run with defaults - idempotent, same every time
python scripts/eval/eval_runner.py --run
```

**What happens:**
1. Loads 458 questions from gold corpus
2. Runs full pipeline (retrieval → rerank → generation → judge)
3. Compares to latest baseline
4. Saves results to `runs/{DATE}__{MODEL}__p{K}__{ID}/`
5. Prints comparison report

### Ad-Hoc Testing (Change Parameters)

```bash
# Interactive mode - shows defaults, lets you change
python scripts/eval/eval_runner.py

# Quick test with different model
python scripts/eval/eval_runner.py --quick 10 --model gemini-3-flash-preview

# Different precision
python scripts/eval/eval_runner.py --quick 20 --precision 12

# Fewer workers (slower but gentler on quota)
python scripts/eval/eval_runner.py --workers 2

# Full run with different model, save as new baseline
python scripts/eval/eval_runner.py --run --model gemini-3-flash-preview --update-baseline

# Cloud mode
python scripts/eval/eval_runner.py --cloud --endpoint https://your-cloud-run.app --quick 10
```

### Command Quick Reference

| Use Case | Command |
|----------|---------|
| **Daily run** | `python scripts/eval/eval_runner.py --run` |
| **Quick test (10 q)** | `python scripts/eval/eval_runner.py --quick 10` |
| **Interactive** | `python scripts/eval/eval_runner.py` |
| **Different model** | `--model gemini-3-flash-preview` |
| **Different precision** | `--precision 12` |
| **Save as baseline** | `--update-baseline` |
| **Cloud mode** | `--cloud --endpoint URL` |
| **Dry run (see config)** | `--dry-run` |
| **List baselines** | `--list-baselines` |

### Changing Defaults Permanently

Edit `scripts/eval/eval_runner.py` line 29:

```python
DEFAULTS = {
    "client": "BFAI",
    "generator_model": "gemini-2.5-flash",  # <-- Change this
    "precision_k": 25,                       # <-- Or this
    "workers": 5,                            # <-- Or this
    "mode": "local",                         # <-- Or "cloud"
    "endpoint": None,                        # <-- Cloud Run URL
    ...
}
```

### Output Locations

| What | Where |
|------|-------|
| **Run results** | `runs/{DATE}__{MODEL}__p{K}__{ID}/` |
| **Per-question JSONL** | `runs/.../results.jsonl` |
| **Summary JSON** | `runs/.../run_summary.json` |
| **Comparison report** | `runs/.../comparison.md` |
| **Baselines** | `baselines/baseline_{CLIENT}_v{N}__{DATE}__q{COUNT}.json` |

### See Also

For detailed score definitions, failure archetypes, and GCS data management, see:
- `docs/EVAL_RUNBOOK.md` - Comprehensive 800-line reference guide

---

*Last updated: 2025-12-18*
