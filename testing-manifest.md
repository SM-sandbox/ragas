# Testing & Evaluation Manifest

**Date:** 2025-12-19  
**Status:** SPECIFICATION ONLY - No implementation  
**Purpose:** Define what evaluations and tests SHOULD exist after reorg

---

## Table of Contents

1. [Evaluation Strategy Overview](#1-evaluation-strategy-overview)
2. [Baseline Evaluations (Scheduled)](#2-baseline-evaluations-scheduled)
3. [CI/CD Evaluations](#3-cicd-evaluations)
4. [Ad Hoc Evaluation Framework](#4-ad-hoc-evaluation-framework)
5. [Evaluation Inventory Matrix](#5-evaluation-inventory-matrix)
6. [Unit Test Specifications](#6-unit-test-specifications)
7. [Integration Tests](#7-integration-tests)
8. [Non-Functional Checks](#8-non-functional-checks)
9. [Execution Plan](#9-execution-plan)
10. [How to Run](#10-how-to-run)

---

## 1. Evaluation Strategy Overview

### Purpose of Each Execution Mode

| Mode | Purpose | Trigger | Failure Semantics |
|------|---------|---------|-------------------|
| **Baseline** | Establish and track system performance over time | Daily cron / Cloud Run job | Informational - alerts on regression |
| **CI/CD** | Gate code/config changes on quality thresholds | PR merge, config change | Blocking - fails pipeline on regression |
| **Ad Hoc** | Explore new models, prompts, retrievers, datasets | Manual | Informational - comparison only |

### Stability vs Sensitivity Goals

| Metric | Stability Goal | Sensitivity Goal |
|--------|----------------|------------------|
| **Pass Rate** | ±2% variance acceptable | Detect 3%+ regression |
| **Fail Rate** | ±1% variance acceptable | Detect 2%+ increase |
| **MRR** | ±2% variance acceptable | Detect 3%+ regression |
| **Latency** | ±20% variance acceptable | Detect 50%+ increase |
| **Cost** | ±10% variance acceptable | Detect 25%+ increase |

### Failure Semantics

**What breaks CI:**
- Pass rate drops >3% from baseline
- Fail rate increases >2% from baseline
- Any test errors (non-evaluation failures)

**What is informational only:**
- Latency changes (logged, not blocking)
- Cost changes (logged, not blocking)
- Score distribution shifts within thresholds

---

## 2. Baseline Evaluations (Scheduled)

### BASELINE-001: Daily Gold Standard Evaluation

**What it measures:**
- End-to-end RAG pipeline quality on gold corpus
- 5 judgment dimensions: correctness, completeness, faithfulness, relevance, clarity
- Retrieval quality: Recall@100, MRR

**Datasets used:**
- `clients/BFAI/qa/QA_BFAI_gold_v1-0__q458.json` (458 questions)

**Metrics and thresholds:**

| Metric | Baseline Target | Alert Threshold |
|--------|-----------------|-----------------|
| Pass Rate (≥4) | 92% | <89% |
| Acceptable Rate (≥3) | 98% | <95% |
| Fail Rate (1-2) | <2% | >5% |
| Recall@100 | 99% | <97% |
| MRR | 0.72 | <0.68 |
| Overall Score | 4.7 | <4.5 |

**Expected stability:**
- Day-to-day variance: <1% (temperature=0.0)
- Week-to-week variance: <2%

**Alerting criteria:**
- Slack notification on any threshold breach
- Email on consecutive failures (2+ days)
- PagerDuty on critical regression (pass_rate <85%)

**Schedule:**
- Daily at 06:00 UTC
- Full corpus (458 questions)
- ~60 minutes runtime

---

### BASELINE-002: Weekly Cost & Latency Benchmark

**What it measures:**
- Token consumption patterns
- Latency by phase (retrieval, rerank, generation, judge)
- Cost per question trends

**Datasets used:**
- Same as BASELINE-001

**Metrics and thresholds:**

| Metric | Baseline Target | Alert Threshold |
|--------|-----------------|-----------------|
| Avg Latency | 8.5s | >12s |
| Generation Latency | 7.5s | >10s |
| Cost per Question | $0.0004 | >$0.001 |
| Total Run Cost | $0.18 | >$0.50 |

**Schedule:**
- Weekly on Sunday at 02:00 UTC
- Full corpus

---

## 3. CI/CD Evaluations

### CICD-001: Quick Regression Check

**Trigger condition:**
- Any PR to `main` branch
- Any change to `lib/`, `evaluations/`, `config/`

**Scope:**
- Reduced dataset: 30 questions (5 per bucket)
- Stratified sample ensures coverage

**Pass/fail thresholds:**

| Metric | Pass | Fail |
|--------|------|------|
| Pass Rate | ≥88% | <85% |
| Fail Rate | ≤5% | >8% |
| Errors | 0 | >0 |

**Runtime constraints:**
- Maximum: 5 minutes
- Target: 3 minutes

**Artifacts produced:**
- `ci_results.json` - Full results
- `ci_summary.txt` - One-line pass/fail
- Exit code: 0 (pass) or 1 (fail)

---

### CICD-002: Config Change Validation

**Trigger condition:**
- Any change to `config/eval_config.yaml`
- Any change to baseline files

**Scope:**
- 10 questions (quick sanity check)
- Validates config loads correctly
- Validates baseline comparison works

**Pass/fail thresholds:**
- Config loads without error
- Baseline comparison produces valid output
- No Python exceptions

**Runtime constraints:**
- Maximum: 2 minutes

---

### CICD-003: Import Validation

**Trigger condition:**
- Any PR (always runs)

**Scope:**
- Validate all imports resolve
- No circular dependencies
- All entry points executable

**Pass/fail thresholds:**
- All imports succeed
- Entry points show help without error

**Runtime constraints:**
- Maximum: 30 seconds

---

## 4. Ad Hoc Evaluation Framework

### Supported Dimensions

| Dimension | Parameter | Values |
|-----------|-----------|--------|
| **Model** | `--model` | gemini-2.5-flash, gemini-3-flash-preview, gemini-2.5-pro |
| **Reasoning** | `--reasoning` | low, medium, high |
| **Precision** | `--precision` | 5, 10, 12, 15, 20, 25, 50 |
| **Recall** | `--recall` | 50, 100, 200 |
| **Dataset** | `--corpus` | gold_v1, gold_v2, custom |
| **Environment** | `--mode` | local, cloud |

### Parameterization Strategy

```bash
# Model comparison
python evaluations/adhoc/run_adhoc.py --quick 50 --model gemini-3-flash-preview
python evaluations/adhoc/run_adhoc.py --quick 50 --model gemini-2.5-flash

# Precision sweep
for p in 5 10 15 20 25; do
  python evaluations/adhoc/run_adhoc.py --quick 30 --precision $p
done

# Reasoning effort comparison
python evaluations/adhoc/run_adhoc.py --quick 30 --reasoning low
python evaluations/adhoc/run_adhoc.py --quick 30 --reasoning high
```

### Expected Outputs

| Output | Location | Format |
|--------|----------|--------|
| Run Summary | `runs/{date}__{model}__p{k}__{id}/run_summary.json` | JSON |
| Per-Question Results | `runs/{date}__{model}__p{k}__{id}/results.jsonl` | JSONL |
| Baseline Comparison | `runs/{date}__{model}__p{k}__{id}/comparison.md` | Markdown |

### Comparison Modes

1. **vs Baseline** - Default, compares to latest baseline
2. **vs Previous Run** - Compare two specific runs
3. **A/B Test** - Side-by-side on same questions

---

## 5. Evaluation Inventory Matrix

| Evaluation | Mode | Components Used | Metrics | Thresholds | Missing Coverage | Priority |
|------------|------|-----------------|---------|------------|------------------|----------|
| **BASELINE-001** | Baseline | evaluator, baseline_manager, cost_calculator | pass_rate, mrr, recall | 92%/0.72/99% | None | P0 |
| **BASELINE-002** | Baseline | evaluator, cost_calculator | latency, cost | 8.5s/$0.0004 | None | P1 |
| **CICD-001** | CI/CD | evaluator, baseline_manager | pass_rate, fail_rate | 88%/5% | **Not implemented** | P0 |
| **CICD-002** | CI/CD | config loader, baseline_manager | config_valid | boolean | **Not implemented** | P1 |
| **CICD-003** | CI/CD | import checker | imports_valid | boolean | **Not implemented** | P1 |
| **ADHOC-MODEL** | Ad Hoc | evaluator | all | comparison | None | P2 |
| **ADHOC-PRECISION** | Ad Hoc | evaluator | all | comparison | None | P2 |
| **ADHOC-REASONING** | Ad Hoc | evaluator | all | comparison | None | P2 |

### Missing Coverage Summary

| Gap | Description | Priority |
|-----|-------------|----------|
| CI/CD entry point | `evaluations/cicd/run_cicd_eval.py` not implemented | P0 |
| Import validation | No automated import check | P1 |
| Config validation | No config schema validation | P1 |
| Failure archetype analysis | Manual process, should be automated | P2 |

---

## 6. Unit Test Specifications

### Existing Tests (199 total)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/eval/test_baseline_manager.py` | 38 | Baseline CRUD, comparison, versioning |
| `tests/eval/test_cost_calculator.py` | 23 | Pricing, cost calculation, edge cases |
| `tests/eval/test_core_eval.py` | 22 | Run ID, folder paths, JSONL |
| `tests/eval/test_run_gold_eval.py` | 40 | JSON extraction, corpus loading, retry |
| `tests/eval/test_generate_report.py` | 39 | Report generation, MRR matrix |
| `tests/eval/test_integration.py` | 11 | Component interactions |
| `tests/eval/test_e2e.py` | 26 | Real API calls (skipped by default) |

### Required New Unit Tests

#### lib/core/evaluator.py

| Test | Description | Priority |
|------|-------------|----------|
| `test_load_corpus_full` | Load full corpus, verify count | P1 |
| `test_load_corpus_test_mode` | Load test mode, verify 30 questions | P1 |
| `test_extract_json_clean` | Extract JSON from clean response | P0 |
| `test_extract_json_markdown` | Extract JSON from markdown block | P0 |
| `test_extract_json_nested` | Extract nested JSON | P1 |
| `test_gold_evaluator_init_local` | Initialize in local mode | P1 |
| `test_gold_evaluator_init_cloud` | Initialize in cloud mode | P1 |
| `test_process_question_success` | Process single question | P1 |
| `test_process_question_retry` | Retry on failure | P1 |
| `test_aggregate_metrics` | Aggregate results correctly | P0 |

#### lib/core/baseline_manager.py

| Test | Description | Priority |
|------|-------------|----------|
| `test_parse_filename_valid` | Parse valid baseline filename | P0 |
| `test_parse_filename_invalid` | Reject invalid filename | P0 |
| `test_list_baselines_empty` | Handle empty baselines dir | P1 |
| `test_list_baselines_filtered` | Filter by client | P1 |
| `test_compare_regression` | Detect regression correctly | P0 |
| `test_compare_improvement` | Detect improvement correctly | P0 |
| `test_auto_increment_version` | Auto-increment baseline version | P1 |

#### lib/clients/gemini_client.py

| Test | Description | Priority |
|------|-------------|----------|
| `test_get_api_key_cached` | API key caching works | P1 |
| `test_generate_success` | Basic generation works | P1 |
| `test_generate_json_valid` | JSON generation works | P1 |
| `test_generate_for_judge` | Judge-specific config applied | P0 |
| `test_model_info` | Model info returned correctly | P1 |

#### lib/utils/metrics.py

| Test | Description | Priority |
|------|-------------|----------|
| `test_normalize_doc_name` | Document name normalization | P0 |
| `test_is_relevant_doc` | Relevance matching | P0 |
| `test_compute_recall_at_k` | Recall calculation | P0 |
| `test_compute_mrr` | MRR calculation | P0 |
| `test_compute_precision_at_k` | Precision calculation | P0 |

---

## 7. Integration Tests

### INT-001: End-to-End Evaluation Run

**Description:** Run complete evaluation on tiny fixture (3 questions)

**Components tested:**
- Corpus loading
- Retrieval (mocked or real)
- Generation
- Judging
- Result aggregation
- Baseline comparison
- Report generation

**Fixture:** `tests/fixtures/tiny_corpus.json` (3 questions)

**Assertions:**
- All 3 questions processed
- No errors in results
- Metrics calculated correctly
- Comparison report generated

---

### INT-002: Baseline Workflow

**Description:** Full baseline save → load → compare cycle

**Components tested:**
- `save_baseline()`
- `load_baseline()`
- `compare_to_baseline()`
- `format_comparison_report()`

**Assertions:**
- Baseline saved with correct filename
- Baseline loads with all fields
- Comparison detects known delta
- Report contains expected sections

---

### INT-003: Cross-Mode Consistency

**Description:** Same questions produce same results in local vs cloud mode

**Components tested:**
- Local evaluator
- Cloud evaluator (mocked endpoint)

**Assertions:**
- Same questions → same judgments
- Same metrics within tolerance
- Same report structure

---

### INT-004: Failure Propagation

**Description:** Errors propagate correctly without crashing

**Components tested:**
- API timeout handling
- Invalid response handling
- Checkpoint recovery

**Assertions:**
- Partial results saved on error
- Checkpoint allows resume
- Error logged with context

---

## 8. Non-Functional Checks

### Runtime Budgets

| Evaluation | Target | Maximum | Action if Exceeded |
|------------|--------|---------|-------------------|
| CICD-001 (30 questions) | 3 min | 5 min | Fail CI |
| BASELINE-001 (458 questions) | 45 min | 90 min | Alert |
| Ad Hoc (50 questions) | 8 min | 15 min | Warning |

### Determinism and Reproducibility

**Requirements:**
- `temperature=0.0` for all evaluations
- Same corpus version → same results (within API variance)
- Run ID uniquely identifies each run

**Validation:**
- Run same evaluation twice
- Compare results: should be identical or within 1% variance

### Backward Compatibility

**Requirements:**
- Old baseline files loadable by new code
- Old result files parseable
- CLI arguments backward compatible

**Validation:**
- Load baseline v1 with current code
- Parse results from TID_01 through TID_10
- Run with old CLI syntax

---

## 9. Execution Plan

### Phase 1: Critical Path (Week 1)

| Task | Size | Dependencies |
|------|------|--------------|
| Implement `evaluations/cicd/run_cicd_eval.py` | M | None |
| Add CICD-001 to GitHub Actions | S | run_cicd_eval.py |
| Create `tests/fixtures/tiny_corpus.json` | S | None |
| Add INT-001 integration test | M | tiny_corpus.json |

### Phase 2: Test Coverage (Week 2)

| Task | Size | Dependencies |
|------|------|--------------|
| Add unit tests for evaluator.py | L | None |
| Add unit tests for baseline_manager.py | M | None |
| Add unit tests for gemini_client.py | M | None |
| Add INT-002 baseline workflow test | M | None |

### Phase 3: Robustness (Week 3)

| Task | Size | Dependencies |
|------|------|--------------|
| Add CICD-002 config validation | S | None |
| Add CICD-003 import validation | S | None |
| Add INT-003 cross-mode consistency | M | None |
| Add INT-004 failure propagation | M | None |

### Phase 4: Automation (Week 4)

| Task | Size | Dependencies |
|------|------|--------------|
| Set up daily baseline cron job | M | BASELINE-001 working |
| Add Slack alerting | S | Cron job |
| Add cost tracking dashboard | M | BASELINE-002 working |
| Document runbooks for alerts | S | Alerting |

### Size Legend

- **S** = Small (< 2 hours)
- **M** = Medium (2-8 hours)
- **L** = Large (1-2 days)

---

## 10. How to Run

### Baseline Evaluations

```bash
# Daily baseline (full corpus)
python evaluations/baseline/run_baseline.py --client BFAI

# With baseline update
python evaluations/baseline/run_baseline.py --client BFAI --update-baseline

# Quick test (10 questions)
python evaluations/baseline/run_baseline.py --client BFAI --quick 10
```

### CI/CD Evaluations

```bash
# Quick regression check (30 questions)
python evaluations/cicd/run_cicd_eval.py --quick 30

# Config validation only
python evaluations/cicd/run_cicd_eval.py --validate-config

# Import check only
python evaluations/cicd/run_cicd_eval.py --check-imports
```

### Ad Hoc Evaluations

```bash
# Interactive mode
python evaluations/adhoc/run_adhoc.py

# Quick test with model override
python evaluations/adhoc/run_adhoc.py --quick 10 --model gemini-3-flash-preview

# Test mode (30 questions, 5 per bucket)
python evaluations/adhoc/run_adhoc.py --test

# Dry run (show config only)
python evaluations/adhoc/run_adhoc.py --dry-run

# List available baselines
python evaluations/adhoc/run_adhoc.py --list-baselines
```

### Unit Tests

```bash
# All unit tests (fast, no API calls)
python -m pytest tests/ -v --ignore=tests/eval/test_e2e.py

# Specific test file
python -m pytest tests/eval/test_baseline_manager.py -v

# With coverage
python -m pytest tests/ -v --cov=lib --cov-report=html
```

### Integration Tests

```bash
# Integration tests only
python -m pytest tests/integration/ -v

# E2E tests (slow, costs money)
python -m pytest tests/eval/test_e2e.py -v -m e2e
```

### Verification Commands

```bash
# Verify imports work
python -c "from lib.core.evaluator import GoldEvaluator; print('OK')"
python -c "from lib.core.baseline_manager import list_baselines; print(list_baselines())"

# Verify entry points
python evaluations/baseline/run_baseline.py --help
python evaluations/adhoc/run_adhoc.py --help

# Verify config loads
python -c "import yaml; print(yaml.safe_load(open('config/eval_config.yaml')))"
```

---

## Appendix A: Test Fixtures Required

| Fixture | Location | Contents |
|---------|----------|----------|
| `tiny_corpus.json` | `tests/fixtures/` | 3 questions (1 per difficulty) |
| `sample_baseline.json` | `tests/fixtures/` | Valid baseline for testing |
| `sample_results.jsonl` | `tests/fixtures/` | 3 result records |
| `mock_retrieval.json` | `tests/fixtures/` | Mock retrieval responses |
| `mock_generation.json` | `tests/fixtures/` | Mock generation responses |
| `mock_judgment.json` | `tests/fixtures/` | Mock judge responses |

---

## Appendix B: GitHub Actions Workflow (Specification)

```yaml
# .github/workflows/ci.yml (TO BE IMPLEMENTED)
name: CI

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run ruff
        run: ruff check .

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run unit tests
        run: pytest tests/ -v --ignore=tests/eval/test_e2e.py

  eval:
    runs-on: ubuntu-latest
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - name: Run CI evaluation
        run: python evaluations/cicd/run_cicd_eval.py --quick 30
        env:
          GOOGLE_APPLICATION_CREDENTIALS: ${{ secrets.GCP_SA_KEY }}
```

---

## Appendix C: Alerting Rules (Specification)

| Alert | Condition | Channel | Severity |
|-------|-----------|---------|----------|
| Baseline Regression | pass_rate < 89% | Slack #eval-alerts | Warning |
| Critical Regression | pass_rate < 85% | PagerDuty | Critical |
| Consecutive Failures | 2+ days below threshold | Email | Warning |
| Cost Spike | cost > 2x baseline | Slack #eval-alerts | Info |
| Latency Spike | latency > 2x baseline | Slack #eval-alerts | Info |

---

*This manifest defines WHAT should exist. Implementation is a separate effort.*
