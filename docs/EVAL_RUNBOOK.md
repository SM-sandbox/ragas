# RAG Evaluation Runbook

**Version:** 1.1  
**Last Updated:** December 18, 2025  
**Purpose:** Comprehensive guide for running RAG evaluation experiments with complete data capture.

> **Note:** For the quick-start runbook and interactive CLI, see `docs/EVAL_SYSTEM.md` Appendix D.  
> This document provides detailed reference material (score definitions, failure archetypes, GCS management).

---

## Table of Contents

1. [Test Types](#1-test-types)
2. [Pre-Flight Checklist](#2-pre-flight-checklist)
3. [Required Data Capture](#3-required-data-capture)
4. [Configuration Template](#4-configuration-template)
5. [Execution Protocol](#5-execution-protocol)
6. [Output Specifications](#6-output-specifications)
7. [Error Handling](#7-error-handling)
8. [Report Template](#8-report-template)
9. [Post-Run Checklist](#9-post-run-checklist)

---

## 1. Test Types

### 1.1 Core Tests

Standard evaluation on a new corpus with default settings. Used for:

- **Baseline establishment** - First evaluation of a new corpus
- **Client deployments** - BFAI demo suite, ClientABC, etc.
- **Regression testing** - Verify system still works after changes

**Characteristics:**

- Full corpus run (all questions)
- Default configuration (Recall@100, P@25, gemini-2.5-flash)
- Produces baseline metrics for comparison

### 1.2 Ad-Hoc Tests

Targeted investigation with modified settings. Used for:

- **Debugging failures** - Investigate why specific questions fail
- **Parameter tuning** - Find optimal settings (temperature, context size)
- **Experiments** - Compare models, embeddings, rerankers

**Characteristics:**

- May use subset of questions
- Modified configuration (different recall, precision, model, etc.)
- Hypothesis-driven with specific success criteria

### 1.3 Test ID Assignment

Every test gets a unique ID: `TID_XX`

- Assigned sequentially (TID_01, TID_02, ...)
- Logged in `EVAL_LOG.md`
- Used for GCS path: `gs://bfai-eval-suite/{CLIENT}/TID_XX/`

---

## 2. Pre-Flight Checklist

Run `scripts/preflight_check.py` before every evaluation. It verifies:

### 1.1 Authentication
- [ ] GCP ADC configured (`gcloud auth application-default login`)
- [ ] Project ID accessible
- [ ] Location/region set

### 1.2 Components
- [ ] VectorSearchRetriever initializes
- [ ] GoogleRanker initializes
- [ ] GeminiAnswerGenerator initializes
- [ ] LLM Judge (ChatVertexAI) responds

### 1.3 Data
- [ ] Corpus file exists and loads
- [ ] Corpus has required fields: `question`, `ground_truth_answer`, `source_filenames`, `difficulty`, `question_type`
- [ ] Output directory writable

### 1.4 Quick Smoke Test
- [ ] Run 3-5 questions end-to-end before full run
- [ ] Verify all phases complete without error
- [ ] Check output JSON structure

---

## 2. Required Data Capture

**EVERY evaluation run MUST capture the following data per question:**

### 2.1 Question Metadata
```json
{
  "question_id": "q_0001",
  "question": "What is the voltage rating?",
  "question_type": "single_hop|multi_hop",
  "difficulty": "easy|medium|hard",
  "source_filenames": ["doc1.pdf", "doc2.pdf"],
  "ground_truth_answer": "The voltage rating is 480V AC."
}
```

### 2.2 Retrieval Phase
```json
{
  "retrieval": {
    "time_ms": 252,
    "chunks_retrieved": 100,
    "recall_hit": true,
    "mrr": 0.843,
    "first_relevant_rank": 1,
    "top_k_docs": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
  }
}
```

### 2.3 Reranking Phase
```json
{
  "reranking": {
    "time_ms": 196,
    "input_chunks": 100,
    "output_chunks": 25,
    "precision_at_k": 0.64
  }
}
```

### 2.4 Generation Phase
```json
{
  "generation": {
    "time_ms": 7742,
    "model": "gemini-2.5-flash",
    "temperature": 0.0,
    "answer_text": "The voltage rating is 480V AC [1].",
    "answer_length_chars": 35,
    "answer_length_tokens": 12,
    "citations_count": 1
  }
}
```

### 2.5 LLM Judge Phase
```json
{
  "judge": {
    "time_ms": 1342,
    "model": "gemini-2.0-flash",
    "temperature": 0.0,
    "scores": {
      "correctness": 5,
      "completeness": 5,
      "faithfulness": 5,
      "relevance": 5,
      "clarity": 5,
      "overall_score": 5
    },
    "verdict": "pass|partial|fail",
    "explanation": "Answer is fully correct and complete."
  }
}
```

### 2.6 Totals
```json
{
  "total_time_ms": 9532,
  "success": true,
  "error": null,
  "retry_count": 0
}
```

---

## 3. Configuration Template

### 3.1 Eval Config JSON
```json
{
  "experiment_name": "gold_standard_p25",
  "description": "Gold standard eval with precision@25",
  "timestamp": "2025-12-17T08:00:00Z",
  
  "corpus": {
    "path": "corpus/qa_corpus_gold_500.json",
    "total_questions": 458,
    "sample_size": null,
    "stratified_sample": false
  },
  
  "retrieval": {
    "job_id": "bfai__eval66a_g1_1536_tt",
    "recall_top_k": 100,
    "enable_hybrid": true,
    "rrf_alpha": 0.5
  },
  
  "reranking": {
    "enabled": true,
    "precision_top_n": 25,
    "model": "google-ranking-api"
  },
  
  "generation": {
    "model": "gemini-2.5-flash",
    "temperature": 0.0,
    "max_tokens": 2048
  },
  
  "judge": {
    "model": "gemini-2.0-flash",
    "temperature": 0.0,
    "retry_count": 5,
    "retry_delay_ms": 500
  },
  
  "execution": {
    "workers": 1,
    "checkpoint_interval": 10,
    "timeout_per_question_s": 120
  },
  
  "output": {
    "results_file": "reports/gold_standard_eval/results_p25.json",
    "checkpoint_file": "reports/gold_standard_eval/checkpoint_p25.json",
    "report_file": "reports/gold_standard_eval/report_p25.md"
  }
}
```

### 3.2 Required Environment Variables
```bash
GOOGLE_CLOUD_PROJECT=civic-athlete-473921-c0
GOOGLE_CLOUD_LOCATION=us-east1
PYTHONUNBUFFERED=1
```

---

## 4. Execution Protocol

### 4.1 Before Running
1. Run pre-flight check: `python scripts/preflight_check.py`
2. Clear any stale checkpoints if starting fresh
3. Verify disk space for output files
4. Set up logging to file

### 4.2 Running the Eval

```bash
# Test run (30 questions, 5 per bucket)
python scripts/eval/run_gold_eval.py --test --precision 25

# Quick test (N questions only)
python scripts/eval/run_gold_eval.py --quick 20 --workers 5

# Full run with parallel execution (5 workers default)
python scripts/eval/run_gold_eval.py --precision 25 --workers 5

# Full run with nohup for long-running
nohup python scripts/eval/run_gold_eval.py --precision 25 --workers 5 > logs/run_p25.log 2>&1 &

# Monitor progress
watch -n 30 'cat reports/gold_standard_eval/checkpoint_p25.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d))"'
```

### 4.3 Parallel Execution

The eval script supports parallel execution via ThreadPoolExecutor:

| Workers | RPM Quota | Use Case |
|---------|-----------|----------|
| 1 | Any | Sequential, debugging |
| 5 | 60 (default) | Safe parallel, ~4x speedup |
| 15-25 | 1500 | After quota increase, ~10x speedup |

**Command line options:**

```bash
--workers N    # Number of parallel workers (default: 5)
--quick N      # Run only N questions (for testing)
```

**Rate limiting:** The script uses `tenacity` for exponential backoff:
- 5 retry attempts
- Wait: 1s → 2s → 4s → 8s → 16s (max 60s)
- Handles 429 rate limit errors automatically

**Tested performance (20 questions, 5 workers):**
- Sequential estimate: 48s
- Parallel actual: 11s
- **Speedup: 4.4x**

### 4.3 Checkpointing
- Checkpoint every N questions (default: 10)
- Checkpoint file is a JSON array of completed results
- On restart, load checkpoint and skip completed question_ids
- Final save writes both checkpoint and results file

### 4.4 Retry Logic
```python
MAX_RETRIES = 5
RETRY_DELAY_MS = 500

for attempt in range(MAX_RETRIES):
    try:
        result = call_api()
        break
    except Exception as e:
        if attempt == MAX_RETRIES - 1:
            # Log error, return partial result
            return {"error": str(e), "retry_count": attempt + 1}
        time.sleep(RETRY_DELAY_MS / 1000)
```

---

## 5. Output Specifications

### 5.1 Results JSON Structure
```json
{
  "metadata": {
    "experiment_name": "gold_standard_p25",
    "timestamp_start": "2025-12-17T08:00:00Z",
    "timestamp_end": "2025-12-17T09:15:00Z",
    "duration_minutes": 75,
    "config": { /* full config object */ }
  },
  
  "summary": {
    "total_questions": 458,
    "completed": 458,
    "errors": 0,
    "pass_rate_4plus": 0.856,
    "partial_rate_3": 0.105,
    "acceptable_rate_3plus": 0.961,
    "fail_rate_1_2": 0.039,
    "recall_at_100": 0.991,
    "mrr": 0.717,
    "avg_latency_ms": 8300,
    "scores": {
      "correctness": 4.64,
      "completeness": 4.70,
      "faithfulness": 4.91,
      "relevance": 4.95,
      "clarity": 4.98,
      "overall_score": 4.71
    }
  },
  
  "distributions": {
    "correctness": {"1": 17, "2": 9, "3": 12, "4": 47, "5": 373},
    "completeness": {"1": 15, "2": 2, "3": 8, "4": 54, "5": 379},
    /* ... other dimensions ... */
  },
  
  "by_difficulty": {
    "easy": {"count": 161, "pass_rate": 0.90, "avg_latency_ms": 6900},
    "medium": {"count": 161, "pass_rate": 0.85, "avg_latency_ms": 8300},
    "hard": {"count": 136, "pass_rate": 0.80, "avg_latency_ms": 10000}
  },
  
  "by_question_type": {
    "single_hop": {"count": 229, "pass_rate": 0.88},
    "multi_hop": {"count": 229, "pass_rate": 0.83}
  },
  
  "latency": {
    "retrieval_avg_ms": 252,
    "reranking_avg_ms": 196,
    "generation_avg_ms": 7742,
    "judge_avg_ms": 1342,
    "total_avg_ms": 9532
  },
  
  "results": [
    /* array of per-question results with full data capture */
  ]
}
```

### 5.2 Per-Question Result Structure
```json
{
  "question_id": "q_0001",
  "question": "What is the voltage rating?",
  "question_type": "single_hop",
  "difficulty": "easy",
  "ground_truth": "The voltage rating is 480V AC.",
  
  "retrieval": {
    "time_ms": 245,
    "recall_hit": true,
    "mrr": 1.0,
    "first_relevant_rank": 1
  },
  
  "reranking": {
    "time_ms": 180
  },
  
  "generation": {
    "time_ms": 6500,
    "answer_text": "The voltage rating is 480V AC [1].",
    "answer_length_chars": 35
  },
  
  "judge": {
    "time_ms": 1200,
    "scores": {
      "correctness": 5,
      "completeness": 5,
      "faithfulness": 5,
      "relevance": 5,
      "clarity": 5,
      "overall_score": 5
    },
    "verdict": "pass"
  },
  
  "total_time_ms": 8125,
  "success": true
}
```

---

## 6. Error Handling

### 6.1 Retrieval Errors
- Timeout: Retry up to 3 times with exponential backoff
- Empty results: Log warning, continue with empty context
- API error: Log error, mark question as failed

### 6.2 Generation Errors
- Rate limit (429): Wait and retry with exponential backoff
- Timeout: Retry up to 3 times
- Malformed response: Log and retry

### 6.3 Judge Errors
- JSON parse failure: Retry up to 5 times with cleaner prompt
- Rate limit: Wait and retry
- Final failure: Return partial scores (3/5 for all dimensions)

### 6.4 Checkpoint Recovery
```python
def load_checkpoint(checkpoint_file):
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            completed = {r["question_id"]: r for r in json.load(f)}
        print(f"Resuming from checkpoint: {len(completed)} done")
        return completed
    return {}
```

---

## 7. Report Template

Every evaluation MUST produce a markdown report with:

### 7.1 Header
- Experiment name and description
- Date/time
- Corpus details (size, source)
- Configuration summary

### 7.2 Executive Summary
- Pass/Partial/Fail rates
- Key metrics (Recall, MRR, Overall Score)
- Recommendation

### 7.3 Score Distributions
- Table showing count and % for scores 1-5 for each dimension
- ≥3 threshold counts

### 7.4 Latency Analysis
- Phase breakdown (retrieval, reranking, generation, judge)
- By difficulty (easy, medium, hard)
- Outlier analysis

### 7.5 Answer Analysis
- Average answer length by difficulty
- Citation counts

### 7.6 Failure Analysis
- List of failed questions with reasons
- Common failure patterns

---

## Appendix A: Score Scale Definitions

### CORRECTNESS (1-5)
| Score | Definition |
|-------|------------|
| 5 | Fully correct - All facts match ground truth exactly |
| 4 | Mostly correct - Minor omissions or slight inaccuracies |
| 3 | Partially correct - Some correct info but notable errors/gaps |
| 2 | Mostly incorrect - Major factual errors, limited correct info |
| 1 | Incorrect - Fundamentally wrong or contradicts ground truth |

### COMPLETENESS (1-5)
| Score | Definition |
|-------|------------|
| 5 | Comprehensive - Covers all key points from ground truth |
| 4 | Mostly complete - Covers most key points, minor gaps |
| 3 | Partially complete - Covers some key points, notable gaps |
| 2 | Incomplete - Missing most key points |
| 1 | Severely incomplete - Fails to address the question substantively |

### FAITHFULNESS (1-5)
| Score | Definition |
|-------|------------|
| 5 | Fully faithful - All claims supported by retrieved context |
| 4 | Mostly faithful - Minor unsupported claims |
| 3 | Partially faithful - Some hallucinated or unsupported content |
| 2 | Mostly unfaithful - Significant hallucinations |
| 1 | Unfaithful - Answer contradicts or ignores context |

### RELEVANCE (1-5)
| Score | Definition |
|-------|------------|
| 5 | Highly relevant - Directly addresses the question |
| 4 | Mostly relevant - Addresses question with minor tangents |
| 3 | Partially relevant - Some relevant content, some off-topic |
| 2 | Mostly irrelevant - Largely off-topic |
| 1 | Irrelevant - Does not address the question |

### CLARITY (1-5)
| Score | Definition |
|-------|------------|
| 5 | Excellent clarity - Well-organized, easy to understand |
| 4 | Good clarity - Clear with minor structural issues |
| 3 | Adequate clarity - Understandable but could be clearer |
| 2 | Poor clarity - Confusing or poorly organized |
| 1 | Very poor clarity - Incoherent or incomprehensible |

### OVERALL SCORE (1-5)
| Score | Definition |
|-------|------------|
| 5 | Excellent - Would fully satisfy a user's information need |
| 4 | Good - Useful answer with minor issues |
| 3 | Acceptable - Adequate but has notable shortcomings |
| 2 | Poor - Significant issues, limited usefulness |
| 1 | Unacceptable - Fails to provide useful information |

---

## Appendix B: Corpus Naming Convention

### File Naming

**Format:** `qa_corpus_{tier}_v{version}_{count}.json`

| Component | Values | Description |
|-----------|--------|-------------|
| **tier** | `silver`, `gold` | Silver = raw generated, Gold = quality-filtered |
| **version** | `v1`, `v2`, ... | Iteration number |
| **count** | `458`, `500`, ... | Number of questions |

**Example:** `qa_corpus_gold_v1_458.json`

### Question ID Format

**Format:** `{hop}_{difficulty}_{number}`

| Prefix | Meaning | Example |
|--------|---------|---------|
| `sh_easy` | Single-hop, easy | `sh_easy_001` |
| `sh_med` | Single-hop, medium | `sh_med_015` |
| `sh_hard` | Single-hop, hard | `sh_hard_042` |
| `mh_easy` | Multi-hop, easy | `mh_easy_001` |
| `mh_med` | Multi-hop, medium | `mh_med_015` |
| `mh_hard` | Multi-hop, hard | `mh_hard_042` |

### Corpus Pipeline

1. **Silver (Raw):** Generate questions from documents
2. **Score:** Rate each question for relevance (1-5)
3. **Filter:** Keep relevance >= 4 (Critical + Relevant)
4. **Gold:** Apply question ID nomenclature
5. **Log:** Record in `CORPUS_LOG.md` at project root

### Current Gold Corpus

- **File:** `corpus/qa_corpus_gold_v1_458.json`
- **Questions:** 458
- **Created:** 2025-12-16

---

## Appendix C: Failure Archetypes

When evaluations fail (score 1-2), classify failures into these archetypes for root cause analysis:

| Archetype | Description | Fix |
|-----------|-------------|-----|
| **INCOMPLETE_CONTEXT** | Retrieved chunks missing full answer | Better chunking, keep related specs together |
| **WRONG_DOCUMENT** | Relevant doc ranked poorly (low MRR) | Cross-encoder reranking |
| **HALLUCINATION** | Generated plausible but incorrect info | Stricter prompts, cite sources |
| **COMPLEX_REASONING** | Multi-step reasoning failed | Query decomposition, chain-of-thought |
| **NUMERICAL_PRECISION** | Exact numbers wrong or missing | Improve table extraction |
| **NO_FAILURE** | Judge disagreement - actually correct | Judge calibration |

### Failure Analysis Process

1. Extract all questions with overall_score 1-2
2. Run LLM archetype classifier (Gemini 2.5 Flash)
3. Group by difficulty and question type
4. Generate recommendations based on archetype distribution

### Typical Distribution (from Gold Standard v1)

- **55% retrieval-related** (INCOMPLETE_CONTEXT + WRONG_DOCUMENT)
- **17% hallucination** - addressable via prompt engineering
- **11% complex reasoning** - needs multi-hop handling
- **11% judge disagreement** - false positives

---

## Appendix C: Benchmark Reference Data

### Phase Timing (from E2E Orchestrator, n=224)

| Phase | Avg | Min | Max | % of Total |
|-------|-----|-----|-----|------------|
| Retrieval | 0.252s | 0.166s | 0.452s | 2.6% |
| Reranking | 0.196s | 0.091s | 1.480s | 2.1% |
| Generation | 7.742s | 1.736s | 46.486s | 81.2% |
| LLM Judge | 1.342s | 0.880s | 2.148s | 14.1% |
| Total | 9.532s | 3.289s | 48.539s | 100% |

### Answer Statistics
- Average answer length: 864 chars
- Typical citation count: 1-3

---

## Appendix C: Quick Reference Commands

### Using eval_runner.py (Recommended)

```bash
# Interactive mode - shows defaults, lets you change
python scripts/eval/eval_runner.py

# Run with defaults (no prompts)
python scripts/eval/eval_runner.py --run

# Quick test (10 questions)
python scripts/eval/eval_runner.py --quick 10

# Dry run - see config without running
python scripts/eval/eval_runner.py --dry-run

# Different model
python scripts/eval/eval_runner.py --quick 10 --model gemini-3-flash-preview

# Save as new baseline
python scripts/eval/eval_runner.py --run --update-baseline

# Cloud mode
python scripts/eval/eval_runner.py --cloud --endpoint https://your-cloud-run.app --quick 10
```

### Using run_gold_eval.py (Direct)

```bash
# Pre-flight check
python scripts/preflight_check.py

# Test run (30 questions)
python scripts/eval/run_gold_eval.py --test --precision 25

# Full run (background)
nohup python scripts/eval/run_gold_eval.py --precision 25 > logs/run.log 2>&1 &

# Monitor progress
cat reports/gold_standard_eval/checkpoint_p25.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} done')"

# Check if running
ps aux | grep run_gold | grep -v grep

# Kill if stuck
pkill -f run_gold_eval
```

---

## 9. Post-Run Checklist

**EVERY evaluation run MUST complete these steps:**

### 9.1 Assign Test ID

1. Check `EVAL_LOG.md` for the last TID number
2. Assign next sequential TID (e.g., TID_11)
3. Create test directory: `clients/{CLIENT}/tests/TID_XX_YYYY-MM-DD_description/`

### 9.2 Organize Results

1. Move results to test directory:

   ```bash
   mkdir -p clients/BFAI/tests/TID_XX_YYYY-MM-DD_description/data
   cp results.json checkpoint.json clients/BFAI/tests/TID_XX_YYYY-MM-DD_description/data/
   ```

### 9.3 Upload to GCS

1. Upload data folder:

   ```bash
   gcloud storage cp -r clients/BFAI/tests/TID_XX_*/ gs://bfai-eval-suite/BFAI/tests/TID_XX/
   ```

2. Verify upload succeeded:

   ```bash
   gcloud storage ls gs://bfai-eval-suite/BFAI/tests/TID_XX/
   ```

### 9.4 Update EVAL_LOG.md

Add entry to `EVAL_LOG.md` with:

- TID number and date
- Test type (Core/Ad-Hoc)
- Purpose and hypothesis
- Configuration details
- Key results
- GCS location

### 9.5 Update GCS_MANIFEST.md

Add new TID to `clients/{CLIENT}/GCS_MANIFEST.md`

### 9.6 Verification Checklist

- [ ] Test directory created with data/
- [ ] Results uploaded to GCS
- [ ] GCS upload verified
- [ ] EVAL_LOG.md updated
- [ ] GCS_MANIFEST.md updated
- [ ] Report generated (if applicable)

---

## 10. Data Management

### 10.1 Directory Structure

```
bfai_eval_suite/
├── clients/                     # All client data
│   └── BFAI/                    # Client folder
│       ├── corpus/              # Foundational data [GITIGNORED]
│       │   ├── documents/       # Source PDFs
│       │   ├── metadata/        # Per-doc analysis
│       │   ├── chunks/          # Chunked text
│       │   └── knowledge_graph.json
│       ├── qa/                  # QA test sets [IN REPO]
│       │   ├── QA_BFAI_gold_v1-0__q458.json
│       │   └── archive/
│       ├── tests/               # Evaluation runs
│       │   └── TID_XX/data/     # [GITIGNORED]
│       └── GCS_MANIFEST.md      # Pointer to GCS
```

### 10.2 What Goes Where

| Data Type | Local | Git | GCS |
|-----------|-------|-----|-----|
| Source PDFs | ✅ | ❌ | ✅ |
| Doc metadata | ✅ | ❌ | ✅ |
| Chunks | ✅ | ❌ | ✅ |
| Knowledge graph | ✅ | ❌ | ✅ |
| QA JSON files | ✅ | ✅ | ✅ |
| Test results | ✅ | ❌ | ✅ |
| GCS_MANIFEST.md | ✅ | ✅ | ❌ |

### 10.3 GCS Bucket Structure

**Bucket:** `gs://bfai-eval-suite`

```
gs://bfai-eval-suite/
├── BFAI/
│   ├── corpus/
│   │   ├── documents/
│   │   ├── metadata/
│   │   ├── chunks/
│   │   └── knowledge_graph.json
│   ├── qa/
│   │   └── QA_BFAI_gold_v1-0__q458.json
│   └── tests/
│       ├── TID_01/
│       └── ...
├── ClientABC/
│   └── ...
```

### 10.4 Sync Commands

```bash
# Upload corpus to GCS
gcloud storage cp -r clients/BFAI/corpus/ gs://bfai-eval-suite/BFAI/corpus/

# Download corpus from GCS (new machine setup)
gcloud storage cp -r gs://bfai-eval-suite/BFAI/corpus/ clients/BFAI/corpus/

# Upload test results
gcloud storage cp -r clients/BFAI/tests/TID_XX/ gs://bfai-eval-suite/BFAI/tests/TID_XX/

# Verify
gcloud storage ls gs://bfai-eval-suite/BFAI/
```

---

*This runbook is the authoritative reference for all RAG evaluation experiments.*
