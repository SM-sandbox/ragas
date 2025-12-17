# Evaluation Test Log

This log tracks all RAG evaluation runs for the BFAI Eval Suite.

**GCS Bucket:** `gs://bfai-eval-suite`

---

## Naming Conventions

### Test ID Format
`TID_XX` - Sequential test identifier

### QA Corpus Format
`QA_{client}_{tier}_v{version}__q{count}.json`

Example: `QA_BFAI_gold_v1-0__q458.json`

### GCS Path Structure
```
gs://bfai-eval-suite/
├── BFAI/                    # Client/project
│   ├── TID_01/              # Test ID
│   │   ├── data/            # Results, checkpoints
│   │   └── config.json      # Test configuration
│   ├── TID_02/
│   ...
├── ClientABC/
│   ├── TID_01/
│   ...
```

---

## Test Types

| Type | Description | Use Case |
|------|-------------|----------|
| **Core** | Standard eval on new corpus, default settings | Baseline establishment, client deployments |
| **Ad-Hoc** | Targeted investigation, modified settings | Debugging, parameter tuning, experiments |

---

## Evaluation History

### TID_01 | 2024-12-14 | Ad-Hoc | Embedding Model Comparison

**Purpose:** Compare embedding models for retrieval quality

**Config:**
- Corpus: qa_corpus_200
- Models tested: Multiple embedding variants

**Location:** `experiments/2024-12-14_embedding_model_comparison/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_01/`

---

### TID_02 | 2024-12-14 | Ad-Hoc | Embedding Dimension Test

**Purpose:** Test impact of embedding dimensions on retrieval

**Config:**
- Corpus: qa_corpus_200
- Dimensions tested: Various

**Location:** `experiments/2024-12-14_embedding_dimension_test/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_02/`

---

### TID_03 | 2024-12-14 | Ad-Hoc | Azure vs GCP Comparison

**Purpose:** Compare RAG performance across cloud providers

**Config:**
- Corpus: qa_corpus_200
- Platforms: Azure AI Search vs GCP Vertex AI

**Location:** `experiments/2024-12-14_azure_vs_gcp/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_03/`

---

### TID_04 | 2025-12-15 | Ad-Hoc | Temperature Sweep

**Purpose:** Find optimal generation temperature

**Config:**
- Corpus: qa_corpus_200
- Temperatures: 0.0, 0.1, 0.2, 0.3
- Model: gemini-2.5-flash

**Results:** Temperature 0.0 optimal for reproducibility

**Location:** `experiments/2025-12-15_temp_context_sweep/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_04/`

---

### TID_05 | 2025-12-15 | Ad-Hoc | Context Size Sweep

**Purpose:** Find optimal context window size

**Config:**
- Corpus: qa_corpus_200
- Context sizes: 5, 10, 15, 20, 25, 50, 100
- Model: gemini-2.5-flash

**Results:** P@25 optimal balance of quality vs latency

**Location:** `experiments/2025-12-15_temp_context_sweep/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_05/`

---

### TID_06 | 2025-12-15 | Ad-Hoc | Gemini 2.5 Pro Low Reasoning

**Purpose:** Test Pro model with reduced reasoning budget

**Config:**
- Corpus: qa_corpus_200
- Model: gemini-2.5-pro (low reasoning)

**Location:** `experiments/2025-12-15_temp_context_sweep/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_06/`

---

### TID_07 | 2025-12-15 | Ad-Hoc | E2E Orchestrator Consistency

**Purpose:** Verify orchestrator produces consistent results

**Config:**
- Corpus: qa_corpus_200
- Runs: 3 identical runs
- Model: gemini-2.5-flash

**Results:** High consistency across runs

**Location:** `experiments/2025-12-15_e2e_orchestrator/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_07/`

---

### TID_08 | 2025-12-16 | Core | Gold Standard P@12

**Purpose:** Baseline evaluation with Precision@12

**Config:**
- Corpus: QA_BFAI_gold_v1-0__q458.json
- Questions: 458
- Recall: 100, Precision: 12
- Model: gemini-2.5-flash
- Judge: gemini-2.5-flash

**Results:**
- Pass rate: 95.4%
- Avg score: 4.52

**Location:** `reports/gold_standard_eval/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_08/`

---

### TID_09 | 2025-12-16 | Core | Gold Standard P@25

**Purpose:** Baseline evaluation with Precision@25 (recommended config)

**Config:**
- Corpus: QA_BFAI_gold_v1-0__q458.json
- Questions: 458
- Recall: 100, Precision: 25
- Model: gemini-2.5-flash
- Judge: gemini-2.5-flash

**Results:**
- Pass rate: 96.1%
- Avg score: 4.58
- Failures: 18 (3.9%)

**Location:** `reports/gold_standard_eval/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_09/`

---

### TID_10 | 2025-12-17 | Ad-Hoc | Failure Rerun Enhanced

**Purpose:** Investigate if failures can be rescued with enhanced settings

**Config:**
- Corpus: QA_BFAI_gold_v1-0__q458.json (18 failed questions only)
- Recall: 200, Precision: 100
- Model: gemini-2.5-pro
- Judge: gemini-2.5-pro

**Results:**
- 3/18 fully fixed (17%)
- 9/18 improved to partial (50%)
- 6/18 still failing (33%)
- Latency: 5.1x increase

**Location:** `reports/gold_standard_eval/`
**GCS:** `gs://bfai-eval-suite/BFAI/TID_10/`

---

## Adding New Evaluations

After each evaluation run:

1. Assign next TID number
2. Add entry to this log with all details
3. Upload data to GCS:
   ```bash
   gcloud storage cp -r ./data gs://bfai-eval-suite/{CLIENT}/TID_XX/
   ```
4. Verify upload:
   ```bash
   gcloud storage ls gs://bfai-eval-suite/{CLIENT}/TID_XX/
   ```

---

*Last updated: 2025-12-17*
