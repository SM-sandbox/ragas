# Context Size 100 Experiment Report

**Generated:** 2025-12-16 10:55:52
**Duration:** 2h 8m 16s

---

## 1. Configuration

### Experiment Settings

| Setting | Value |
|---------|-------|
| Experiment Name | Context Size 100 Experiment |
| Model | gemini-2.5-flash |
| Temperature | 0.0 |
| Context Size | 100 chunks |
| Embedding Model | gemini-1536-RETRIEVAL_QUERY |
| Job ID | bfai__eval66a_g1_1536_tt |
| Corpus | qa_corpus_200 (448 questions) |
| Recall Top-K | 100 |
| Reranker | semantic-ranker-default@latest |
| Judge Model | gemini-2.5-flash (temp 0.0) |

### Model Details

| Property | Value |
|----------|-------|
| Model ID | gemini-2.5-flash |
| Display Name | Gemini 2.5 Flash |
| Family | gemini-2.5 |
| Version | stable |
| Status | ga |
| Cost Tier | low |
| Supports Thinking | Yes |
| Thinking Config Type | budget |
| Can Disable Thinking | Yes |
| Thinking Budget Range | 0 - 24576 |
| Max Input Tokens | 1,048,576 |
| Max Output Tokens | 65,535 |

---

## Pre-flight Checks

**Timestamp:** 2025-12-16T10:55:47.832346
**Duration:** 4619ms

| Check | Status | Message |
|-------|--------|---------|
| GCP Auth | ✅ pass | GCP authentication valid |
| Project Access | ✅ pass | Access to project civic-athlete-473921-c0 confirmed |
| Orchestrator Import | ✅ pass | All orchestrator modules imported successfully |
| Model Registry | ✅ pass | Found 8 approved models |
| Job Config | ✅ pass | Job 'bfai__eval66a_g1_1536_tt' found |
| Corpus File | ✅ pass | Corpus loaded: 224 items |
| Model Validation | ✅ pass | Model 'gemini-2.5-flash' is valid |
| Metrics Data | ✅ pass | All required fields present for metrics |
| API Connectivity | ⏭️ skip | Skipped (skip_api_check=True) |

---

## 3. Retrieval Metrics

| Metric | Value |
|--------|-------|
| Recall@100 | 98.1% |
| Precision@5 | 48.5% |
| Precision@10 | 40.6% |
| Precision@15 | 36.8% |
| Precision@20 | 34.0% |
| Precision@25 | 32.0% |
| MRR@10 | 0.715 |

---

## 4. LLM Judge Results

### Verdict Distribution

| Verdict | Count | Percentage |
|---------|-------|------------|
| ✅ Pass | 342 | 76.3% |
| ⚠️ Partial | 66 | 14.7% |
| ❌ Fail | 40 | 8.9% |
| 🔴 Error | 0 | 0.0% |

**Pass Rate: 76.3%**

### Quality Scores (1-5 scale)

| Dimension | Score |
|-----------|-------|
| **Overall Score** | **4.44** |
| Correctness | 4.45 |
| Completeness | 4.29 |
| Faithfulness | 4.73 |
| Relevance | 4.60 |
| Clarity | 4.94 |

---

## 5. Answer Length

| Metric | Value |
|--------|-------|
| Avg Characters | 748 |
| Min Characters | 58 |
| Max Characters | 6,308 |
| Avg Words | 111 |
| Min Words | 9 |
| Max Words | 909 |

---

## 7. Timing Breakdown

| Phase | Avg | Min | Max | Total |
|-------|-----|-----|-----|-------|
| Retrieval | 0.2s | 0.1s | 0.5s | 1m 22s |
| Reranking | 0.0s | 0.0s | 0.0s | 0.0s |
| Generation | 8.9s | 2.3s | 39.3s | 1h 6m 40s |
| Judge | 8.1s | 2.7s | 61.3s | 1h 0m 13s |
| Total | 17.2s | 5.5s | 66.2s | 2h 8m 16s |

**Throughput:** 0.06 questions/second

---

## 8. Summary

- **Pass Rate:** 76.3%
- **Overall Score:** 4.44/5
- **Avg Generation Time:** 8.93s
- **Total Duration:** 2h 8m 16s
