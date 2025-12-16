# E2E Orchestrator Test Report

**Generated:** 2025-12-15 17:07
**Runs:** 3

---

## 1. Test Configuration

| Parameter | Value |
|-----------|-------|
| Job ID | `bfai__eval66a_g1_1536_tt` |
| Embedding Model | gemini-embedding-001 |
| Dimensions | 1536 |
| Task Type | RETRIEVAL_QUERY |
| Hybrid Alpha | 0.5 (50/50 dense/sparse) |
| Reranking | Google Ranking API |
| Questions | 224 |

---

## 2. Consistency Analysis (Across 3 Runs)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Recall@10 | 93.3% | 0.00% | 93.3% | 93.3% |
| MRR@10 | 0.843 | 0.0000 | 0.843 | 0.843 |
| Precision@10 | 57.7% | 0.00% | 57.7% | 57.7% |
| Pass Rate | 89.0% | 0.26% | 88.8% | 89.3% |
| Overall Score | 4.43/5 | 0.009 | 4.42 | 4.44 |

---

## 3. Retrieval Metrics (Run 1)

### 3.1 Recall@K

| K | Recall |
|---|--------|
| 5 | 89.3% |
| 10 | 93.3% |
| 15 | 94.2% |
| 20 | 95.1% |
| 25 | 96.0% |
| 50 | 97.8% |
| 100 | 98.7% |

### 3.2 MRR & Precision

| Metric | Value |
|--------|-------|
| MRR@10 | 0.843 |
| Precision@10 | 57.7% |

---

## 4. LLM-as-Judge Results (Run 1)

### 4.1 Quality Scores (1-5 scale)

| Metric | Score |
|--------|-------|
| Correctness | 4.40 |
| Completeness | 4.27 |
| Faithfulness | 4.98 |
| Relevance | 4.90 |
| Clarity | 4.96 |
| Overall_score | 4.42 |

### 4.2 Verdict Distribution

| Verdict | Count |
|---------|-------|
| Pass | 167 |
| Partial | 32 |
| Fail | 25 |
| Error | 0 |

**Pass Rate:** 88.8%

---

## 5. Answer Statistics

| Metric | Value |
|--------|-------|
| Avg Answer Length | 864 chars |

---

## 6. Timing Breakdown (Run 1)

| Phase | Avg | Min | Max | Total |
|-------|-----|-----|-----|-------|
| Retrieval | 0.252s | 0.166s | 0.452s | 56.3s |
| Reranking | 0.196s | 0.091s | 1.480s | 43.9s |
| Generation | 7.742s | 1.736s | 46.486s | 1734.3s |
| Judge | 1.342s | 0.880s | 2.148s | 300.5s |
| Total | 9.532s | 3.289s | 48.539s | 2135.2s |

---

## 7. Comparison with Previous Results

### Previous LLM Judge Results (2025-12-13)

| Metric | Previous | Current | Δ |
|--------|----------|---------|---|
| Overall Score | 4.16/5 | 4.42/5 | +0.26 |
| Pass Rate | 82.6% | 88.8% | +6.2% |
| Correctness | 4.13/5 | 4.40/5 | +0.27 |
| Faithfulness | 4.95/5 | 4.98/5 | +0.03 |

---

## 8. Key Findings

1. **Consistency**: Results are highly consistent across 3 runs (Recall@10 std: 0.00%)

2. **Retrieval Quality**: Recall@10 = 93.3%, MRR@10 = 0.843

3. **Answer Quality**: Overall LLM Judge score = 4.43/5, Pass rate = 89.0%

4. **Performance**: Average total time per query = 9.53s
   - Retrieval: 0.252s
   - Reranking: 0.196s
   - Generation: 7.742s
   - LLM Judge: 1.342s
