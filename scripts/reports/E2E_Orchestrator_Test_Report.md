# E2E Orchestrator Test Report

**Generated:** 2025-12-18 08:48
**Runs:** 1

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
| Questions | 10 |

---

## 2. Consistency Analysis (Across 1 Runs)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Recall@10 | 100.0% | 0.00% | 100.0% | 100.0% |
| MRR@10 | 1.000 | 0.0000 | 1.000 | 1.000 |
| Precision@10 | 100.0% | 0.00% | 100.0% | 100.0% |
| Pass Rate | 100.0% | 0.00% | 100.0% | 100.0% |
| Overall Score | 4.90/5 | 0.000 | 4.90 | 4.90 |

---

## 3. Retrieval Metrics (Run 1)

### 3.1 Recall@K

| K | Recall |
|---|--------|
| 5 | 100.0% |
| 10 | 100.0% |
| 15 | 100.0% |
| 20 | 100.0% |
| 25 | 100.0% |
| 50 | 100.0% |
| 100 | 100.0% |

### 3.2 MRR & Precision

| Metric | Value |
|--------|-------|
| MRR@10 | 1.000 |
| Precision@10 | 100.0% |

---

## 4. LLM-as-Judge Results (Run 1)

### 4.1 Quality Scores (1-5 scale)

| Metric | Score |
|--------|-------|
| Correctness | 4.80 |
| Completeness | 4.80 |
| Faithfulness | 5.00 |
| Relevance | 5.00 |
| Clarity | 5.00 |
| Overall_score | 4.90 |

### 4.2 Verdict Distribution

| Verdict | Count |
|---------|-------|
| Pass | 9 |
| Partial | 1 |
| Fail | 0 |
| Error | 0 |

**Pass Rate:** 100.0%

---

## 5. Answer Statistics

| Metric | Value |
|--------|-------|
| Avg Answer Length | 197 chars |

---

## 6. Timing Breakdown (Run 1)

| Phase | Avg | Min | Max | Total |
|-------|-----|-----|-----|-------|
| Retrieval | 0.743s | 0.198s | 1.187s | 7.4s |
| Reranking | 0.914s | 0.191s | 1.688s | 9.1s |
| Generation | 5.981s | 3.000s | 8.683s | 59.8s |
| Judge | 1.512s | 1.094s | 2.279s | 15.1s |
| Total | 9.151s | 6.774s | 12.721s | 91.5s |

---

## 7. Comparison with Previous Results

### Previous LLM Judge Results (2025-12-13)

| Metric | Previous | Current | Δ |
|--------|----------|---------|---|
| Overall Score | 4.16/5 | 4.90/5 | +0.74 |
| Pass Rate | 82.6% | 100.0% | +17.4% |
| Correctness | 4.13/5 | 4.80/5 | +0.67 |
| Faithfulness | 4.95/5 | 5.00/5 | +0.05 |

---

## 8. Key Findings

1. **Consistency**: Results are highly consistent across 1 runs (Recall@10 std: 0.00%)

2. **Retrieval Quality**: Recall@10 = 100.0%, MRR@10 = 1.000

3. **Answer Quality**: Overall LLM Judge score = 4.90/5, Pass rate = 100.0%

4. **Performance**: Average total time per query = 9.15s
   - Retrieval: 0.743s
   - Reranking: 0.914s
   - Generation: 5.981s
   - LLM Judge: 1.512s
