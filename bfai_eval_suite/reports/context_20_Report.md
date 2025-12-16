# Context 20 Report

**Date:** 2025-12-16 06:20

---

## Configuration

| Setting | Value |
|---------|-------|
| Temperature | 0.0 |
| Context Size | 20 chunks |
| Questions | 224 |
| Embedding | gemini-1536-RETRIEVAL_QUERY |
| Generation Model | gemini-2.5-flash |
| Judge Model | gemini-2.5-flash (temp 0.0) |

---

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| Recall@100 | 70.1% |
| Precision@20 | 24.4% |

---

## LLM Judge Results

### Pass Rate

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | 160 | 71.4% |
| Partial | 34 | 15.2% |
| Fail | 30 | 13.4% |
| Error | 0 | 0.0% |

### Quality Scores (1-5)

| Dimension | Score |
|-----------|-------|
| Overall Score | 4.29 |
| Correctness | 4.23 |
| Completeness | 4.12 |
| Faithfulness | 4.62 |
| Relevance | 4.47 |
| Clarity | 4.92 |

---

## Timing

| Phase | Avg per Query | Total |
|-------|---------------|-------|
| Retrieval | 0.183s | 41.1s |
| Reranking | 0.000s | 0.0s |
| Generation | 7.126s | 1596.3s |
| LLM Judge | 7.509s | 1682.1s |
| **Total** | 14.819s | 3319.5s |
