# Context 25 Report

**Date:** 2025-12-16 07:15

---

## Configuration

| Setting | Value |
|---------|-------|
| Temperature | 0.0 |
| Context Size | 25 chunks |
| Questions | 224 |
| Embedding | gemini-1536-RETRIEVAL_QUERY |
| Generation Model | gemini-2.5-flash |
| Judge Model | gemini-2.5-flash (temp 0.0) |

---

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| Recall@100 | 70.1% |
| Precision@25 | 22.9% |

---

## LLM Judge Results

### Pass Rate

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | 165 | 73.7% |
| Partial | 32 | 14.3% |
| Fail | 27 | 12.1% |
| Error | 0 | 0.0% |

### Quality Scores (1-5)

| Dimension | Score |
|-----------|-------|
| Overall Score | 4.38 |
| Correctness | 4.37 |
| Completeness | 4.23 |
| Faithfulness | 4.72 |
| Relevance | 4.56 |
| Clarity | 4.92 |

---

## Timing

| Phase | Avg per Query | Total |
|-------|---------------|-------|
| Retrieval | 0.183s | 41.1s |
| Reranking | 0.000s | 0.0s |
| Generation | 7.392s | 1655.8s |
| LLM Judge | 7.508s | 1681.7s |
| **Total** | 15.083s | 3378.6s |
