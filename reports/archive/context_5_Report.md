# Context 5 Report

**Date:** 2025-12-16 03:33

---

## Configuration

| Setting | Value |
|---------|-------|
| Temperature | 0.0 |
| Context Size | 5 chunks |
| Questions | 224 |
| Embedding | gemini-1536-RETRIEVAL_QUERY |
| Generation Model | gemini-2.5-flash |
| Judge Model | gemini-2.5-flash (temp 0.0) |

---

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| Recall@100 | 70.1% |
| Precision@5 | 34.9% |

---

## LLM Judge Results

### Pass Rate

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | 129 | 57.6% |
| Partial | 45 | 20.1% |
| Fail | 50 | 22.3% |
| Error | 0 | 0.0% |

### Quality Scores (1-5)

| Dimension | Score |
|-----------|-------|
| Overall Score | 3.89 |
| Correctness | 4.00 |
| Completeness | 3.53 |
| Faithfulness | 4.60 |
| Relevance | 4.08 |
| Clarity | 4.92 |

---

## Timing

| Phase | Avg per Query | Total |
|-------|---------------|-------|
| Retrieval | 0.183s | 41.1s |
| Reranking | 0.000s | 0.0s |
| Generation | 6.082s | 1362.5s |
| LLM Judge | 7.859s | 1760.4s |
| **Total** | 14.125s | 3164.0s |
