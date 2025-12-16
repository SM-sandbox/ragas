# Context 15 Report

**Date:** 2025-12-16 05:25

---

## Configuration

| Setting | Value |
|---------|-------|
| Temperature | 0.0 |
| Context Size | 15 chunks |
| Questions | 224 |
| Embedding | gemini-1536-RETRIEVAL_QUERY |
| Generation Model | gemini-2.5-flash |
| Judge Model | gemini-2.5-flash (temp 0.0) |

---

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| Recall@100 | 70.1% |
| Precision@15 | 26.4% |

---

## LLM Judge Results

### Pass Rate

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | 148 | 66.1% |
| Partial | 40 | 17.9% |
| Fail | 36 | 16.1% |
| Error | 0 | 0.0% |

### Quality Scores (1-5)

| Dimension | Score |
|-----------|-------|
| Overall Score | 4.15 |
| Correctness | 4.17 |
| Completeness | 3.94 |
| Faithfulness | 4.63 |
| Relevance | 4.34 |
| Clarity | 4.94 |

---

## Timing

| Phase | Avg per Query | Total |
|-------|---------------|-------|
| Retrieval | 0.183s | 41.1s |
| Reranking | 0.000s | 0.0s |
| Generation | 7.247s | 1623.4s |
| LLM Judge | 7.788s | 1744.6s |
| **Total** | 15.219s | 3409.1s |
