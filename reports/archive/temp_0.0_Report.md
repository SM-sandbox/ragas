# Temp 0.0 Report

**Date:** 2025-12-16 00:02

---

## Configuration

| Setting | Value |
|---------|-------|
| Temperature | 0.0 |
| Context Size | 10 chunks |
| Questions | 224 |
| Embedding | gemini-1536-RETRIEVAL_QUERY |
| Generation Model | gemini-2.5-flash |
| Judge Model | gemini-2.5-flash (temp 0.0) |

---

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| Recall@100 | 70.1% |
| Precision@10 | 29.2% |

---

## LLM Judge Results

### Pass Rate

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | 141 | 62.9% |
| Partial | 40 | 17.9% |
| Fail | 43 | 19.2% |
| Error | 0 | 0.0% |

### Quality Scores (1-5)

| Dimension | Score |
|-----------|-------|
| Overall Score | 4.04 |
| Correctness | 4.06 |
| Completeness | 3.77 |
| Faithfulness | 4.58 |
| Relevance | 4.20 |
| Clarity | 4.90 |

---

## Timing

| Phase | Avg per Query | Total |
|-------|---------------|-------|
| Retrieval | 0.183s | 41.1s |
| Reranking | 0.000s | 0.0s |
| Generation | 6.345s | 1421.3s |
| LLM Judge | 7.267s | 1627.9s |
| **Total** | 13.796s | 3090.2s |
