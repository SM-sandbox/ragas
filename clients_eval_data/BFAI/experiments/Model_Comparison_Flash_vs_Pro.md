# Model Comparison: Gemini 2.5 Flash vs Pro

**Date:** December 16, 2025  
**Category:** Experiment  
**Status:** ✅ Complete

---

## 1. Objective

Compare Gemini 2.5 Flash and Gemini 2.5 Pro (Low Reasoning) for answer generation quality and latency.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Models Tested | gemini-2.5-flash, gemini-2.5-pro |
| Temperature | 0.0 |
| Context Size | 100 chunks |
| Questions | 224 |
| Pro Thinking Budget | 1024 tokens |
| Embedding | gemini-1536-RETRIEVAL_QUERY |

---

## 3. Results

### Overall Comparison

| Metric | Flash | Pro | Delta |
|--------|-------|-----|-------|
| **Pass Rate** | 76.3% | 77.2% | +0.9% |
| **Overall Score** | 4.44/5 | 4.45/5 | +0.02 |
| **Avg Gen Time** | 8.9s | 16.4s | +7.5s (+84%) |

### Verdict Distribution

| Verdict | Flash | Pro | Delta |
|---------|-------|-----|-------|
| Pass | 171 (76.3%) | 173 (77.2%) | +2 |
| Partial | 33 (14.7%) | 31 (13.8%) | -2 |
| Fail | 20 (8.9%) | 20 (8.9%) | 0 |

### Detailed Scores

| Dimension | Flash | Pro | Delta |
|-----------|-------|-----|-------|
| Correctness | 4.46 | 4.45 | -0.01 |
| Completeness | 4.29 | 4.29 | 0.00 |
| Faithfulness | 4.73 | 4.78 | +0.05 |
| Relevance | 4.61 | 4.54 | -0.07 |
| Clarity | 4.93 | 4.94 | +0.01 |

### Total Experiment Time

| Model | Total Time |
|-------|------------|
| Flash | ~63 min |
| Pro | ~90 min |

---

## 4. Key Findings

1. **Marginal quality difference** - Pro is only +0.9% better on pass rate (77.2% vs 76.3%).

2. **Significant latency penalty** - Pro is 84% slower (16.4s vs 8.9s per question).

3. **No dimension wins clearly** - Flash is slightly better on Correctness and Relevance; Pro is slightly better on Faithfulness.

4. **Same fail rate** - Both models fail on the same 20 questions (8.9%), suggesting these are genuinely hard questions.

5. **Cost implications** - Pro costs more per token AND takes longer, doubling the effective cost.

---

## 5. Recommendation

✅ **Use Gemini 2.5 Flash for production.**

The quality difference is marginal (+0.9%), but Pro is 84% slower. Flash provides:

- Nearly identical quality
- 47% faster response time
- Lower cost per query
- Better user experience

Pro may be worth considering for:

- Offline batch processing where latency doesn't matter
- Extremely high-stakes queries where every 1% matters
- Complex multi-hop reasoning (not tested here)

---

## 6. Related Reports

- [Temperature Sweep](Temperature_Sweep.md)
- [Context Size Sweep](Context_Size_Sweep.md)
- [LLM Judge Consistency](../foundational/06_LLM_Judge_Consistency.md)
