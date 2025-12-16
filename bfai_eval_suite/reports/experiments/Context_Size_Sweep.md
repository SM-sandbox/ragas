# Context Size Sweep Experiment

**Date:** December 16, 2025  
**Category:** Experiment  
**Status:** ✅ Complete

---

## 1. Objective

Determine the optimal number of retrieved chunks to include in the LLM context for answer generation.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Model | gemini-2.5-flash |
| Temperature | 0.0 |
| Context Sizes Tested | 5, 10, 15, 20, 25, 50, 100 |
| Questions | 224 |
| Embedding | gemini-1536-RETRIEVAL_QUERY |
| Retrieval | Top 100 (hybrid 50/50) |

---

## 3. Results

### Quality Metrics by Context Size

| Context | Pass Rate | Overall Score | Correctness | Completeness | Faithfulness |
|---------|-----------|---------------|-------------|--------------|--------------|
| 5 | 57.6% | 3.89 | 4.00 | 3.53 | 4.60 |
| 10 | 62.9% | 4.04 | 4.06 | 3.77 | 4.58 |
| 15 | 66.1% | 4.15 | 4.17 | 3.94 | 4.63 |
| 20 | 71.4% | 4.29 | 4.23 | 4.12 | 4.62 |
| **25** | **73.7%** | **4.38** | 4.37 | 4.23 | 4.72 |
| 50 | 73.7% | 4.40 | 4.45 | 4.23 | 4.76 |
| **100** | **76.3%** | **4.44** | 4.45 | 4.29 | 4.72 |

### Generation Time by Context Size

| Context | Avg Gen Time | Delta from 5 |
|---------|--------------|--------------|
| 5 | 6.08s | Baseline |
| 10 | 6.88s | +0.80s |
| 15 | 7.25s | +1.17s |
| 20 | 7.13s | +1.05s |
| 25 | 7.39s | +1.31s |
| 50 | 7.49s | +1.41s |
| 100 | 8.97s | +2.89s |

### Improvement Curve

| Transition | Pass Rate Δ | Time Δ | Efficiency |
|------------|-------------|--------|------------|
| 5 → 10 | +5.3% | +0.80s | 6.6%/s |
| 10 → 15 | +3.2% | +0.37s | 8.6%/s |
| 15 → 20 | +5.3% | -0.12s | ∞ (faster!) |
| 20 → 25 | +2.3% | +0.26s | 8.8%/s |
| 25 → 50 | +0.0% | +0.10s | 0%/s |
| 50 → 100 | +2.6% | +1.48s | 1.8%/s |

---

## 4. Key Findings

1. **Strong improvement up to 25 chunks** - Pass rate improves from 57.6% to 73.7% (+16.1%).

2. **Plateau at 25-50** - No improvement from 25 to 50 chunks (both 73.7%).

3. **Small gain at 100** - Going to 100 chunks adds +2.6% pass rate but +1.5s latency.

4. **Completeness benefits most** - Score improves from 3.53 to 4.29 (+0.76) with more context.

5. **Diminishing returns** - The efficiency (% improvement per second) drops significantly after 25 chunks.

---

## 5. Recommendation

### Production (Latency-Sensitive)

✅ **Use 25 chunks**

- 73.7% pass rate
- 7.4s generation time
- Best efficiency tradeoff

### Quality-Critical Applications

✅ **Use 100 chunks**

- 76.3% pass rate (+2.6%)
- 9.0s generation time (+1.6s)
- Maximum quality

---

## 6. Related Reports

- [Temperature Sweep](Temperature_Sweep.md)
- [Model Comparison: Flash vs Pro](Model_Comparison_Flash_vs_Pro.md)
- [Reranker Impact](../foundational/05_Reranker_Impact.md)
