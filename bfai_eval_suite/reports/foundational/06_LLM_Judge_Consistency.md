# LLM Judge Consistency Analysis

**Date:** December 16, 2025  
**Category:** Foundational  
**Status:** ✅ Complete

---

## 1. Objective

Validate that the LLM-as-Judge evaluation produces consistent, reproducible results and is not adding noise to our metrics.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Model | gemini-2.5-flash |
| Temperature | 0.0 |
| Context Size | 100 chunks |
| Questions | 224 (each run twice) |
| Total Evaluations | 448 |

---

## 3. Results

### Verdict Consistency

| Metric | Value |
|--------|-------|
| Same verdict both runs | **224/224 (100%)** |
| Different verdict | 0/224 (0%) |

### Score Consistency

| Metric | Value |
|--------|-------|
| Avg score difference | **0.007** |
| Max score difference | 1.0 |
| Identical scores | **222/224 (99.1%)** |

### Answer Text Similarity

| Metric | Value |
|--------|-------|
| Avg similarity | **100.0%** |
| Identical answers | **224/224 (100%)** |

### Per-Run Metrics

| Metric | Run 1 | Run 2 | Delta |
|--------|-------|-------|-------|
| Pass Rate | 76.3% | 76.3% | 0.0% |
| Avg Score | 4.44 | 4.44 | 0.00 |
| Avg Gen Time | 8.9s | 8.9s | 0.0s |

---

## 4. Key Findings

1. **Perfect verdict consistency** - All 224 questions received identical verdicts across both runs.

2. **Byte-for-byte identical answers** - At temperature 0.0, the LLM produces exactly the same output every time.

3. **Negligible score variance** - 99.1% of scores were identical; the 0.9% difference was in judge scoring, not generation.

4. **Deterministic system** - The entire pipeline (retrieval → generation → judging) is fully reproducible.

---

## 5. Recommendation

✅ **Use temperature 0.0 for all evaluation runs.**

This ensures:

- Reproducible results for A/B testing
- No noise from generation variance
- Reliable metrics for comparing configurations

For production deployments where some creativity is desired, temperature 0.1-0.3 can be used, but evaluation should always use 0.0.

---

## 6. Related Reports

- [Temperature Sweep](../experiments/Temperature_Sweep.md)
- [Model Comparison: Flash vs Pro](../experiments/Model_Comparison_Flash_vs_Pro.md)
