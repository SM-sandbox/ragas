# Gold Standard Benchmark Report

## R003: Test vs Benchmark Comparison

**Report ID:** R003  
**Date:** 2025-12-19  
**Client:** BFAI  
**Verdict:** ⚠️ **WARN**

---

## Benchmark Reference

| Property | Value |
|----------|-------|
| **Version** | v1.3 |
| **Type** | cloud |
| **Date** | 2025-12-18 |
| **File** | `benchmark_BFAI_v1.3__cloud__2025-12-18.json` |

---

## Test Run

| Property | Value |
|----------|-------|
| **Run ID** | R003 |
| **Type** | cloud |
| **Timestamp** | 2025-12-19T00:34:14.534290 |
| **File** | `results_p25_cloud_gemini_3_pro_preview.json` |

---

## Executive Summary

| Metric | Benchmark | Test | Δ | Status |
|--------|-----------|------|---|--------|
| **pass_rate** | 92.6% | 91.0% | -1.6% | ✅ |
| **partial_rate** | 6.6% | 7.7% | +1.1% | ✅ |
| **fail_rate** | 0.9% | 1.3% | +0.4% | ✅ |
| **acceptable_rate** | 99.1% | 98.7% | -0.4% | ✅ |
| **recall_at_100** | 99.1% | 99.1% | -0.0% | ✅ |
| **mrr** | 74.1% | 73.8% | -0.3% | ✅ |
| **overall_score_avg** | 4.82 | 4.80 | -0.02 | ✅ |

### Key Finding

**Minor regressions detected in: latency**

### Recommendation

Review regressions before deploying. May be acceptable.

---

## Latency Analysis

| Metric | Benchmark | Test | Δ | Speedup |
|--------|-----------|------|---|---------|
| **Total (with Judge)** | 9.4s | 35.01s | +25.61s | 0.27x |
| **Client Experience** | 8.14s | 33.65s | +25.51s | 0.24x |

> **Note:** Client Experience excludes judge latency (eval-only). This is what end users experience.

---

## Cost Analysis

| Metric | Benchmark | Test | Δ |
|--------|-----------|------|---|
| **Total Cost** | $0.4968 | $0.0000 | $-0.4968 |
| **Per Question** | $0.001085 | $0.000000 | $-0.001085 |

---

## Breakdown by Question Type

| Type | Benchmark | Test | Δ |
|------|-----------|------|---|
| **single_hop** | 94.1% | 93.7% | -0.5% |
| **multi_hop** | 91.1% | 88.6% | -2.5% |

---

## Breakdown by Difficulty

| Difficulty | Benchmark | Test | Δ |
|------------|-----------|------|---|
| **easy** | 89.4% | 87.6% | -1.9% |
| **medium** | 95.0% | 93.8% | -1.3% |
| **hard** | 93.4% | 91.9% | -1.5% |

---

## Failures Analysis

| Category | Count | Question IDs |
|----------|-------|--------------|
| **Benchmark Only** (fixed) | 1 | sh_easy_066 |
| **Test Only** (new) | 3 | mh_easy_038, sh_easy_083, sh_med_034 |
| **Both** (persistent) | 3 | sh_easy_021, sh_easy_079, sh_hard_020 |

---

## Configuration Match

| Parameter | Benchmark | Test | Match |
|-----------|-----------|------|-------|
| **model** | gemini-3-flash-preview | cloud | ❌ |
| **reasoning_effort** | low | None | ❌ |
| **temperature** | 0.0 | 0.0 | ✅ |
| **recall_top_k** | 100 | 100 | ✅ |
| **precision_top_n** | 25 | 25 | ✅ |

---

*Report generated: 2025-12-19T00:34:28.371713*  
*Benchmark: v1.3 (cloud)*  
*Report ID: R003*
