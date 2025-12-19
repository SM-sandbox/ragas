# Gold Standard Benchmark Report

## R002: Test vs Benchmark Comparison

**Report ID:** R002  
**Date:** 2025-12-18  
**Client:** BFAI  
**Verdict:** ❌ **FAIL**

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
| **Run ID** | R002 |
| **Type** | cloud |
| **Timestamp** | 2025-12-18T23:31:55.106336 |
| **File** | `results_p12_cloud.json` |

---

## Executive Summary

| Metric | Benchmark | Test | Δ | Status |
|--------|-----------|------|---|--------|
| **pass_rate** | 92.6% | 86.2% | -6.3% | ⚠️ |
| **partial_rate** | 6.6% | 10.3% | +3.7% | 🏆 |
| **fail_rate** | 0.9% | 3.5% | +2.6% | ⚠️ |
| **acceptable_rate** | 99.1% | 96.5% | -2.6% | ⚠️ |
| **recall_at_100** | 99.1% | 99.1% | +0.0% | ✅ |
| **mrr** | 74.1% | 74.1% | +0.0% | ✅ |
| **overall_score_avg** | 4.82 | 4.66 | -0.16 | ⚠️ |

### Key Finding

**Significant regressions in: pass_rate, fail_rate, acceptable_rate, overall_score_avg**

### Recommendation

Do not deploy. Investigate and fix regressions.

---

## Latency Analysis

| Metric | Benchmark | Test | Δ | Speedup |
|--------|-----------|------|---|---------|
| **Total (with Judge)** | 9.4s | 7.4s | -2.00s | 1.27x |
| **Client Experience** | 8.14s | 6.2s | -1.94s | 1.31x |

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
| **single_hop** | 94.1% | 90.5% | -3.6% |
| **multi_hop** | 91.1% | 82.2% | -8.9% |

---

## Breakdown by Difficulty

| Difficulty | Benchmark | Test | Δ |
|------------|-----------|------|---|
| **easy** | 89.4% | 83.9% | -5.6% |
| **medium** | 95.0% | 92.5% | -2.5% |
| **hard** | 93.4% | 81.6% | -11.8% |

---

## Failures Analysis

| Category | Count | Question IDs |
|----------|-------|--------------|
| **Benchmark Only** (fixed) | 0 |  |
| **Test Only** (new) | 12 | mh_easy_020, mh_easy_038, mh_easy_065, mh_hard_020, mh_hard_026... |
| **Both** (persistent) | 4 | sh_easy_021, sh_easy_066, sh_easy_079, sh_hard_020 |

---

## Configuration Match

| Parameter | Benchmark | Test | Match |
|-----------|-----------|------|-------|
| **model** | gemini-3-flash-preview | cloud | ❌ |
| **reasoning_effort** | low | None | ❌ |
| **temperature** | 0.0 | 0.0 | ✅ |
| **recall_top_k** | 100 | 100 | ✅ |
| **precision_top_n** | 25 | 12 | ❌ |

---

*Report generated: 2025-12-18T23:32:10.307592*  
*Benchmark: v1.3 (cloud)*  
*Report ID: R002*
