# Gold Standard Benchmark Report

## R001: Test vs Benchmark Comparison

**Report ID:** R001  
**Date:** 2025-12-18  
**Client:** BFAI  
**Verdict:** ✅ **PASS**

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
| **Run ID** | R001 |
| **Type** | cloud |
| **Timestamp** | 2025-12-18T21:02:03.043618 |
| **File** | `results_p25_cloud.json` |

---

## Executive Summary

| Metric | Benchmark | Test | Δ | Status |
|--------|-----------|------|---|--------|
| **pass_rate** | 92.6% | 92.6% | +0.0% | ✅ |
| **partial_rate** | 6.6% | 6.6% | +0.0% | ✅ |
| **fail_rate** | 0.9% | 0.9% | +0.0% | ✅ |
| **acceptable_rate** | 99.1% | 99.1% | +0.0% | ✅ |
| **recall_at_100** | 99.1% | 99.1% | +0.0% | ✅ |
| **mrr** | 74.1% | 74.1% | +0.0% | ✅ |
| **overall_score_avg** | 4.82 | 4.82 | +0.00 | ✅ |

### Key Finding

**Test matches benchmark on all metrics.**

### Recommendation

No action needed. Test meets benchmark standards.

---

## Latency Analysis

| Metric | Benchmark | Test | Δ | Speedup |
|--------|-----------|------|---|---------|
| **Total (with Judge)** | 9.4s | 9.4s | -0.00s | 1.0x |
| **Client Experience** | 8.14s | 8.14s | +0.00s | 1.0x |

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
| **single_hop** | 94.1% | 94.1% | +0.0% |
| **multi_hop** | 91.1% | 91.1% | +0.0% |

---

## Breakdown by Difficulty

| Difficulty | Benchmark | Test | Δ |
|------------|-----------|------|---|
| **easy** | 89.4% | 89.4% | +0.0% |
| **medium** | 95.0% | 95.0% | +0.0% |
| **hard** | 93.4% | 93.4% | +0.0% |

---

## Failures Analysis

| Category | Count | Question IDs |
|----------|-------|--------------|
| **Benchmark Only** (fixed) | 0 |  |
| **Test Only** (new) | 0 |  |
| **Both** (persistent) | 4 | sh_easy_021, sh_easy_066, sh_easy_079, sh_hard_020 |

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

*Report generated: 2025-12-18T23:09:57.437544*  
*Benchmark: v1.3 (cloud)*  
*Report ID: R001*
