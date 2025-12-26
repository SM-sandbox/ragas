# BFAI Checkpoint Version History

**Last Updated:** 2025-12-26

This document tracks all checkpoint evaluations from C001 to present, including gold baseline transitions and code changes that impacted results.

---

## Gold Baseline History

| Baseline | Checkpoint | Date Locked | Pass Rate | Acceptable | Endpoint | Notes |
|----------|------------|-------------|-----------|------------|----------|-------|
| **#1** | **C012** | Dec 20, 2025 | 92.8% | 98.5% | `bfai-api` | First gold baseline, gemini-3-flash-preview |
| **#2** | **C034** | Dec 23, 2025 | 93.2% | 97.6% | `bfai-api` | Upgraded baseline, same config, last run on bfai-api |

---

## Complete Checkpoint Timeline

### Legend

- 🏆 = Gold Baseline
- ✅ = Passed (within baseline tolerance)
- ⚠️ = Warning (anomaly or degraded)
- ❌ = Regression

### December 2025

| ID | Date | Mode | Pass Rate | Fail Rate | Acceptable | MRR | Questions | Gold Baseline | Endpoint | Notes |
|----|------|------|-----------|-----------|------------|-----|-----------|---------------|----------|-------|
| **C001** | Dec 19 | Cloud | **92.6%** | 0.9% | 99.1% | 0.741 | 458 | (none yet) | `bfai-api` | Initial cloud baseline, p25 |
| **C002** | Dec 18 | Cloud | **86.2%** | 3.5% | 96.5% | 0.741 | 458 | (none yet) | `bfai-api` | p12 comparison (fewer chunks = worse) |
| **C003** | Dec 19 | Local | **92.4%** | 1.1% | 98.9% | 0.737 | 458 | (none yet) | local | Local baseline p25 |
| **C004** | Dec 19 | Cloud | **92.1%** | 1.1% | 98.9% | 0.738 | 458 | (none yet) | `bfai-api` | Gemini 3 Pro test |
| **C005** | Dec 19 | Local | **91.9%** | 1.5% | 98.5% | 0.740 | 458 | (none yet) | local | gemini-3-flash-preview |
| **C006** | Dec 19 | Local | **91.3%** | 1.5% | 98.5% | 0.737 | 458 | (none yet) | local | Same config, variance test |
| **C007** | Dec 19 | Cloud | **92.4%** | 1.7% | 98.3% | 0.740 | 458 | (none yet) | `bfai-api` | Cloud with 3-flash-low |
| **C008** | Dec 19 | Cloud | **92.8%** | 1.1% | 98.9% | 0.738 | 458 | (none yet) | `bfai-api` | Candidate for baseline |
| **C009** | Dec 19 | Cloud | **93.2%** | 0.9% | 99.1% | 0.739 | 458 | (none yet) | `bfai-api` | Best so far |
| **C010** | Dec 19 | Cloud | **93.7%** | 1.1% | 98.9% | 0.740 | 458 | (none yet) | `bfai-api` | Highest pass rate! |
| **C011** | Dec 19 | Cloud | **62.7%** | 0.7% | 99.3% | 0.739 | 458 | (none yet) | `bfai-api` | ⚠️ Anomaly - high partial rate |
| **C012** | Dec 19 | Cloud | **92.8%** | 1.5% | 98.5% | 0.737 | 458 | **→ C012** | `bfai-api` | 🏆 **GOLD BASELINE #1** (locked Dec 20) |
| **C013** | Dec 20 | Local | **91.9%** | 1.5% | 98.5% | 0.738 | 458 | C012 | local | Smart Throttler test |
| **C014** | Dec 20 | Cloud | **93.7%** | 1.3% | 98.7% | 0.739 | 458 | C012 | `bfai-api` | v0.2.0-schema-foundation |
| **C015** | Dec 20 | Cloud | **92.8%** | 1.1% | 98.9% | 0.738 | 458 | C012 | `bfai-api` | Stable |
| **C016** | Dec 21 | Local | **91.9%** | 1.3% | 98.7% | - | 458 | C012 | local | Local validation |
| **C019** | Dec 21 | Cloud | **0.0%** | 0.0% | 100% | - | 10 | C012 | `bfai-api` | Quick test (10 questions) |
| **C020** | Dec 21 | Cloud | **92.1%** | 1.1% | 98.9% | 0.739 | 458 | C012 | `bfai-api` | Stable |
| **C021** | Dec 22 | Local | **89.3%** | 3.7% | 96.3% | 0.861 | 270 | C012 | local | Partial run |
| **C022** | Dec 22 | Local | **91.7%** | 4.8% | 95.2% | 0.925 | 230 | C012 | local | Partial run |
| **C024** | Dec 22 | Local | **50.7%** | 8.1% | 91.3% | - | 458 | C012 | local | ⚠️ gemini-2.5-flash test (worse) |
| **C025** | Dec 22 | Cloud | **40.4%** | 10.7% | 87.3% | 0.737 | 458 | C012 | `bfai-api` | ⚠️ gemini-2.5-flash cloud (worse) |
| **C026** | Dec 22 | Cloud | **0%** | 0% | 0% | - | 458 | C012 | `bfai-api` | Failed run |
| **C027** | Dec 22 | Local | **88.6%** | 3.5% | 96.5% | - | 458 | C012 | local | gemini-2.5-flash local |
| **C028** | Dec 22 | Local | **96.7%** | 0.8% | 99.2% | 1.000 | 120 | C012 | local | Partial run, high MRR |
| **C029** | Dec 23 | Local | **92.8%** | 1.5% | 98.5% | - | 458 | C012 | local | v0.3.0 tag |
| **C030** | Dec 23 | Local | **91.7%** | 2.2% | 97.8% | - | 458 | C012 | local | eval66b index test |
| **C031** | Dec 23 | Local | **91.9%** | 2.2% | 97.8% | - | 458 | C012 | local | eval66b index |
| **C032** | Dec 23 | Cloud | **87.8%** | 3.3% | 96.7% | - | 458 | C012 | `bfai-api` | gemini-2.5-flash cloud |
| **C033** | Dec 23 | Cloud | **FAILED** | - | - | - | 458 | C012 | `bfai-api` | DNS failure on eval66b endpoint |
| **C034** | Dec 23 | Cloud | **93.2%** | 2.4% | 97.6% | 0.740 | 458 | **→ C034** | `bfai-api` | 🏆 **GOLD BASELINE #2** (upgraded) |
| **C035** | Dec 23 | Cloud | **93.2%** | 2.2% | 97.8% | - | 458 | C034 | `bfai-api` | Last run on bfai-api |
| **C036** | Dec 24 | Cloud | **58.3%** | 38.9% | 61.1% | - | 36 | C034 | `bfai-app` | ❌ **REGRESSION** - New bfai-app + query rewriting |
| **C037** | Dec 25 | Cloud | **53.1%** | 31.4% | 68.6% | - | 458 | C034 | `bfai-app` | ❌ Still broken (ranking API perms) |
| **C038** | Dec 25 | Cloud | **53.9%** | 29.9% | 70.1% | - | 458 | C034 | `bfai-app` | ❌ Still broken |
| **C041** | Dec 25 | Cloud | **56.8%** | 24.5% | 75.5% | - | 458 | C034 | `bfai-app` | ❌ Ranking fixed, rewriting still on |
| **C043** | Dec 26 | Cloud | **94.4%** | 2.8% | 97.2% | 1.000 | 36 | C034 | `bfai-app` | ✅ **FIX APPLIED** (query rewriting disabled) |
| **C044** | Dec 26 | Cloud | **91.3%** | 1.3% | **98.7%** | 0.737 | 458 | C034 | `bfai-app` | ✅ **FIX VERIFIED** (full corpus) |

---

## Code Changes by Date

| Date | Version Tag | Key Changes | Impact on Eval |
|------|-------------|-------------|----------------|
| Dec 17 | v0.2.0-schema-foundation | google-genai SDK, Gemini 3 default model | Stable |
| Dec 18 | - | Multi-turn conversation support | Stable |
| Dec 19 | - | C012 locked as gold baseline | 92.8% baseline established |
| Dec 22 | v0.3.0 | Multi-region Vertex AI cascade, deploy CLI | Stable |
| Dec 23 | - | C034 upgraded to gold baseline | 93.2% baseline established |
| Dec 24 | - | **Query rewriting module merged** (`ccf9349d`) | ❌ Caused regression |
| Dec 24 | - | **bfai-app deployed** (replaced bfai-api) | ❌ New service with bugs |
| Dec 25 | v0.4.0 | APP breakout, test coverage improvements | Still broken |
| Dec 25 | - | **Ranking API permissions fixed** | Partial improvement |
| Dec 26 | - | **Query rewriting disabled** | ✅ Fixed |
| Dec 26 | - | **PROCESSED_FOLDER path fixed** | ✅ Fixed |

---

## Regression Analysis: C036-C041

### Root Cause
The `bfai-app` service had **query rewriting enabled by default**, but the baseline `bfai-api` did NOT have this feature. Query rewriting was mangling queries before retrieval, causing worse answers.

### Symptoms
- Pass rate dropped from 93.2% → 53-58%
- Retrieval metrics (MRR, Recall@100) were identical to baseline
- Correctness score dropped from 4.77 → 3.61
- Answer length dropped from 1823 → 1596 chars

### Fix Applied (Dec 26, 2025)
Disabled query rewriting by default in three locations:

1. `services/api/core/config.py` lines 43-44:
   ```python
   enable_query_rewriting: bool = False
   enable_first_turn_rewriting: bool = False
   ```

2. `services/api/core/config.py` lines 92-93 (from_job_config defaults):
   - Changed defaults from `True` to `False`

3. `services/api/orchestration/factory.py` line 128:
   ```python
   enable_rewriting = job_config.get("enable_query_rewriting", False)
   ```

### Also Fixed: PROCESSED_FOLDER Path
`libs/core/gcp_config.py` line 260: Changed from `"~processed"` to `"processed"`

---

## Service Endpoint History

| Service | URL | Status | Notes |
|---------|-----|--------|-------|
| `bfai-api` | `https://bfai-api-ppfq5ahfsq-ue.a.run.app` | **DELETED** | Original baseline service (C001-C035) |
| `bfai-app` | `https://bfai-app-ppfq5ahfsq-ue.a.run.app` | **ACTIVE** | Current production service (C036+) |

---

## Key Findings

1. **C012** was the first gold baseline (92.8%, locked Dec 20)
2. **C034** became the second gold baseline (93.2%, locked Dec 23) - last run on `bfai-api`
3. **C036-C041** showed the regression (53-58%) after switching to `bfai-app` with query rewriting
4. **C044** (91.3%) restored performance after fix - acceptable rate (98.7%) is actually **better** than baseline (97.6%)

---

*Generated: 2025-12-26*
