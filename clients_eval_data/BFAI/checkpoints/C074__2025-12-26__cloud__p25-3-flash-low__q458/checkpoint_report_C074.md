# Checkpoint Report: C074

**Generated:** 2025-12-26 21:47:23
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 0.0% |
| **Partial Rate** | 0.0% |
| **Fail Rate** | 0.0% |
| **Acceptable Rate** | 0.0% |
| **Overall Score** | 0.00/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 0.0% | -92.1% | ❌ |
| **Acceptable Rate** | 98.9% | 0.0% | -98.9% | ❌ |
| **Fail Rate** | 1.1% | 0.0% | -1.1% | ✅ |
| **MRR** | 0.740 | 0.000 | -0.740 | ⚠️ |
| **Recall@100** | 99.1% | 0.0% | -99.1% | ⚠️ |
| **Overall Score** | 4.82/5 | 0.00/5 | -4.82 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 0.0s | -57.7s | 0.00x ❌ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.0000 | $-0.8664 |
| **Per Question** | $0.001892 | $0.000000 | $-0.001892 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 0.0% | -92.8% |
| **Multi-Hop** | 91.5% | 0.0% | -91.5% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 0.0% | -89.4% |
| **Medium** | 93.2% | 0.0% | -93.2% |
| **Hard** | 94.1% | 0.0% | -94.1% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 92.1%, acceptable rate dropped 98.9%, latency increased 57.7s

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-app-eb2qyzdzvq-ue.a.run.app |
| **Precision@K** | 25 |
| **Recall@K** | 100 |
| **Max Workers** | 100 |
| **Effective Workers** | 50 |

### Generator Model

| Parameter | Value |
|-----------|-------|
| **Model** | unknown |
| **Reasoning Effort** | low |
| **Temperature** | 0.0 |
| **Seed** | 42 |

### Judge Model

| Parameter | Value |
|-----------|-------|
| **Model** | gemini-3-flash-preview |
| **Reasoning Effort** | low |
| **Seed** | 42 |

## Retrieval Metrics

### By Question Type

|Metric          |  Total|  Single-Hop|  Multi-Hop|
|----------------|------:|-----------:|----------:|
|**Recall@100**  |   0.0%|        0.0%|       0.0%|
|**MRR**         |  0.000|       0.000|      0.000|

### By Difficulty

|Metric          |   Easy|  Medium|   Hard|
|----------------|------:|-------:|------:|
|**Recall@100**  |   0.0%|    0.0%|   0.0%|
|**MRR**         |  0.000|   0.000|  0.000|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  0.00/5|      0.00/5|     0.00/5|
|**Pass Rate**      |    0.0%|        0.0%|       0.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  0.00/5|  0.00/5|  0.00/5|
|**Pass Rate**      |    0.0%|    0.0%|    0.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  0.00/5|  0.00/5|  0.00/5|  0.00/5|
|**Completeness**  |  0.00/5|  0.00/5|  0.00/5|  0.00/5|
|**Faithfulness**  |  0.00/5|  0.00/5|  0.00/5|  0.00/5|
|**Relevance**     |  0.00/5|  0.00/5|  0.00/5|  0.00/5|
|**Clarity**       |  0.00/5|  0.00/5|  0.00/5|  0.00/5|
|**Overall**       |  0.00/5|  0.00/5|  0.00/5|  0.00/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  0.00/5|      0.00/5|     0.00/5|
|**Completeness**  |  0.00/5|      0.00/5|     0.00/5|
|**Faithfulness**  |  0.00/5|      0.00/5|     0.00/5|
|**Relevance**     |  0.00/5|      0.00/5|     0.00/5|
|**Clarity**       |  0.00/5|      0.00/5|     0.00/5|
|**Overall**       |  0.00/5|      0.00/5|     0.00/5|

## Latency Analysis


**Min Latency:** 0.00s  |  **Max Latency:** 0.00s

## Token & Cost Analysis

### Token Breakdown

|Token Type               |  Total|  Per Question|
|-------------------------|------:|-------------:|
|**Prompt (Input)**       |      0|             0|
|**Completion (Output)**  |      0|             0|
|**Thinking**             |      0|             0|
|**Cached**               |      0|             0|
|**Total**                |      0|             0|

### Cost Estimate (Gemini 3 Flash)

|Component                |         Cost|
|-------------------------|------------:|
|**Input**                |      $0.0000|
|**Output**               |      $0.0000|
|**Thinking**             |      $0.0000|
|**Cached Savings**       |     -$0.0000|
|**Total**                |  **$0.0000**|
|**Per Question**         |    $0.000000|
|**Per 1,000 Questions**  |        $0.00|

## Breakdown by Question Type


## Breakdown by Difficulty


## Breakdown by Type × Difficulty


## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  438.4s|
|**Questions/Second**   |    0.00|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |       0|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.00%|
|**TPM Utilization**    |   0.00%|

## Index & Orchestrator

|Field             |                                     Value|
|------------------|-----------------------------------------:|
|**Index/Job ID**  |                  bfai__eval66a_g1_1536_tt|
|**Mode**          |                                     cloud|
|**Endpoint**      |  https://bfai-app-eb2qyzdzvq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-26 21:47:23*
*Checkpoint: C074*
*Judge Model: gemini-3-flash-preview*