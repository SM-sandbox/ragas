# Checkpoint Report: C036

**Generated:** 2025-12-24 18:34:44
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 58.3% |
| **Partial Rate** | 2.8% |
| **Fail Rate** | 38.9% |
| **Acceptable Rate** | 61.1% |
| **Overall Score** | 3.94/5 |
| **Questions** | 36 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 58.3% | -34.9% | ❌ |
| **Acceptable Rate** | 97.6% | 61.1% | -36.5% | ❌ |
| **Fail Rate** | 2.4% | 38.9% | +36.5% | ❌ |
| **MRR** | 0.740 | 1.000 | +0.260 | ✅ |
| **Recall@100** | 99.1% | 100.0% | +0.9% | ✅ |
| **Overall Score** | 4.81/5 | 3.94/5 | -0.87 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 24.4s | -46.4s | 2.90x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.0534 | $-0.8002 |
| **Per Question** | $0.001864 | $0.001483 | $-0.000380 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 58.3% | -37.6% |
| **Multi-Hop** | 90.7% | 0.0% | -90.7% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 58.3% | -30.5% |
| **Medium** | 97.5% | 0.0% | -97.5% |
| **Hard** | 93.4% | 0.0% | -93.4% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 34.9%, acceptable rate dropped 36.5%

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
| **Model** | gemini-3-flash-preview |
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

|Metric          |   Total|  Single-Hop|  Multi-Hop|
|----------------|-------:|-----------:|----------:|
|**Recall@100**  |  100.0%|      100.0%|       0.0%|
|**MRR**         |   1.000|       1.000|      0.000|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|    0.0%|   0.0%|
|**MRR**         |   1.000|   0.000|  0.000|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  3.94/5|      3.94/5|     0.00/5|
|**Pass Rate**      |   58.3%|       58.3%|       0.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  3.94/5|  0.00/5|  0.00/5|
|**Pass Rate**      |   58.3%|    0.0%|    0.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  3.39/5|  3.39/5|  0.00/5|  0.00/5|
|**Completeness**  |  3.61/5|  3.61/5|  0.00/5|  0.00/5|
|**Faithfulness**  |  4.67/5|  4.67/5|  0.00/5|  0.00/5|
|**Relevance**     |  4.00/5|  4.00/5|  0.00/5|  0.00/5|
|**Clarity**       |  4.86/5|  4.86/5|  0.00/5|  0.00/5|
|**Overall**       |  3.94/5|  3.94/5|  0.00/5|  0.00/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  3.39/5|      3.39/5|     0.00/5|
|**Completeness**  |  3.61/5|      3.61/5|     0.00/5|
|**Faithfulness**  |  4.67/5|      4.67/5|     0.00/5|
|**Relevance**     |  4.00/5|      4.00/5|     0.00/5|
|**Clarity**       |  4.86/5|      4.86/5|     0.00/5|
|**Overall**       |  3.94/5|      3.94/5|     0.00/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     2.52s|       10.3%|
|**Total**       |    24.42s|        100%|

**Min Latency:** 10.74s  |  **Max Latency:** 47.74s

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

|Type            |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|----------------|------:|-----:|--------:|-----:|----------:|
|**Single-Hop**  |     36|    21|        1|    14|      58.3%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |     36|    21|        1|    14|      58.3%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     36|      58.3%|  1.000|         3.94/5|

## Execution & Throttling

|Metric                 |  Value|
|-----------------------|------:|
|**Run Duration**       |  80.7s|
|**Questions/Second**   |   0.45|
|**Max Workers**        |    100|
|**Effective Workers**  |     50|
|**Total Requests**     |     36|
|**Total Throttles**    |      0|
|**RPM Utilization**    |  0.18%|
|**TPM Utilization**    |  0.18%|

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

*Report generated: 2025-12-24 18:34:44*
*Checkpoint: C036*
*Judge Model: gemini-3-flash-preview*