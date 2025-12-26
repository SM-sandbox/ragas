# Checkpoint Report: C043

**Generated:** 2025-12-26 07:13:46
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 94.4% |
| **Partial Rate** | 2.8% |
| **Fail Rate** | 2.8% |
| **Acceptable Rate** | 97.2% |
| **Overall Score** | 4.92/5 |
| **Questions** | 36 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 94.4% | +1.2% | ✅ |
| **Acceptable Rate** | 97.6% | 97.2% | -0.4% | ✅ |
| **Fail Rate** | 2.4% | 2.8% | +0.4% | ✅ |
| **MRR** | 0.740 | 1.000 | +0.260 | ✅ |
| **Recall@100** | 99.1% | 100.0% | +0.9% | ✅ |
| **Overall Score** | 4.81/5 | 4.92/5 | +0.11 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 20.1s | -50.7s | 3.52x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.0626 | $-0.7910 |
| **Per Question** | $0.001864 | $0.001740 | $-0.000124 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 94.4% | -1.5% |
| **Multi-Hop** | 90.7% | 0.0% | -90.7% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 94.4% | +5.6% |
| **Medium** | 97.5% | 0.0% | -97.5% |
| **Hard** | 93.4% | 0.0% | -93.4% |

### Key Finding

✅ **All metrics within acceptable range of baseline.**

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-app-ppfq5ahfsq-ue.a.run.app |
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
|**Overall Score**  |  4.92/5|      4.92/5|     0.00/5|
|**Pass Rate**      |   94.4%|       94.4%|       0.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.92/5|  0.00/5|  0.00/5|
|**Pass Rate**      |   94.4%|    0.0%|    0.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.89/5|  4.89/5|  0.00/5|  0.00/5|
|**Completeness**  |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Faithfulness**  |  4.89/5|  4.89/5|  0.00/5|  0.00/5|
|**Relevance**     |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Clarity**       |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Overall**       |  4.92/5|  4.92/5|  0.00/5|  0.00/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.89/5|      4.89/5|     0.00/5|
|**Completeness**  |  5.00/5|      5.00/5|     0.00/5|
|**Faithfulness**  |  4.89/5|      4.89/5|     0.00/5|
|**Relevance**     |  5.00/5|      5.00/5|     0.00/5|
|**Clarity**       |  5.00/5|      5.00/5|     0.00/5|
|**Overall**       |  4.92/5|      4.92/5|     0.00/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     2.40s|       11.9%|
|**Total**       |    20.12s|        100%|

**Min Latency:** 8.12s  |  **Max Latency:** 34.70s

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
|**Single-Hop**  |     36|    34|        1|     1|      94.4%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |     36|    34|        1|     1|      94.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     36|      94.4%|  1.000|         4.92/5|

## Execution & Throttling

|Metric                 |  Value|
|-----------------------|------:|
|**Run Duration**       |  77.0s|
|**Questions/Second**   |   0.47|
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
|**Endpoint**      |  https://bfai-app-ppfq5ahfsq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-26 07:13:46*
*Checkpoint: C043*
*Judge Model: gemini-3-flash-preview*