# Checkpoint Report: C047

**Generated:** 2025-12-26 08:12:07
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.4% |
| **Partial Rate** | 6.1% |
| **Fail Rate** | 1.5% |
| **Acceptable Rate** | 98.5% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 92.4% | -0.9% | ✅ |
| **Acceptable Rate** | 97.6% | 98.5% | +0.9% | ✅ |
| **Fail Rate** | 2.4% | 1.5% | -0.9% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 4.82/5 | +0.02 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 50.0s | -20.8s | 1.42x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8650 | $+0.0114 |
| **Per Question** | $0.001864 | $0.001889 | $+0.000025 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 93.7% | -2.3% |
| **Multi-Hop** | 90.7% | 91.1% | +0.4% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 89.4% | +0.6% |
| **Medium** | 97.5% | 93.2% | -4.3% |
| **Hard** | 93.4% | 94.9% | +1.5% |

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

|Metric          |  Total|  Single-Hop|  Multi-Hop|
|----------------|------:|-----------:|----------:|
|**Recall@100**  |  99.1%|      100.0%|      98.3%|
|**MRR**         |  0.739|       1.000|      0.494|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.783|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.86/5|     4.78/5|
|**Pass Rate**      |   92.4%|       93.7%|      91.1%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.77/5|  4.85/5|  4.85/5|
|**Pass Rate**      |   89.4%|   93.2%|   94.9%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.76/5|  4.71/5|  4.79/5|  4.79/5|
|**Completeness**  |  4.86/5|  4.82/5|  4.88/5|  4.88/5|
|**Faithfulness**  |  4.88/5|  4.84/5|  4.89/5|  4.93/5|
|**Relevance**     |  4.97/5|  4.95/5|  4.97/5|  4.99/5|
|**Clarity**       |  4.99/5|  5.00/5|  4.99/5|  4.99/5|
|**Overall**       |  4.82/5|  4.77/5|  4.85/5|  4.85/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.76/5|      4.82/5|     4.70/5|
|**Completeness**  |  4.86/5|      4.91/5|     4.82/5|
|**Faithfulness**  |  4.88/5|      4.85/5|     4.92/5|
|**Relevance**     |  4.97/5|      4.94/5|     5.00/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.99/5|
|**Overall**       |  4.82/5|      4.86/5|     4.78/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.75s|        3.5%|
|**Total**       |    50.00s|        100%|

**Min Latency:** 6.37s  |  **Max Latency:** 110.90s

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
|**Single-Hop**  |    222|   208|        9|     5|      93.7%|
|**Multi-Hop**   |    236|   215|       19|     2|      91.1%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   144|       11|     6|      89.4%|
|**Medium**  |    161|   150|       10|     1|      93.2%|
|**Hard**    |    136|   129|        7|     0|      94.9%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      92.0%|  1.000|         4.83/5|
|**Single-Hop**  |      Medium|     78|      92.3%|  1.000|         4.86/5|
|**Single-Hop**  |        Hard|     56|      98.2%|  1.000|         4.92/5|
|**Multi-Hop**   |        Easy|     73|      86.3%|  0.459|         4.70/5|
|**Multi-Hop**   |      Medium|     83|      94.0%|  0.580|         4.84/5|
|**Multi-Hop**   |        Hard|     80|      92.5%|  0.437|         4.81/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  275.5s|
|**Questions/Second**   |    1.66|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.43%|
|**TPM Utilization**    |   0.43%|

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

*Report generated: 2025-12-26 08:12:07*
*Checkpoint: C047*
*Judge Model: gemini-3-flash-preview*