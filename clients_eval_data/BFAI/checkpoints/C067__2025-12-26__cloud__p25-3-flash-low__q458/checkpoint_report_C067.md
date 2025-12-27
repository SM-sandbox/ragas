# Checkpoint Report: C067

**Generated:** 2025-12-26 14:35:32
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 90.6% |
| **Partial Rate** | 7.4% |
| **Fail Rate** | 2.0% |
| **Acceptable Rate** | 98.0% |
| **Overall Score** | 4.79/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 90.6% | -1.5% | ✅ |
| **Acceptable Rate** | 98.9% | 98.0% | -0.9% | ✅ |
| **Fail Rate** | 1.1% | 2.0% | +0.9% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.79/5 | -0.03 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 61.2s | +3.5s | 0.94x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.8991 | $+0.0327 |
| **Per Question** | $0.001892 | $0.001963 | $+0.000072 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 91.4% | -1.4% |
| **Multi-Hop** | 91.5% | 89.8% | -1.7% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 87.6% | -1.9% |
| **Medium** | 93.2% | 91.3% | -1.9% |
| **Hard** | 94.1% | 93.4% | -0.7% |

### Key Finding

✅ **All metrics within acceptable range of baseline.**

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

|Metric          |  Total|  Single-Hop|  Multi-Hop|
|----------------|------:|-----------:|----------:|
|**Recall@100**  |  99.1%|      100.0%|      98.3%|
|**MRR**         |  0.739|       1.000|      0.493|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.752|   0.784|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.79/5|      4.82/5|     4.77/5|
|**Pass Rate**      |   90.6%|       91.4%|      89.8%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.73/5|  4.83/5|  4.82/5|
|**Pass Rate**      |   87.6%|   91.3%|   93.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.74/5|  4.68/5|  4.79/5|  4.76/5|
|**Completeness**  |  4.85/5|  4.83/5|  4.86/5|  4.86/5|
|**Faithfulness**  |  4.85/5|  4.77/5|  4.88/5|  4.90/5|
|**Relevance**     |  4.98/5|  4.99/5|  4.97/5|  4.97/5|
|**Clarity**       |  4.99/5|  5.00/5|  4.99/5|  4.99/5|
|**Overall**       |  4.79/5|  4.73/5|  4.83/5|  4.82/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.74/5|      4.79/5|     4.70/5|
|**Completeness**  |  4.85/5|      4.89/5|     4.81/5|
|**Faithfulness**  |  4.85/5|      4.77/5|     4.93/5|
|**Relevance**     |  4.98/5|      4.98/5|     4.97/5|
|**Clarity**       |  4.99/5|      4.99/5|     4.99/5|
|**Overall**       |  4.79/5|      4.82/5|     4.77/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.84s|        3.0%|
|**Total**       |    61.20s|        100%|

**Min Latency:** 6.83s  |  **Max Latency:** 109.91s

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
|**Single-Hop**  |    222|   203|       11|     8|      91.4%|
|**Multi-Hop**   |    236|   212|       23|     1|      89.8%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   141|       13|     7|      87.6%|
|**Medium**  |    161|   147|       13|     1|      91.3%|
|**Hard**    |    136|   127|        8|     1|      93.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      87.5%|  1.000|         4.73/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.84/5|
|**Single-Hop**  |        Hard|     56|      98.2%|  1.000|         4.92/5|
|**Multi-Hop**   |        Easy|     73|      87.7%|  0.452|         4.73/5|
|**Multi-Hop**   |      Medium|     83|      91.6%|  0.582|         4.82/5|
|**Multi-Hop**   |        Hard|     80|      90.0%|  0.438|         4.76/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  356.2s|
|**Questions/Second**   |    1.29|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.29%|
|**TPM Utilization**    |   0.29%|

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

*Report generated: 2025-12-26 14:35:32*
*Checkpoint: C067*
*Judge Model: gemini-3-flash-preview*