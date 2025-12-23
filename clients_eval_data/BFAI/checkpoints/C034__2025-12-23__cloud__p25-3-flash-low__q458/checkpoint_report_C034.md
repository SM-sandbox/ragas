# Checkpoint Report: C034

**Generated:** 2025-12-23 11:23:45
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 93.2% |
| **Partial Rate** | 4.4% |
| **Fail Rate** | 2.4% |
| **Acceptable Rate** | 97.6% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 93.2% | +0.4% | ✅ |
| **Acceptable Rate** | 98.5% | 97.6% | -0.9% | ✅ |
| **Fail Rate** | 1.5% | 2.4% | +0.9% | ✅ |
| **MRR** | 0.737 | 0.740 | +0.003 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.81/5 | -0.03 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 70.8s | +28.0s | 0.60x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.8536 | $+0.0102 |
| **Per Question** | $0.001841 | $0.001864 | $+0.000022 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 95.9% | +0.5% |
| **Multi-Hop** | 90.3% | 90.7% | +0.4% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 88.8% | -1.9% |
| **Medium** | 95.0% | 97.5% | +2.5% |
| **Hard** | 92.6% | 93.4% | +0.7% |

### Key Finding

⚠️ **Regressions detected:** latency increased 28.0s

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-api-ppfq5ahfsq-ue.a.run.app |
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
|**MRR**         |  0.740|       1.000|      0.495|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.784|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.81/5|      4.84/5|     4.77/5|
|**Pass Rate**      |   93.2%|       95.9%|      90.7%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.72/5|  4.89/5|  4.81/5|
|**Pass Rate**      |   88.8%|   97.5%|   93.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.77/5|  4.68/5|  4.87/5|  4.75/5|
|**Completeness**  |  4.85/5|  4.76/5|  4.96/5|  4.83/5|
|**Faithfulness**  |  4.85/5|  4.76/5|  4.93/5|  4.85/5|
|**Relevance**     |  4.94/5|  4.89/5|  4.99/5|  4.96/5|
|**Clarity**       |  4.97/5|  4.97/5|  5.00/5|  4.95/5|
|**Overall**       |  4.81/5|  4.72/5|  4.89/5|  4.81/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.77/5|      4.83/5|     4.71/5|
|**Completeness**  |  4.85/5|      4.90/5|     4.81/5|
|**Faithfulness**  |  4.85/5|      4.81/5|     4.89/5|
|**Relevance**     |  4.94/5|      4.95/5|     4.94/5|
|**Clarity**       |  4.97/5|      5.00/5|     4.95/5|
|**Overall**       |  4.81/5|      4.84/5|     4.77/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.84s|        2.6%|
|**Total**       |    70.79s|        100%|

**Min Latency:** 10.04s  |  **Max Latency:** 133.72s

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
|**Single-Hop**  |    222|   213|        3|     6|      95.9%|
|**Multi-Hop**   |    236|   214|       17|     5|      90.7%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   143|       11|     7|      88.8%|
|**Medium**  |    161|   157|        3|     1|      97.5%|
|**Hard**    |    136|   127|        6|     3|      93.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      90.9%|  1.000|         4.75/5|
|**Single-Hop**  |      Medium|     78|     100.0%|  1.000|         4.92/5|
|**Single-Hop**  |        Hard|     56|      98.2%|  1.000|         4.88/5|
|**Multi-Hop**   |        Easy|     73|      86.3%|  0.459|         4.67/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.582|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      90.0%|  0.438|         4.76/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  498.9s|
|**Questions/Second**   |    0.92|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.07%|
|**TPM Utilization**    |   0.07%|

## Index & Orchestrator

|Field             |                                     Value|
|------------------|-----------------------------------------:|
|**Index/Job ID**  |                  bfai__eval66a_g1_1536_tt|
|**Mode**          |                                     cloud|
|**Endpoint**      |  https://bfai-api-ppfq5ahfsq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-23 11:23:45*
*Checkpoint: C034*
*Judge Model: gemini-3-flash-preview*