# Checkpoint Report: C065

**Generated:** 2025-12-26 14:04:50
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 91.3% |
| **Partial Rate** | 7.0% |
| **Fail Rate** | 1.7% |
| **Acceptable Rate** | 98.3% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 91.3% | -0.9% | ✅ |
| **Acceptable Rate** | 98.9% | 98.3% | -0.7% | ✅ |
| **Fail Rate** | 1.1% | 1.7% | +0.7% | ✅ |
| **MRR** | 0.740 | 0.738 | -0.002 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.81/5 | -0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 54.2s | -3.5s | 1.06x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.8774 | $+0.0110 |
| **Per Question** | $0.001892 | $0.001916 | $+0.000024 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 91.9% | -0.9% |
| **Multi-Hop** | 91.5% | 90.7% | -0.8% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 88.8% | -0.6% |
| **Medium** | 93.2% | 91.3% | -1.9% |
| **Hard** | 94.1% | 94.1% | +0.0% |

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
|**MRR**         |  0.738|       1.000|      0.491|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.751|   0.781|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.81/5|      4.84/5|     4.79/5|
|**Pass Rate**      |   91.3%|       91.9%|      90.7%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.77/5|  4.84/5|  4.83/5|
|**Pass Rate**      |   88.8%|   91.3%|   94.1%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.76/5|  4.72/5|  4.80/5|  4.76/5|
|**Completeness**  |  4.86/5|  4.83/5|  4.89/5|  4.86/5|
|**Faithfulness**  |  4.87/5|  4.81/5|  4.87/5|  4.96/5|
|**Relevance**     |  4.97/5|  4.95/5|  4.98/5|  4.97/5|
|**Clarity**       |  5.00/5|  5.00/5|  5.00/5|  4.99/5|
|**Overall**       |  4.81/5|  4.77/5|  4.84/5|  4.83/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.76/5|      4.81/5|     4.71/5|
|**Completeness**  |  4.86/5|      4.91/5|     4.82/5|
|**Faithfulness**  |  4.87/5|      4.82/5|     4.93/5|
|**Relevance**     |  4.97/5|      4.94/5|     5.00/5|
|**Clarity**       |  5.00/5|      5.00/5|     4.99/5|
|**Overall**       |  4.81/5|      4.84/5|     4.79/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.81s|        3.3%|
|**Total**       |    54.21s|        100%|

**Min Latency:** 3.06s  |  **Max Latency:** 107.05s

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
|**Single-Hop**  |    222|   204|       11|     7|      91.9%|
|**Multi-Hop**   |    236|   214|       21|     1|      90.7%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   143|       12|     6|      88.8%|
|**Medium**  |    161|   147|       13|     1|      91.3%|
|**Hard**    |    136|   128|        7|     1|      94.1%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      89.8%|  1.000|         4.80/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.86/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.88/5|
|**Multi-Hop**   |        Easy|     73|      87.7%|  0.452|         4.73/5|
|**Multi-Hop**   |      Medium|     83|      91.6%|  0.576|         4.82/5|
|**Multi-Hop**   |        Hard|     80|      92.5%|  0.439|         4.80/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  300.7s|
|**Questions/Second**   |    1.52|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.40%|
|**TPM Utilization**    |   0.40%|

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

*Report generated: 2025-12-26 14:04:50*
*Checkpoint: C065*
*Judge Model: gemini-3-flash-preview*