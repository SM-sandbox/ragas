# Checkpoint Report: C044

**Generated:** 2025-12-26 07:25:04
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 91.3% |
| **Partial Rate** | 7.4% |
| **Fail Rate** | 1.3% |
| **Acceptable Rate** | 98.7% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 91.3% | -2.0% | ✅ |
| **Acceptable Rate** | 97.6% | 98.7% | +1.1% | ✅ |
| **Fail Rate** | 2.4% | 1.3% | -1.1% | ✅ |
| **MRR** | 0.740 | 0.737 | -0.002 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 4.81/5 | +0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 53.7s | -17.1s | 1.32x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8808 | $+0.0272 |
| **Per Question** | $0.001864 | $0.001923 | $+0.000059 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 91.9% | -4.1% |
| **Multi-Hop** | 90.7% | 90.7% | +0.0% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 88.8% | +0.0% |
| **Medium** | 97.5% | 92.5% | -5.0% |
| **Hard** | 93.4% | 92.6% | -0.7% |

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
|**MRR**         |  0.737|       1.000|      0.490|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.752|   0.781|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.81/5|      4.85/5|     4.77/5|
|**Pass Rate**      |   91.3%|       91.9%|      90.7%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.77/5|  4.85/5|  4.81/5|
|**Pass Rate**      |   88.8%|   92.5%|   92.6%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.76/5|  4.73/5|  4.80/5|  4.75/5|
|**Completeness**  |  4.87/5|  4.84/5|  4.92/5|  4.85/5|
|**Faithfulness**  |  4.85/5|  4.81/5|  4.88/5|  4.86/5|
|**Relevance**     |  4.96/5|  4.94/5|  4.98/5|  4.95/5|
|**Clarity**       |  4.99/5|  5.00/5|  5.00/5|  4.96/5|
|**Overall**       |  4.81/5|  4.77/5|  4.85/5|  4.81/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.76/5|      4.83/5|     4.69/5|
|**Completeness**  |  4.87/5|      4.91/5|     4.83/5|
|**Faithfulness**  |  4.85/5|      4.81/5|     4.88/5|
|**Relevance**     |  4.96/5|      4.94/5|     4.98/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.98/5|
|**Overall**       |  4.81/5|      4.85/5|     4.77/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.82s|        3.4%|
|**Total**       |    53.71s|        100%|

**Min Latency:** 4.09s  |  **Max Latency:** 111.32s

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
|**Single-Hop**  |    222|   204|       14|     4|      91.9%|
|**Multi-Hop**   |    236|   214|       20|     2|      90.7%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   143|       14|     4|      88.8%|
|**Medium**  |    161|   149|       12|     0|      92.5%|
|**Hard**    |    136|   126|        8|     2|      92.6%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      92.0%|  1.000|         4.84/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.86/5|
|**Single-Hop**  |        Hard|     56|      92.9%|  1.000|         4.86/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.452|         4.69/5|
|**Multi-Hop**   |      Medium|     83|      94.0%|  0.576|         4.84/5|
|**Multi-Hop**   |        Hard|     80|      92.5%|  0.437|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  310.9s|
|**Questions/Second**   |    1.47|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.33%|
|**TPM Utilization**    |   0.33%|

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

*Report generated: 2025-12-26 07:25:04*
*Checkpoint: C044*
*Judge Model: gemini-3-flash-preview*