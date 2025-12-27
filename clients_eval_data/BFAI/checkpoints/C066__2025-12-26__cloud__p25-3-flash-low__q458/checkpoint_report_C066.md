# Checkpoint Report: C066

**Generated:** 2025-12-26 14:29:36
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.1% |
| **Partial Rate** | 6.3% |
| **Fail Rate** | 1.5% |
| **Acceptable Rate** | 98.5% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 92.1% | +0.0% | ✅ |
| **Acceptable Rate** | 98.9% | 98.5% | -0.4% | ✅ |
| **Fail Rate** | 1.1% | 1.5% | +0.4% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.81/5 | -0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 58.9s | +1.2s | 0.98x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.8666 | $+0.0002 |
| **Per Question** | $0.001892 | $0.001892 | $+0.000001 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 92.8% | +0.0% |
| **Multi-Hop** | 91.5% | 91.5% | +0.0% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 89.4% | +0.0% |
| **Medium** | 93.2% | 93.2% | +0.0% |
| **Hard** | 94.1% | 94.1% | +0.0% |

### Key Finding

✅ **All metrics within acceptable range of baseline.**

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-app-xcshqh7sqq-ue.a.run.app |
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
|**MRR**         |   0.755|   0.781|  0.671|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.81/5|      4.84/5|     4.78/5|
|**Pass Rate**      |   92.1%|       92.8%|      91.5%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.77/5|  4.85/5|  4.82/5|
|**Pass Rate**      |   89.4%|   93.2%|   94.1%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.77/5|  4.72/5|  4.81/5|  4.77/5|
|**Completeness**  |  4.86/5|  4.83/5|  4.88/5|  4.86/5|
|**Faithfulness**  |  4.88/5|  4.86/5|  4.89/5|  4.90/5|
|**Relevance**     |  4.96/5|  4.94/5|  4.98/5|  4.95/5|
|**Clarity**       |  4.98/5|  5.00/5|  4.98/5|  4.96/5|
|**Overall**       |  4.81/5|  4.77/5|  4.85/5|  4.82/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.77/5|      4.82/5|     4.72/5|
|**Completeness**  |  4.86/5|      4.90/5|     4.82/5|
|**Faithfulness**  |  4.88/5|      4.84/5|     4.92/5|
|**Relevance**     |  4.96/5|      4.93/5|     4.98/5|
|**Clarity**       |  4.98/5|      4.99/5|     4.97/5|
|**Overall**       |  4.81/5|      4.84/5|     4.78/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.75s|        3.0%|
|**Total**       |    58.93s|        100%|

**Min Latency:** 8.37s  |  **Max Latency:** 116.65s

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
|**Single-Hop**  |    222|   206|       10|     6|      92.8%|
|**Multi-Hop**   |    236|   216|       19|     1|      91.5%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   144|       13|     4|      89.4%|
|**Medium**  |    161|   150|       10|     1|      93.2%|
|**Hard**    |    136|   128|        6|     2|      94.1%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      92.0%|  1.000|         4.82/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.84/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.87/5|
|**Multi-Hop**   |        Easy|     73|      86.3%|  0.459|         4.71/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.576|         4.85/5|
|**Multi-Hop**   |        Hard|     80|      92.5%|  0.440|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  347.3s|
|**Questions/Second**   |    1.32|
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
|**Endpoint**      |  https://bfai-app-xcshqh7sqq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-26 14:29:36*
*Checkpoint: C066*
*Judge Model: gemini-3-flash-preview*