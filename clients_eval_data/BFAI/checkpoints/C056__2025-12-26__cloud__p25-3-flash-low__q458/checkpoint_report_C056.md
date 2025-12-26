# Checkpoint Report: C056

**Generated:** 2025-12-26 09:49:38
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.1% |
| **Partial Rate** | 6.6% |
| **Fail Rate** | 1.3% |
| **Acceptable Rate** | 98.7% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 92.1% | -1.1% | ✅ |
| **Acceptable Rate** | 97.6% | 98.7% | +1.1% | ✅ |
| **Fail Rate** | 2.4% | 1.3% | -1.1% | ✅ |
| **MRR** | 0.740 | 0.741 | +0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 4.81/5 | +0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 51.5s | -19.3s | 1.37x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8753 | $+0.0217 |
| **Per Question** | $0.001864 | $0.001911 | $+0.000047 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 93.2% | -2.7% |
| **Multi-Hop** | 90.7% | 91.1% | +0.4% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 88.8% | +0.0% |
| **Medium** | 97.5% | 93.8% | -3.7% |
| **Hard** | 93.4% | 94.1% | +0.7% |

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
|**MRR**         |  0.741|       1.000|      0.498|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.788|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.81/5|      4.85/5|     4.79/5|
|**Pass Rate**      |   92.1%|       93.2%|      91.1%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.76/5|  4.86/5|  4.83/5|
|**Pass Rate**      |   88.8%|   93.8%|   94.1%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.77/5|  4.73/5|  4.81/5|  4.76/5|
|**Completeness**  |  4.86/5|  4.81/5|  4.92/5|  4.87/5|
|**Faithfulness**  |  4.89/5|  4.84/5|  4.90/5|  4.92/5|
|**Relevance**     |  4.97/5|  4.95/5|  4.99/5|  4.98/5|
|**Clarity**       |  4.99/5|  5.00/5|  4.99/5|  4.99/5|
|**Overall**       |  4.81/5|  4.76/5|  4.86/5|  4.83/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.77/5|      4.82/5|     4.72/5|
|**Completeness**  |  4.86/5|      4.91/5|     4.82/5|
|**Faithfulness**  |  4.89/5|      4.85/5|     4.92/5|
|**Relevance**     |  4.97/5|      4.95/5|     5.00/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.99/5|
|**Overall**       |  4.81/5|      4.85/5|     4.79/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.78s|        3.5%|
|**Total**       |    51.50s|        100%|

**Min Latency:** 5.63s  |  **Max Latency:** 104.86s

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
|**Single-Hop**  |    222|   207|       10|     5|      93.2%|
|**Multi-Hop**   |    236|   215|       20|     1|      91.1%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   143|       13|     5|      88.8%|
|**Medium**  |    161|   151|       10|     0|      93.8%|
|**Hard**    |    136|   128|        7|     1|      94.1%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      92.0%|  1.000|         4.83/5|
|**Single-Hop**  |      Medium|     78|      92.3%|  1.000|         4.86/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.86/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.459|         4.68/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.588|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      92.5%|  0.439|         4.81/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  286.3s|
|**Questions/Second**   |    1.60|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.41%|
|**TPM Utilization**    |   0.41%|

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

*Report generated: 2025-12-26 09:49:38*
*Checkpoint: C056*
*Judge Model: gemini-3-flash-preview*