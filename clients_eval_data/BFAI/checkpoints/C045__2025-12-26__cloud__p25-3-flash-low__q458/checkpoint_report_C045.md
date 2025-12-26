# Checkpoint Report: C045

**Generated:** 2025-12-26 08:02:26
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.1% |
| **Partial Rate** | 6.3% |
| **Fail Rate** | 1.5% |
| **Acceptable Rate** | 98.5% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 92.1% | -1.1% | ✅ |
| **Acceptable Rate** | 97.6% | 98.5% | +0.9% | ✅ |
| **Fail Rate** | 2.4% | 1.5% | -0.9% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 4.82/5 | +0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 54.5s | -16.3s | 1.30x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8794 | $+0.0259 |
| **Per Question** | $0.001864 | $0.001920 | $+0.000056 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 92.8% | -3.2% |
| **Multi-Hop** | 90.7% | 91.5% | +0.8% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 90.7% | +1.9% |
| **Medium** | 97.5% | 91.9% | -5.6% |
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
|**MRR**         |  0.739|       1.000|      0.493|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.751|   0.783|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.85/5|     4.79/5|
|**Pass Rate**      |   92.1%|       92.8%|      91.5%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.80/5|  4.84/5|  4.81/5|
|**Pass Rate**      |   90.7%|   91.9%|   94.1%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.77/5|  4.74/5|  4.81/5|  4.76/5|
|**Completeness**  |  4.86/5|  4.81/5|  4.90/5|  4.86/5|
|**Faithfulness**  |  4.89/5|  4.88/5|  4.88/5|  4.90/5|
|**Relevance**     |  4.96/5|  4.94/5|  4.98/5|  4.95/5|
|**Clarity**       |  4.99/5|  5.00/5|  4.99/5|  4.96/5|
|**Overall**       |  4.82/5|  4.80/5|  4.84/5|  4.81/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.77/5|      4.82/5|     4.72/5|
|**Completeness**  |  4.86/5|      4.90/5|     4.82/5|
|**Faithfulness**  |  4.89/5|      4.86/5|     4.92/5|
|**Relevance**     |  4.96/5|      4.94/5|     4.97/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.97/5|
|**Overall**       |  4.82/5|      4.85/5|     4.79/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.83s|        3.4%|
|**Total**       |    54.48s|        100%|

**Min Latency:** 3.68s  |  **Max Latency:** 111.14s

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
|**Single-Hop**  |    222|   206|       11|     5|      92.8%|
|**Multi-Hop**   |    236|   216|       18|     2|      91.5%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   146|       10|     5|      90.7%|
|**Medium**  |    161|   148|       13|     0|      91.9%|
|**Hard**    |    136|   128|        6|     2|      94.1%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      93.2%|  1.000|         4.84/5|
|**Single-Hop**  |      Medium|     78|      89.7%|  1.000|         4.84/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.86/5|
|**Multi-Hop**   |        Easy|     73|      87.7%|  0.452|         4.75/5|
|**Multi-Hop**   |      Medium|     83|      94.0%|  0.580|         4.84/5|
|**Multi-Hop**   |        Hard|     80|      92.5%|  0.439|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  338.7s|
|**Questions/Second**   |    1.35|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.27%|
|**TPM Utilization**    |   0.27%|

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

*Report generated: 2025-12-26 08:02:26*
*Checkpoint: C045*
*Judge Model: gemini-3-flash-preview*