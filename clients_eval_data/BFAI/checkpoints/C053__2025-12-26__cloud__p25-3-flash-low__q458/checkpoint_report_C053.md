# Checkpoint Report: C053

**Generated:** 2025-12-26 09:40:04
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 91.5% |
| **Partial Rate** | 6.6% |
| **Fail Rate** | 2.0% |
| **Acceptable Rate** | 98.0% |
| **Overall Score** | 4.79/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 91.5% | -1.7% | ✅ |
| **Acceptable Rate** | 97.6% | 98.0% | +0.4% | ✅ |
| **Fail Rate** | 2.4% | 2.0% | -0.4% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 4.79/5 | -0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 60.7s | -10.1s | 1.17x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8879 | $+0.0343 |
| **Per Question** | $0.001864 | $0.001939 | $+0.000075 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 93.2% | -2.7% |
| **Multi-Hop** | 90.7% | 89.8% | -0.8% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 86.3% | -2.5% |
| **Medium** | 97.5% | 95.0% | -2.5% |
| **Hard** | 93.4% | 93.4% | +0.0% |

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
|**MRR**         |   0.752|   0.786|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.79/5|      4.84/5|     4.75/5|
|**Pass Rate**      |   91.5%|       93.2%|      89.8%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.71/5|  4.86/5|  4.82/5|
|**Pass Rate**      |   86.3%|   95.0%|   93.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.74/5|  4.66/5|  4.81/5|  4.76/5|
|**Completeness**  |  4.83/5|  4.76/5|  4.90/5|  4.84/5|
|**Faithfulness**  |  4.85/5|  4.75/5|  4.90/5|  4.92/5|
|**Relevance**     |  4.96/5|  4.93/5|  4.98/5|  4.98/5|
|**Clarity**       |  4.99/5|  4.99/5|  4.99/5|  4.99/5|
|**Overall**       |  4.79/5|  4.71/5|  4.86/5|  4.82/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.74/5|      4.81/5|     4.68/5|
|**Completeness**  |  4.83/5|      4.88/5|     4.79/5|
|**Faithfulness**  |  4.85/5|      4.78/5|     4.92/5|
|**Relevance**     |  4.96/5|      4.95/5|     4.97/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.98/5|
|**Overall**       |  4.79/5|      4.84/5|     4.75/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.93s|        3.2%|
|**Total**       |    60.66s|        100%|

**Min Latency:** 24.06s  |  **Max Latency:** 120.41s

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
|**Single-Hop**  |    222|   207|        9|     6|      93.2%|
|**Multi-Hop**   |    236|   212|       21|     3|      89.8%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   139|       15|     7|      86.3%|
|**Medium**  |    161|   153|        7|     1|      95.0%|
|**Hard**    |    136|   127|        8|     1|      93.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      89.8%|  1.000|         4.77/5|
|**Single-Hop**  |      Medium|     78|      94.9%|  1.000|         4.87/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.90/5|
|**Multi-Hop**   |        Easy|     73|      82.2%|  0.452|         4.63/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.586|         4.85/5|
|**Multi-Hop**   |        Hard|     80|      91.2%|  0.437|         4.76/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  433.1s|
|**Questions/Second**   |    1.06|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.55%|
|**TPM Utilization**    |   0.55%|

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

*Report generated: 2025-12-26 09:40:04*
*Checkpoint: C053*
*Judge Model: gemini-3-flash-preview*