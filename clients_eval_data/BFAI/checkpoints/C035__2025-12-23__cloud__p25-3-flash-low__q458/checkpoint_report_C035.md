# Checkpoint Report: C035

**Generated:** 2025-12-23 14:12:19
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 93.2% |
| **Partial Rate** | 4.6% |
| **Fail Rate** | 2.2% |
| **Acceptable Rate** | 97.8% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 93.2% | +0.0% | ✅ |
| **Acceptable Rate** | 97.6% | 97.8% | +0.2% | ✅ |
| **Fail Rate** | 2.4% | 2.2% | -0.2% | ✅ |
| **MRR** | 0.740 | 0.738 | -0.002 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 4.81/5 | +0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 68.5s | -2.3s | 1.03x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8471 | $-0.0065 |
| **Per Question** | $0.001864 | $0.001849 | $-0.000014 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 95.5% | -0.5% |
| **Multi-Hop** | 90.7% | 91.1% | +0.4% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 89.4% | +0.6% |
| **Medium** | 97.5% | 97.5% | +0.0% |
| **Hard** | 93.4% | 92.6% | -0.7% |

### Key Finding

✅ **All metrics within acceptable range of baseline.**

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
|**Overall Score**  |  4.81/5|      4.84/5|     4.78/5|
|**Pass Rate**      |   93.2%|       95.5%|      91.1%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.74/5|  4.89/5|  4.80/5|
|**Pass Rate**      |   89.4%|   97.5%|   92.6%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.78/5|  4.70/5|  4.87/5|  4.75/5|
|**Completeness**  |  4.86/5|  4.78/5|  4.96/5|  4.83/5|
|**Faithfulness**  |  4.85/5|  4.78/5|  4.93/5|  4.85/5|
|**Relevance**     |  4.95/5|  4.91/5|  4.99/5|  4.96/5|
|**Clarity**       |  4.98/5|  4.99/5|  5.00/5|  4.95/5|
|**Overall**       |  4.81/5|  4.74/5|  4.89/5|  4.80/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.78/5|      4.83/5|     4.72/5|
|**Completeness**  |  4.86/5|      4.90/5|     4.82/5|
|**Faithfulness**  |  4.85/5|      4.80/5|     4.90/5|
|**Relevance**     |  4.95/5|      4.95/5|     4.95/5|
|**Clarity**       |  4.98/5|      5.00/5|     4.97/5|
|**Overall**       |  4.81/5|      4.84/5|     4.78/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.84s|        2.7%|
|**Total**       |    68.53s|        100%|

**Min Latency:** 11.40s  |  **Max Latency:** 146.15s

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
|**Single-Hop**  |    222|   212|        4|     6|      95.5%|
|**Multi-Hop**   |    236|   215|       17|     4|      91.1%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   144|       11|     6|      89.4%|
|**Medium**  |    161|   157|        3|     1|      97.5%|
|**Hard**    |    136|   126|        7|     3|      92.6%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      90.9%|  1.000|         4.75/5|
|**Single-Hop**  |      Medium|     78|     100.0%|  1.000|         4.92/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.87/5|
|**Multi-Hop**   |        Easy|     73|      87.7%|  0.452|         4.72/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.576|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      90.0%|  0.438|         4.76/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  461.2s|
|**Questions/Second**   |    0.99|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.17%|
|**TPM Utilization**    |   0.17%|

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

*Report generated: 2025-12-23 14:12:19*
*Checkpoint: C035*
*Judge Model: gemini-3-flash-preview*