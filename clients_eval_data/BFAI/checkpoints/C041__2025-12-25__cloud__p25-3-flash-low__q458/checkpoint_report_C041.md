# Checkpoint Report: C041

**Generated:** 2025-12-25 23:02:07
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 56.8% |
| **Partial Rate** | 18.8% |
| **Fail Rate** | 24.5% |
| **Acceptable Rate** | 75.5% |
| **Overall Score** | 3.90/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 56.8% | -36.5% | ❌ |
| **Acceptable Rate** | 97.6% | 75.5% | -22.1% | ❌ |
| **Fail Rate** | 2.4% | 24.5% | +22.1% | ❌ |
| **MRR** | 0.740 | 0.740 | +0.000 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 3.90/5 | -0.91 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 58.3s | -12.5s | 1.21x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8677 | $+0.0141 |
| **Per Question** | $0.001864 | $0.001894 | $+0.000031 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 64.4% | -31.5% |
| **Multi-Hop** | 90.7% | 49.6% | -41.1% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 55.9% | -32.9% |
| **Medium** | 97.5% | 54.7% | -42.9% |
| **Hard** | 93.4% | 60.3% | -33.1% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 36.5%, acceptable rate dropped 22.1%

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
|**Overall Score**  |  3.90/5|      4.13/5|     3.68/5|
|**Pass Rate**      |   56.8%|       64.4%|      49.6%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  3.89/5|  3.93/5|  3.88/5|
|**Pass Rate**      |   55.9%|   54.7%|   60.3%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  3.61/5|  3.51/5|  3.61/5|  3.74/5|
|**Completeness**  |  3.68/5|  3.61/5|  3.70/5|  3.73/5|
|**Faithfulness**  |  4.75/5|  4.71/5|  4.75/5|  4.81/5|
|**Relevance**     |  4.29/5|  4.36/5|  4.29/5|  4.22/5|
|**Clarity**       |  4.94/5|  4.97/5|  4.94/5|  4.91/5|
|**Overall**       |  3.90/5|  3.89/5|  3.93/5|  3.88/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  3.61/5|      3.80/5|     3.44/5|
|**Completeness**  |  3.68/5|      3.93/5|     3.44/5|
|**Faithfulness**  |  4.75/5|      4.74/5|     4.76/5|
|**Relevance**     |  4.29/5|      4.42/5|     4.17/5|
|**Clarity**       |  4.94/5|      4.96/5|     4.92/5|
|**Overall**       |  3.90/5|      4.13/5|     3.68/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.82s|        3.1%|
|**Total**       |    58.33s|        100%|

**Min Latency:** 10.66s  |  **Max Latency:** 121.56s

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
|**Single-Hop**  |    222|   143|       22|    57|      64.4%|
|**Multi-Hop**   |    236|   117|       64|    55|      49.6%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|    90|       26|    45|      55.9%|
|**Medium**  |    161|    88|       39|    34|      54.7%|
|**Hard**    |    136|    82|       21|    33|      60.3%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      62.5%|  1.000|         4.08/5|
|**Single-Hop**  |      Medium|     78|      57.7%|  1.000|         4.06/5|
|**Single-Hop**  |        Hard|     56|      76.8%|  1.000|         4.33/5|
|**Multi-Hop**   |        Easy|     73|      47.9%|  0.459|         3.66/5|
|**Multi-Hop**   |      Medium|     83|      51.8%|  0.582|         3.81/5|
|**Multi-Hop**   |        Hard|     80|      48.8%|  0.439|         3.56/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  352.9s|
|**Questions/Second**   |    1.30|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.24%|
|**TPM Utilization**    |   0.24%|

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

*Report generated: 2025-12-25 23:02:07*
*Checkpoint: C041*
*Judge Model: gemini-3-flash-preview*