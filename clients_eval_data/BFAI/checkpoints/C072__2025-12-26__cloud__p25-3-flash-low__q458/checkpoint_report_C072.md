# Checkpoint Report: C072

**Generated:** 2025-12-26 16:11:34
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 88.0% |
| **Partial Rate** | 8.5% |
| **Fail Rate** | 3.5% |
| **Acceptable Rate** | 96.5% |
| **Overall Score** | 4.73/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 88.0% | -4.1% | ⚠️ |
| **Acceptable Rate** | 98.9% | 96.5% | -2.4% | ❌ |
| **Fail Rate** | 1.1% | 3.5% | +2.4% | ❌ |
| **MRR** | 0.740 | 0.737 | -0.003 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.73/5 | -0.09 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 74.5s | +16.8s | 0.77x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.9612 | $+0.0949 |
| **Per Question** | $0.001892 | $0.002099 | $+0.000207 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 89.6% | -3.2% |
| **Multi-Hop** | 91.5% | 86.4% | -5.1% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 83.9% | -5.6% |
| **Medium** | 93.2% | 90.1% | -3.1% |
| **Hard** | 94.1% | 90.4% | -3.7% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 4.1%, acceptable rate dropped 2.4%, latency increased 16.8s

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
| **Model** | gemini-2.5-flash |
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
|**MRR**         |   0.755|   0.777|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.73/5|      4.76/5|     4.70/5|
|**Pass Rate**      |   88.0%|       89.6%|      86.4%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.64/5|  4.80/5|  4.75/5|
|**Pass Rate**      |   83.9%|   90.1%|   90.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.70/5|  4.63/5|  4.76/5|  4.71/5|
|**Completeness**  |  4.79/5|  4.71/5|  4.86/5|  4.81/5|
|**Faithfulness**  |  4.75/5|  4.64/5|  4.76/5|  4.87/5|
|**Relevance**     |  4.95/5|  4.96/5|  4.96/5|  4.93/5|
|**Clarity**       |  4.98/5|  4.98/5|  4.98/5|  4.99/5|
|**Overall**       |  4.73/5|  4.64/5|  4.80/5|  4.75/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.70/5|      4.77/5|     4.64/5|
|**Completeness**  |  4.79/5|      4.84/5|     4.75/5|
|**Faithfulness**  |  4.75/5|      4.66/5|     4.83/5|
|**Relevance**     |  4.95/5|      4.93/5|     4.97/5|
|**Clarity**       |  4.98/5|      4.98/5|     4.98/5|
|**Overall**       |  4.73/5|      4.76/5|     4.70/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     2.19s|        2.9%|
|**Total**       |    74.50s|        100%|

**Min Latency:** 12.20s  |  **Max Latency:** 142.78s

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
|**Single-Hop**  |    222|   199|       12|    11|      89.6%|
|**Multi-Hop**   |    236|   204|       27|     5|      86.4%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   135|       17|     9|      83.9%|
|**Medium**  |    161|   145|       12|     4|      90.1%|
|**Hard**    |    136|   123|       10|     3|      90.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      86.4%|  1.000|         4.70/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.82/5|
|**Single-Hop**  |        Hard|     56|      92.9%|  1.000|         4.78/5|
|**Multi-Hop**   |        Easy|     73|      80.8%|  0.459|         4.57/5|
|**Multi-Hop**   |      Medium|     83|      89.2%|  0.568|         4.78/5|
|**Multi-Hop**   |        Hard|     80|      88.8%|  0.437|         4.73/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  441.4s|
|**Questions/Second**   |    1.04|
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
|**Endpoint**      |  https://bfai-app-xcshqh7sqq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-26 16:11:34*
*Checkpoint: C072*
*Judge Model: gemini-3-flash-preview*