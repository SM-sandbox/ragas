# Checkpoint Report: C068

**Generated:** 2025-12-26 15:02:08
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 87.8% |
| **Partial Rate** | 8.7% |
| **Fail Rate** | 3.5% |
| **Acceptable Rate** | 96.5% |
| **Overall Score** | 4.72/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 87.8% | -4.4% | ⚠️ |
| **Acceptable Rate** | 98.9% | 96.5% | -2.4% | ❌ |
| **Fail Rate** | 1.1% | 3.5% | +2.4% | ❌ |
| **MRR** | 0.740 | 0.737 | -0.003 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.72/5 | -0.10 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 73.6s | +15.9s | 0.78x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.9624 | $+0.0961 |
| **Per Question** | $0.001892 | $0.002101 | $+0.000210 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 88.3% | -4.5% |
| **Multi-Hop** | 91.5% | 87.3% | -4.2% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 84.5% | -5.0% |
| **Medium** | 93.2% | 90.7% | -2.5% |
| **Hard** | 94.1% | 88.2% | -5.9% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 4.4%, acceptable rate dropped 2.4%, latency increased 15.9s

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
|**MRR**         |   0.751|   0.780|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.72/5|      4.73/5|     4.72/5|
|**Pass Rate**      |   87.8%|       88.3%|      87.3%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.66/5|  4.78/5|  4.73/5|
|**Pass Rate**      |   84.5%|   90.7%|   88.2%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.69/5|  4.64/5|  4.76/5|  4.68/5|
|**Completeness**  |  4.78/5|  4.73/5|  4.83/5|  4.79/5|
|**Faithfulness**  |  4.73/5|  4.63/5|  4.75/5|  4.84/5|
|**Relevance**     |  4.94/5|  4.94/5|  4.94/5|  4.93/5|
|**Clarity**       |  4.98/5|  4.99/5|  4.98/5|  4.99/5|
|**Overall**       |  4.72/5|  4.66/5|  4.78/5|  4.73/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.69/5|      4.74/5|     4.65/5|
|**Completeness**  |  4.78/5|      4.82/5|     4.74/5|
|**Faithfulness**  |  4.73/5|      4.63/5|     4.83/5|
|**Relevance**     |  4.94/5|      4.92/5|     4.96/5|
|**Clarity**       |  4.98/5|      4.99/5|     4.98/5|
|**Overall**       |  4.72/5|      4.73/5|     4.72/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     2.16s|        2.9%|
|**Total**       |    73.62s|        100%|

**Min Latency:** 13.19s  |  **Max Latency:** 140.64s

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
|**Single-Hop**  |    222|   196|       15|    11|      88.3%|
|**Multi-Hop**   |    236|   206|       25|     5|      87.3%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   136|       16|     9|      84.5%|
|**Medium**  |    161|   146|       11|     4|      90.7%|
|**Hard**    |    136|   120|       13|     3|      88.2%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      85.2%|  1.000|         4.69/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.76/5|
|**Single-Hop**  |        Hard|     56|      89.3%|  1.000|         4.75/5|
|**Multi-Hop**   |        Easy|     73|      83.6%|  0.452|         4.62/5|
|**Multi-Hop**   |      Medium|     83|      90.4%|  0.574|         4.80/5|
|**Multi-Hop**   |        Hard|     80|      87.5%|  0.438|         4.71/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  479.3s|
|**Questions/Second**   |    0.96|
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
|**Endpoint**      |  https://bfai-app-eb2qyzdzvq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-26 15:02:08*
*Checkpoint: C068*
*Judge Model: gemini-3-flash-preview*