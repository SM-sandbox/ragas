# Checkpoint Report: C032

**Generated:** 2025-12-23 08:52:10
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 87.8% |
| **Partial Rate** | 9.0% |
| **Fail Rate** | 3.3% |
| **Acceptable Rate** | 96.7% |
| **Overall Score** | 4.71/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 87.8% | -5.0% | ❌ |
| **Acceptable Rate** | 98.5% | 96.7% | -1.7% | ⚠️ |
| **Fail Rate** | 1.5% | 3.3% | +1.7% | ⚠️ |
| **MRR** | 0.737 | 0.744 | +0.006 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.71/5 | -0.12 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 87.9s | +45.2s | 0.49x ❌ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.9467 | $+0.1033 |
| **Per Question** | $0.001841 | $0.002067 | $+0.000226 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 89.6% | -5.9% |
| **Multi-Hop** | 90.3% | 86.0% | -4.2% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 82.0% | -8.7% |
| **Medium** | 95.0% | 92.5% | -2.5% |
| **Hard** | 92.6% | 89.0% | -3.7% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 5.0%, acceptable rate dropped 1.7%, latency increased 45.2s

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
|**MRR**         |  0.744|       1.000|      0.503|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.763|   0.785|  0.672|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.71/5|      4.75/5|     4.67/5|
|**Pass Rate**      |   87.8%|       89.6%|      86.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.61/5|  4.79/5|  4.74/5|
|**Pass Rate**      |   82.0%|   92.5%|   89.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.67/5|  4.63/5|  4.73/5|  4.66/5|
|**Completeness**  |  4.79/5|  4.75/5|  4.83/5|  4.79/5|
|**Faithfulness**  |  4.76/5|  4.53/5|  4.86/5|  4.90/5|
|**Relevance**     |  4.94/5|  4.93/5|  4.97/5|  4.93/5|
|**Clarity**       |  4.98/5|  4.98/5|  4.99/5|  4.99/5|
|**Overall**       |  4.71/5|  4.61/5|  4.79/5|  4.74/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.67/5|      4.73/5|     4.62/5|
|**Completeness**  |  4.79/5|      4.84/5|     4.74/5|
|**Faithfulness**  |  4.76/5|      4.73/5|     4.78/5|
|**Relevance**     |  4.94/5|      4.95/5|     4.94/5|
|**Clarity**       |  4.98/5|      4.98/5|     4.99/5|
|**Overall**       |  4.71/5|      4.75/5|     4.67/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     2.36s|        2.7%|
|**Total**       |    87.90s|        100%|

**Min Latency:** 15.13s  |  **Max Latency:** 157.24s

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
|**Single-Hop**  |    222|   199|       13|    10|      89.6%|
|**Multi-Hop**   |    236|   203|       28|     5|      86.0%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   132|       22|     7|      82.0%|
|**Medium**  |    161|   149|        6|     6|      92.5%|
|**Hard**    |    136|   121|       13|     2|      89.0%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      86.4%|  1.000|         4.69/5|
|**Single-Hop**  |      Medium|     78|      92.3%|  1.000|         4.77/5|
|**Single-Hop**  |        Hard|     56|      91.1%|  1.000|         4.81/5|
|**Multi-Hop**   |        Easy|     73|      76.7%|  0.477|         4.51/5|
|**Multi-Hop**   |      Medium|     83|      92.8%|  0.583|         4.80/5|
|**Multi-Hop**   |        Hard|     80|      87.5%|  0.443|         4.69/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  700.3s|
|**Questions/Second**   |    0.65|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.11%|
|**TPM Utilization**    |   0.11%|

## Index & Orchestrator

|Field             |                                     Value|
|------------------|-----------------------------------------:|
|**Index/Job ID**  |                     bfai__eval66b_g1536tt|
|**Mode**          |                                     cloud|
|**Endpoint**      |  https://bfai-api-ppfq5ahfsq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-23 08:52:10*
*Checkpoint: C032*
*Judge Model: gemini-3-flash-preview*