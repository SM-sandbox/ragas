# Checkpoint Report: C071

**Generated:** 2025-12-26 16:04:12
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 87.6% |
| **Partial Rate** | 9.2% |
| **Fail Rate** | 3.3% |
| **Acceptable Rate** | 96.7% |
| **Overall Score** | 4.71/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.1% | 87.6% | -4.6% | ⚠️ |
| **Acceptable Rate** | 98.9% | 96.7% | -2.2% | ❌ |
| **Fail Rate** | 1.1% | 3.3% | +2.2% | ❌ |
| **MRR** | 0.740 | 0.740 | +0.000 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.82/5 | 4.71/5 | -0.11 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 57.7s | 73.9s | +16.1s | 0.78x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8664 | $0.9587 | $+0.0923 |
| **Per Question** | $0.001892 | $0.002093 | $+0.000201 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 92.8% | 88.7% | -4.1% |
| **Multi-Hop** | 91.5% | 86.4% | -5.1% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 89.4% | 83.2% | -6.2% |
| **Medium** | 93.2% | 90.1% | -3.1% |
| **Hard** | 94.1% | 89.7% | -4.4% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 4.6%, acceptable rate dropped 2.2%, latency increased 16.1s

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
|**MRR**         |  0.740|       1.000|      0.496|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.786|  0.668|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.71/5|      4.75/5|     4.68/5|
|**Pass Rate**      |   87.6%|       88.7%|      86.4%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.63/5|  4.77/5|  4.74/5|
|**Pass Rate**      |   83.2%|   90.1%|   89.7%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.68/5|  4.64/5|  4.73/5|  4.68/5|
|**Completeness**  |  4.78/5|  4.72/5|  4.83/5|  4.79/5|
|**Faithfulness**  |  4.76/5|  4.65/5|  4.76/5|  4.89/5|
|**Relevance**     |  4.94/5|  4.94/5|  4.94/5|  4.95/5|
|**Clarity**       |  4.98/5|  4.98/5|  4.99/5|  4.99/5|
|**Overall**       |  4.71/5|  4.63/5|  4.77/5|  4.74/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.68/5|      4.72/5|     4.64/5|
|**Completeness**  |  4.78/5|      4.81/5|     4.75/5|
|**Faithfulness**  |  4.76/5|      4.68/5|     4.83/5|
|**Relevance**     |  4.94/5|      4.92/5|     4.97/5|
|**Clarity**       |  4.98/5|      4.98/5|     4.99/5|
|**Overall**       |  4.71/5|      4.75/5|     4.68/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     2.16s|        2.9%|
|**Total**       |    73.85s|        100%|

**Min Latency:** 15.42s  |  **Max Latency:** 123.79s

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
|**Single-Hop**  |    222|   197|       15|    10|      88.7%|
|**Multi-Hop**   |    236|   204|       27|     5|      86.4%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   134|       20|     7|      83.2%|
|**Medium**  |    161|   145|       10|     6|      90.1%|
|**Hard**    |    136|   122|       12|     2|      89.7%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      86.4%|  1.000|         4.71/5|
|**Single-Hop**  |      Medium|     78|      89.7%|  1.000|         4.75/5|
|**Single-Hop**  |        Hard|     56|      91.1%|  1.000|         4.79/5|
|**Multi-Hop**   |        Easy|     73|      79.5%|  0.459|         4.54/5|
|**Multi-Hop**   |      Medium|     83|      90.4%|  0.586|         4.78/5|
|**Multi-Hop**   |        Hard|     80|      88.8%|  0.436|         4.71/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  433.5s|
|**Questions/Second**   |    1.06|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.30%|
|**TPM Utilization**    |   0.30%|

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

*Report generated: 2025-12-26 16:04:12*
*Checkpoint: C071*
*Judge Model: gemini-3-flash-preview*