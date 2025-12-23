# Checkpoint Report: C029

**Generated:** 2025-12-23 06:55:25
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.8% |
| **Partial Rate** | 5.7% |
| **Fail Rate** | 1.5% |
| **Acceptable Rate** | 98.5% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 92.8% | +0.0% | ✅ |
| **Acceptable Rate** | 98.5% | 98.5% | +0.0% | ✅ |
| **Fail Rate** | 1.5% | 1.5% | +0.0% | ✅ |
| **MRR** | 0.737 | 0.739 | +0.002 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.82/5 | -0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 77.1s | +34.3s | 0.55x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.8326 | $-0.0108 |
| **Per Question** | $0.001841 | $0.001818 | $-0.000023 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 95.0% | -0.5% |
| **Multi-Hop** | 90.3% | 90.7% | +0.4% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 90.1% | -0.6% |
| **Medium** | 95.0% | 95.0% | +0.0% |
| **Hard** | 92.6% | 93.4% | +0.7% |

### Key Finding

⚠️ **Regressions detected:** latency increased 34.3s

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | local |
| **Endpoint** | N/A |
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
|**MRR**         |   0.755|   0.781|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.87/5|     4.78/5|
|**Pass Rate**      |   92.8%|       95.0%|      90.7%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.78/5|  4.86/5|  4.82/5|
|**Pass Rate**      |   90.1%|   95.0%|   93.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.76/5|  4.73/5|  4.81/5|  4.74/5|
|**Completeness**  |  4.88/5|  4.86/5|  4.93/5|  4.84/5|
|**Faithfulness**  |  4.93/5|  4.91/5|  4.98/5|  4.91/5|
|**Relevance**     |  4.97/5|  4.96/5|  4.99/5|  4.96/5|
|**Clarity**       |  4.98/5|  4.98/5|  4.99/5|  4.98/5|
|**Overall**       |  4.82/5|  4.78/5|  4.86/5|  4.82/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.76/5|      4.82/5|     4.71/5|
|**Completeness**  |  4.88/5|      4.94/5|     4.82/5|
|**Faithfulness**  |  4.93/5|      4.95/5|     4.92/5|
|**Relevance**     |  4.97/5|      4.97/5|     4.97/5|
|**Clarity**       |  4.98/5|      5.00/5|     4.97/5|
|**Overall**       |  4.82/5|      4.87/5|     4.78/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.65s|        0.8%|
|**Reranking**   |    15.37s|       19.9%|
|**Generation**  |    59.50s|       77.2%|
|**Judge**       |     1.56s|        2.0%|
|**Total**       |    77.08s|        100%|

**Min Latency:** 37.97s  |  **Max Latency:** 130.98s

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
|**Single-Hop**  |    222|   211|        6|     5|      95.0%|
|**Multi-Hop**   |    236|   214|       20|     2|      90.7%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   145|       12|     4|      90.1%|
|**Medium**  |    161|   153|        7|     1|      95.0%|
|**Hard**    |    136|   127|        7|     2|      93.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      94.3%|  1.000|         4.84/5|
|**Single-Hop**  |      Medium|     78|      94.9%|  1.000|         4.87/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.90/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.459|         4.71/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.576|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      91.2%|  0.439|         4.77/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  425.7s|
|**Questions/Second**   |    1.08|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.18%|
|**TPM Utilization**    |   0.18%|

## Index & Orchestrator

|Field             |                     Value|
|------------------|-------------------------:|
|**Index/Job ID**  |  bfai__eval66a_g1_1536_tt|
|**Mode**          |                     local|
|**Endpoint**      |                       N/A|

---

*Report generated: 2025-12-23 06:55:25*
*Checkpoint: C029*
*Judge Model: gemini-3-flash-preview*