# Checkpoint Report: C020

**Generated:** 2025-12-22 17:02:16
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.1% |
| **Partial Rate** | 6.8% |
| **Fail Rate** | 1.1% |
| **Acceptable Rate** | 98.9% |
| **Overall Score** | 4.83/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 92.1% | -0.7% | ✅ |
| **Acceptable Rate** | 98.5% | 98.9% | +0.4% | ✅ |
| **Fail Rate** | 1.5% | 1.1% | -0.4% | ✅ |
| **MRR** | 0.737 | 0.739 | +0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.83/5 | -0.00 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 78.3s | +35.5s | 0.55x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.8608 | $+0.0175 |
| **Per Question** | $0.001841 | $0.001880 | $+0.000038 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 94.1% | -1.4% |
| **Multi-Hop** | 90.3% | 90.3% | +0.0% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 90.1% | -0.6% |
| **Medium** | 95.0% | 95.7% | +0.6% |
| **Hard** | 92.6% | 90.4% | -2.2% |

### Key Finding

⚠️ **Regressions detected:** latency increased 35.5s

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
|**MRR**         |   0.751|   0.783|  0.671|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.83/5|      4.86/5|     4.79/5|
|**Pass Rate**      |   92.1%|       94.1%|      90.3%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.79/5|  4.89/5|  4.80/5|
|**Pass Rate**      |   90.1%|   95.7%|   90.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.77/5|  4.74/5|  4.85/5|  4.72/5|
|**Completeness**  |  4.87/5|  4.83/5|  4.93/5|  4.84/5|
|**Faithfulness**  |  4.93/5|  4.92/5|  4.94/5|  4.92/5|
|**Relevance**     |  4.97/5|  4.94/5|  4.99/5|  4.97/5|
|**Clarity**       |  4.98/5|  4.99/5|  4.99/5|  4.97/5|
|**Overall**       |  4.83/5|  4.79/5|  4.89/5|  4.80/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.77/5|      4.83/5|     4.72/5|
|**Completeness**  |  4.87/5|      4.91/5|     4.83/5|
|**Faithfulness**  |  4.93/5|      4.92/5|     4.94/5|
|**Relevance**     |  4.97/5|      4.97/5|     4.96/5|
|**Clarity**       |  4.98/5|      5.00/5|     4.97/5|
|**Overall**       |  4.83/5|      4.86/5|     4.79/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.73s|        0.9%|
|**Reranking**   |     4.77s|        6.1%|
|**Generation**  |    71.72s|       91.6%|
|**Judge**       |     1.05s|        1.3%|
|**Total**       |    78.27s|        100%|

**Min Latency:** 14.85s  |  **Max Latency:** 151.17s

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
|**Single-Hop**  |    222|   209|       10|     3|      94.1%|
|**Multi-Hop**   |    236|   213|       21|     2|      90.3%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   145|       13|     3|      90.1%|
|**Medium**  |    161|   154|        7|     0|      95.7%|
|**Hard**    |    136|   123|       11|     2|      90.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      94.3%|  1.000|         4.85/5|
|**Single-Hop**  |      Medium|     78|      96.2%|  1.000|         4.89/5|
|**Single-Hop**  |        Hard|     56|      91.1%|  1.000|         4.83/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.452|         4.70/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.580|         4.89/5|
|**Multi-Hop**   |        Hard|     80|      90.0%|  0.440|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  417.6s|
|**Questions/Second**   |    1.10|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.26%|
|**TPM Utilization**    |   0.26%|

## Index & Orchestrator

|Field             |                     Value|
|------------------|-------------------------:|
|**Index/Job ID**  |  bfai__eval66a_g1_1536_tt|
|**Mode**          |                     local|
|**Endpoint**      |                       N/A|

---

*Report generated: 2025-12-22 17:02:16*
*Checkpoint: C020*
*Judge Model: gemini-3-flash-preview*