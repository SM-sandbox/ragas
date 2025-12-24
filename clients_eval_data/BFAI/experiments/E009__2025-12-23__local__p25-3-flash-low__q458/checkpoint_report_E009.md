# Checkpoint Report: E009

**Generated:** 2025-12-23 16:44:31
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
|**MRR**         |  0.737|       1.000|      0.490|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.752|   0.780|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.87/5|     4.78/5|
|**Pass Rate**      |   92.8%|       95.0%|      90.7%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.77/5|  4.88/5|  4.82/5|
|**Pass Rate**      |   90.1%|   95.7%|   92.6%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.76/5|  4.71/5|  4.83/5|  4.74/5|
|**Completeness**  |  4.87/5|  4.84/5|  4.93/5|  4.84/5|
|**Faithfulness**  |  4.94/5|  4.91/5|  4.98/5|  4.91/5|
|**Relevance**     |  4.97/5|  4.95/5|  4.99/5|  4.96/5|
|**Clarity**       |  4.98/5|  4.98/5|  4.99/5|  4.98/5|
|**Overall**       |  4.82/5|  4.77/5|  4.88/5|  4.82/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.76/5|      4.82/5|     4.71/5|
|**Completeness**  |  4.87/5|      4.93/5|     4.82/5|
|**Faithfulness**  |  4.94/5|      4.95/5|     4.92/5|
|**Relevance**     |  4.97/5|      4.97/5|     4.96/5|
|**Clarity**       |  4.98/5|      5.00/5|     4.97/5|
|**Overall**       |  4.82/5|      4.87/5|     4.78/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.61s|        0.8%|
|**Reranking**   |    15.26s|       19.2%|
|**Generation**  |    59.45s|       74.9%|
|**Judge**       |     4.03s|        5.1%|
|**Total**       |    79.36s|        100%|

**Min Latency:** 40.67s  |  **Max Latency:** 651.72s

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
|**Easy**    |    161|   145|       11|     5|      90.1%|
|**Medium**  |    161|   154|        7|     0|      95.7%|
|**Hard**    |    136|   126|        8|     2|      92.6%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      94.3%|  1.000|         4.82/5|
|**Single-Hop**  |      Medium|     78|      96.2%|  1.000|         4.90/5|
|**Single-Hop**  |        Hard|     56|      94.6%|  1.000|         4.89/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.452|         4.71/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.574|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      91.2%|  0.438|         4.76/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  761.7s|
|**Questions/Second**   |    0.60|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.01%|
|**TPM Utilization**    |   0.01%|

## Index & Orchestrator

|Field             |                     Value|
|------------------|-------------------------:|
|**Index/Job ID**  |  bfai__eval66a_g1_1536_tt|
|**Mode**          |                     local|
|**Endpoint**      |                       N/A|

---

*Report generated: 2025-12-23 16:44:31*
*Checkpoint: E009*
*Judge Model: gemini-3-flash-preview*