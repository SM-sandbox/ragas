# Checkpoint Report: R010

**Generated:** 2025-12-21 12:35:27
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.8% |
| **Partial Rate** | 6.1% |
| **Fail Rate** | 1.1% |
| **Acceptable Rate** | 98.9% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

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
|**MRR**         |  0.740|       1.000|      0.495|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.784|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.85/5|     4.79/5|
|**Pass Rate**      |   92.8%|       93.7%|      91.9%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.79/5|  4.84/5|  4.84/5|
|**Pass Rate**      |   91.9%|   93.2%|   93.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.78/5|  4.74/5|  4.83/5|  4.77/5|
|**Completeness**  |  4.86/5|  4.83/5|  4.90/5|  4.85/5|
|**Faithfulness**  |  4.91/5|  4.93/5|  4.89/5|  4.90/5|
|**Relevance**     |  4.96/5|  4.93/5|  4.98/5|  4.97/5|
|**Clarity**       |  4.99/5|  5.00/5|  4.99/5|  4.98/5|
|**Overall**       |  4.82/5|  4.79/5|  4.84/5|  4.84/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.78/5|      4.82/5|     4.74/5|
|**Completeness**  |  4.86/5|      4.90/5|     4.83/5|
|**Faithfulness**  |  4.91/5|      4.89/5|     4.93/5|
|**Relevance**     |  4.96/5|      4.94/5|     4.97/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.98/5|
|**Overall**       |  4.82/5|      4.85/5|     4.79/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.27s|        1.7%|
|**Total**       |    74.26s|        100%|

**Min Latency:** 7.77s  |  **Max Latency:** 133.50s

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
|**Single-Hop**  |    222|   208|       11|     3|      93.7%|
|**Multi-Hop**   |    236|   217|       17|     2|      91.9%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   148|        9|     4|      91.9%|
|**Medium**  |    161|   150|       11|     0|      93.2%|
|**Hard**    |    136|   127|        8|     1|      93.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      94.3%|  1.000|         4.83/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.81/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.92/5|
|**Multi-Hop**   |        Easy|     73|      89.0%|  0.459|         4.74/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.582|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      91.2%|  0.437|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  545.3s|
|**Questions/Second**   |    0.84|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.04%|
|**TPM Utilization**    |   0.04%|

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

*Report generated: 2025-12-21 12:35:27*
*Checkpoint: R010*
*Judge Model: gemini-3-flash-preview*