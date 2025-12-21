# Checkpoint Report: E005

**Generated:** 2025-12-21 13:18:22
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.6% |
| **Partial Rate** | 6.1% |
| **Fail Rate** | 1.3% |
| **Acceptable Rate** | 98.7% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-api-ppfq5ahfsq-ue.a.run.app |
| **Precision@K** | 25 |
| **Recall@K** | 200 |
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
|**MRR**         |   0.755|   0.783|  0.671|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.85/5|     4.79/5|
|**Pass Rate**      |   92.6%|       93.7%|      91.5%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.78/5|  4.86/5|  4.83/5|
|**Pass Rate**      |   91.3%|   93.8%|   92.6%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.78/5|  4.72/5|  4.84/5|  4.76/5|
|**Completeness**  |  4.86/5|  4.83/5|  4.91/5|  4.83/5|
|**Faithfulness**  |  4.91/5|  4.91/5|  4.91/5|  4.91/5|
|**Relevance**     |  4.96/5|  4.94/5|  4.98/5|  4.97/5|
|**Clarity**       |  4.99/5|  5.00/5|  4.99/5|  4.98/5|
|**Overall**       |  4.82/5|  4.78/5|  4.86/5|  4.83/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.78/5|      4.83/5|     4.72/5|
|**Completeness**  |  4.86/5|      4.90/5|     4.82/5|
|**Faithfulness**  |  4.91/5|      4.91/5|     4.92/5|
|**Relevance**     |  4.96/5|      4.94/5|     4.98/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.98/5|
|**Overall**       |  4.82/5|      4.85/5|     4.79/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.23s|        1.7%|
|**Total**       |    74.20s|        100%|

**Min Latency:** 7.50s  |  **Max Latency:** 134.27s

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
|**Multi-Hop**   |    236|   216|       17|     3|      91.5%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   147|        9|     5|      91.3%|
|**Medium**  |    161|   151|       10|     0|      93.8%|
|**Hard**    |    136|   126|        9|     1|      92.6%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      94.3%|  1.000|         4.83/5|
|**Single-Hop**  |      Medium|     78|      92.3%|  1.000|         4.85/5|
|**Single-Hop**  |        Hard|     56|      94.6%|  1.000|         4.90/5|
|**Multi-Hop**   |        Easy|     73|      87.7%|  0.459|         4.72/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.580|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      91.2%|  0.440|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  508.0s|
|**Questions/Second**   |    0.90|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.10%|
|**TPM Utilization**    |   0.10%|

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

*Report generated: 2025-12-21 13:18:22*
*Checkpoint: E005*
*Judge Model: gemini-3-flash-preview*