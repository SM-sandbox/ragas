# Checkpoint Report: R009

**Generated:** 2025-12-21 12:20:20
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 100.0% |
| **Partial Rate** | 0.0% |
| **Fail Rate** | 0.0% |
| **Acceptable Rate** | 100.0% |
| **Overall Score** | 5.00/5 |
| **Questions** | 6 |

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-api-ppfq5ahfsq-ue.a.run.app |
| **Precision@K** | 25 |
| **Recall@K** | 100 |
| **Max Workers** | 3 |
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

|Metric          |   Total|  Single-Hop|  Multi-Hop|
|----------------|-------:|-----------:|----------:|
|**Recall@100**  |  100.0%|      100.0%|       0.0%|
|**MRR**         |   1.000|       1.000|      0.000|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|    0.0%|   0.0%|
|**MRR**         |   1.000|   0.000|  0.000|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  5.00/5|      5.00/5|     0.00/5|
|**Pass Rate**      |  100.0%|      100.0%|       0.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  5.00/5|  0.00/5|  0.00/5|
|**Pass Rate**      |  100.0%|    0.0%|    0.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Completeness**  |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Faithfulness**  |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Relevance**     |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Clarity**       |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Overall**       |  5.00/5|  5.00/5|  0.00/5|  0.00/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  5.00/5|      5.00/5|     0.00/5|
|**Completeness**  |  5.00/5|      5.00/5|     0.00/5|
|**Faithfulness**  |  5.00/5|      5.00/5|     0.00/5|
|**Relevance**     |  5.00/5|      5.00/5|     0.00/5|
|**Clarity**       |  5.00/5|      5.00/5|     0.00/5|
|**Overall**       |  5.00/5|      5.00/5|     0.00/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.80s|       18.1%|
|**Total**       |     9.91s|        100%|

**Min Latency:** 3.15s  |  **Max Latency:** 19.42s

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
|**Single-Hop**  |      6|     6|        0|     0|     100.0%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |      6|     6|        0|     0|     100.0%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|      6|     100.0%|  1.000|         5.00/5|

## Execution & Throttling

|Metric                 |  Value|
|-----------------------|------:|
|**Run Duration**       |  52.3s|
|**Questions/Second**   |   0.11|
|**Max Workers**        |      3|
|**Effective Workers**  |     50|
|**Total Requests**     |      6|
|**Total Throttles**    |      0|
|**RPM Utilization**    |  0.03%|
|**TPM Utilization**    |  0.03%|

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

*Report generated: 2025-12-21 12:20:20*
*Checkpoint: R009*
*Judge Model: gemini-3-flash-preview*