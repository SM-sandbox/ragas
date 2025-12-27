# Checkpoint Report: E010

**Generated:** 2025-12-27 13:46:29
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 100.0% |
| **Partial Rate** | 0.0% |
| **Fail Rate** | 0.0% |
| **Acceptable Rate** | 100.0% |
| **Overall Score** | 4.96/5 |
| **Questions** | 30 |

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | local |
| **Endpoint** | N/A |
| **Precision@K** | 25 |
| **Recall@K** | 100 |
| **Max Workers** | 5 |
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
|**Recall@100**  |  100.0%|      100.0%|     100.0%|
|**MRR**         |   0.759|       1.000|      0.519|

### By Difficulty

|Metric          |    Easy|  Medium|    Hard|
|----------------|-------:|-------:|-------:|
|**Recall@100**  |  100.0%|  100.0%|  100.0%|
|**MRR**         |   0.758|   0.808|   0.712|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.96/5|      5.00/5|     4.92/5|
|**Pass Rate**      |  100.0%|      100.0%|     100.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.98/5|  5.00/5|  4.90/5|
|**Pass Rate**      |  100.0%|  100.0%|  100.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.90/5|  4.90/5|  5.00/5|  4.80/5|
|**Completeness**  |  5.00/5|  5.00/5|  5.00/5|  5.00/5|
|**Faithfulness**  |  5.00/5|  5.00/5|  5.00/5|  5.00/5|
|**Relevance**     |  5.00/5|  5.00/5|  5.00/5|  5.00/5|
|**Clarity**       |  5.00/5|  5.00/5|  5.00/5|  5.00/5|
|**Overall**       |  4.96/5|  4.98/5|  5.00/5|  4.90/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.90/5|      5.00/5|     4.80/5|
|**Completeness**  |  5.00/5|      5.00/5|     5.00/5|
|**Faithfulness**  |  5.00/5|      5.00/5|     5.00/5|
|**Relevance**     |  5.00/5|      5.00/5|     5.00/5|
|**Clarity**       |  5.00/5|      5.00/5|     5.00/5|
|**Overall**       |  4.96/5|      5.00/5|     4.92/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.79s|        7.3%|
|**Reranking**   |     2.20s|       20.3%|
|**Generation**  |     6.37s|       58.9%|
|**Judge**       |     1.45s|       13.4%|
|**Total**       |    10.82s|        100%|

**Min Latency:** 3.63s  |  **Max Latency:** 21.34s

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
|**Single-Hop**  |     15|    15|        0|     0|     100.0%|
|**Multi-Hop**   |     15|    15|        0|     0|     100.0%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |     10|    10|        0|     0|     100.0%|
|**Medium**  |     10|    10|        0|     0|     100.0%|
|**Hard**    |     10|    10|        0|     0|     100.0%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|      5|     100.0%|  1.000|         5.00/5|
|**Single-Hop**  |      Medium|      5|     100.0%|  1.000|         5.00/5|
|**Single-Hop**  |        Hard|      5|     100.0%|  1.000|         5.00/5|
|**Multi-Hop**   |        Easy|      5|     100.0%|  0.517|         4.96/5|
|**Multi-Hop**   |      Medium|      5|     100.0%|  0.617|         5.00/5|
|**Multi-Hop**   |        Hard|      5|     100.0%|  0.423|         4.80/5|

## Execution & Throttling

|Metric                 |  Value|
|-----------------------|------:|
|**Run Duration**       |  70.5s|
|**Questions/Second**   |   0.43|
|**Max Workers**        |      5|
|**Effective Workers**  |     50|
|**Total Requests**     |     30|
|**Total Throttles**    |      0|
|**RPM Utilization**    |  0.15%|
|**TPM Utilization**    |  0.15%|

## Index & Orchestrator

|Field             |                     Value|
|------------------|-------------------------:|
|**Index/Job ID**  |  bfai__eval66a_g1_1536_tt|
|**Mode**          |                     local|
|**Endpoint**      |                       N/A|

---

*Report generated: 2025-12-27 13:46:29*
*Checkpoint: E010*
*Judge Model: gemini-3-flash-preview*