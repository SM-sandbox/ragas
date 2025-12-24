# Checkpoint Report: E008

**Generated:** 2025-12-23 16:28:54
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 93.3% |
| **Partial Rate** | 6.7% |
| **Fail Rate** | 0.0% |
| **Acceptable Rate** | 100.0% |
| **Overall Score** | 4.89/5 |
| **Questions** | 30 |

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
|**Overall Score**  |  4.89/5|      4.89/5|     0.00/5|
|**Pass Rate**      |   93.3%|       93.3%|       0.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.89/5|  0.00/5|  0.00/5|
|**Pass Rate**      |   93.3%|    0.0%|    0.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.87/5|  4.87/5|  0.00/5|  0.00/5|
|**Completeness**  |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Faithfulness**  |  4.87/5|  4.87/5|  0.00/5|  0.00/5|
|**Relevance**     |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Clarity**       |  5.00/5|  5.00/5|  0.00/5|  0.00/5|
|**Overall**       |  4.89/5|  4.89/5|  0.00/5|  0.00/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.87/5|      4.87/5|     0.00/5|
|**Completeness**  |  5.00/5|      5.00/5|     0.00/5|
|**Faithfulness**  |  4.87/5|      4.87/5|     0.00/5|
|**Relevance**     |  5.00/5|      5.00/5|     0.00/5|
|**Clarity**       |  5.00/5|      5.00/5|     0.00/5|
|**Overall**       |  4.89/5|      4.89/5|     0.00/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     1.68s|        9.7%|
|**Reranking**   |     4.56s|       26.2%|
|**Generation**  |     9.38s|       54.0%|
|**Judge**       |     1.76s|       10.1%|
|**Total**       |    17.37s|        100%|

**Min Latency:** 10.44s  |  **Max Latency:** 25.27s

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
|**Single-Hop**  |     30|    28|        2|     0|      93.3%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |     30|    28|        2|     0|      93.3%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     30|      93.3%|  1.000|         4.89/5|

## Execution & Throttling

|Metric                 |  Value|
|-----------------------|------:|
|**Run Duration**       |  29.0s|
|**Questions/Second**   |   1.04|
|**Max Workers**        |    100|
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

*Report generated: 2025-12-23 16:28:54*
*Checkpoint: E008*
*Judge Model: gemini-3-flash-preview*