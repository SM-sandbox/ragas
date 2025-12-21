# Checkpoint Report: E006

**Generated:** 2025-12-21 14:46:20
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 0.0% |
| **Partial Rate** | 100.0% |
| **Fail Rate** | 0.0% |
| **Acceptable Rate** | 100.0% |
| **Overall Score** | 3.00/5 |
| **Questions** | 3 |

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
|**Overall Score**  |  3.00/5|      3.00/5|     0.00/5|
|**Pass Rate**      |    0.0%|        0.0%|       0.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  3.00/5|  0.00/5|  0.00/5|
|**Pass Rate**      |    0.0%|    0.0%|    0.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  3.00/5|  3.00/5|  0.00/5|  0.00/5|
|**Completeness**  |  3.00/5|  3.00/5|  0.00/5|  0.00/5|
|**Faithfulness**  |  3.00/5|  3.00/5|  0.00/5|  0.00/5|
|**Relevance**     |  3.00/5|  3.00/5|  0.00/5|  0.00/5|
|**Clarity**       |  3.00/5|  3.00/5|  0.00/5|  0.00/5|
|**Overall**       |  3.00/5|  3.00/5|  0.00/5|  0.00/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  3.00/5|      3.00/5|     0.00/5|
|**Completeness**  |  3.00/5|      3.00/5|     0.00/5|
|**Faithfulness**  |  3.00/5|      3.00/5|     0.00/5|
|**Relevance**     |  3.00/5|      3.00/5|     0.00/5|
|**Clarity**       |  3.00/5|      3.00/5|     0.00/5|
|**Overall**       |  3.00/5|      3.00/5|     0.00/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |    48.38s|       89.1%|
|**Total**       |    54.30s|        100%|

**Min Latency:** 48.54s  |  **Max Latency:** 59.01s

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
|**Single-Hop**  |      3|     0|        3|     0|       0.0%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |      3|     0|        3|     0|       0.0%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|      3|       0.0%|  1.000|         3.00/5|

## Execution & Throttling

|Metric                 |  Value|
|-----------------------|------:|
|**Run Duration**       |  74.3s|
|**Questions/Second**   |   0.04|
|**Max Workers**        |    100|
|**Effective Workers**  |     50|
|**Total Requests**     |      3|
|**Total Throttles**    |      0|
|**RPM Utilization**    |  0.01%|
|**TPM Utilization**    |  0.01%|

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

*Report generated: 2025-12-21 14:46:20*
*Checkpoint: E006*
*Judge Model: gemini-3-flash-preview*