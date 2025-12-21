# Checkpoint Report: C019

**Generated:** 2025-12-21 15:28:23
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 0.0% |
| **Partial Rate** | 100.0% |
| **Fail Rate** | 0.0% |
| **Acceptable Rate** | 100.0% |
| **Overall Score** | 3.00/5 |
| **Questions** | 10 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 0.0% | -92.8% | ❌ |
| **Acceptable Rate** | 98.5% | 100.0% | +1.5% | ✅ |
| **Fail Rate** | 1.5% | 0.0% | -1.5% | ✅ |
| **MRR** | 0.737 | 1.000 | +0.263 | ✅ |
| **Recall@100** | 99.1% | 100.0% | +0.9% | ✅ |
| **Overall Score** | 4.83/5 | 3.00/5 | -1.83 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 53.5s | +10.8s | 0.80x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.0172 | $-0.8262 |
| **Per Question** | $0.001841 | $0.001717 | $-0.000125 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 0.0% | -95.5% |
| **Multi-Hop** | 90.3% | 0.0% | -90.3% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 0.0% | -90.7% |
| **Medium** | 95.0% | 0.0% | -95.0% |
| **Hard** | 92.6% | 0.0% | -92.6% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 92.8%, latency increased 10.8s

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
|**Judge**       |    48.20s|       90.1%|
|**Total**       |    53.52s|        100%|

**Min Latency:** 38.10s  |  **Max Latency:** 66.81s

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
|**Single-Hop**  |     10|     0|       10|     0|       0.0%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |     10|     0|       10|     0|       0.0%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     10|       0.0%|  1.000|         3.00/5|

## Execution & Throttling

|Metric                 |  Value|
|-----------------------|------:|
|**Run Duration**       |  82.4s|
|**Questions/Second**   |   0.12|
|**Max Workers**        |    100|
|**Effective Workers**  |     50|
|**Total Requests**     |     10|
|**Total Throttles**    |      0|
|**RPM Utilization**    |  0.02%|
|**TPM Utilization**    |  0.02%|

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

*Report generated: 2025-12-21 15:28:23*
*Checkpoint: C019*
*Judge Model: gemini-3-flash-preview*