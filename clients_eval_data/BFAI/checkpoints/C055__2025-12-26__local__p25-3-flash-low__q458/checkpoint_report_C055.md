# Checkpoint Report: C055

**Generated:** 2025-12-26 09:56:01
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 45.4% |
| **Partial Rate** | 18.6% |
| **Fail Rate** | 36.0% |
| **Acceptable Rate** | 64.0% |
| **Overall Score** | 3.39/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 45.4% | -47.8% | ❌ |
| **Acceptable Rate** | 97.6% | 64.0% | -33.6% | ❌ |
| **Fail Rate** | 2.4% | 36.0% | +33.6% | ❌ |
| **MRR** | 0.740 | 0.000 | -0.740 | ⚠️ |
| **Recall@100** | 99.1% | 0.0% | -99.1% | ⚠️ |
| **Overall Score** | 4.81/5 | 3.39/5 | -1.41 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 146.9s | +76.1s | 0.48x ❌ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8566 | $+0.0031 |
| **Per Question** | $0.001864 | $0.001870 | $+0.000007 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 44.6% | -51.4% |
| **Multi-Hop** | 90.7% | 46.2% | -44.5% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 43.5% | -45.3% |
| **Medium** | 97.5% | 44.1% | -53.4% |
| **Hard** | 93.4% | 49.3% | -44.1% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 47.8%, acceptable rate dropped 33.6%, latency increased 76.1s

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
|**Recall@100**  |   0.0%|        0.0%|       0.0%|
|**MRR**         |  0.000|       0.000|      0.000|

### By Difficulty

|Metric          |   Easy|  Medium|   Hard|
|----------------|------:|-------:|------:|
|**Recall@100**  |   0.0%|    0.0%|   0.0%|
|**MRR**         |  0.000|   0.000|  0.000|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  3.39/5|      3.31/5|     3.48/5|
|**Pass Rate**      |   45.4%|       44.6%|      46.2%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  3.31/5|  3.37/5|  3.53/5|
|**Pass Rate**      |   43.5%|   44.1%|   49.3%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  3.12/5|  3.04/5|  3.04/5|  3.31/5|
|**Completeness**  |  3.64/5|  3.43/5|  3.60/5|  3.94/5|
|**Faithfulness**  |  3.43/5|  3.24/5|  3.42/5|  3.67/5|
|**Relevance**     |  4.17/5|  3.96/5|  4.25/5|  4.32/5|
|**Clarity**       |  4.81/5|  4.73/5|  4.90/5|  4.81/5|
|**Overall**       |  3.39/5|  3.31/5|  3.37/5|  3.53/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  3.12/5|      3.02/5|     3.21/5|
|**Completeness**  |  3.64/5|      3.55/5|     3.72/5|
|**Faithfulness**  |  3.43/5|      3.19/5|     3.65/5|
|**Relevance**     |  4.17/5|      4.19/5|     4.15/5|
|**Clarity**       |  4.81/5|      4.74/5|     4.88/5|
|**Overall**       |  3.39/5|      3.31/5|     3.48/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.82s|        0.6%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |   143.18s|       97.5%|
|**Judge**       |     2.86s|        1.9%|
|**Total**       |   146.87s|        100%|

**Min Latency:** 10.78s  |  **Max Latency:** 291.54s

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
|**Single-Hop**  |    222|    99|       27|    96|      44.6%|
|**Multi-Hop**   |    236|   109|       58|    69|      46.2%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|    70|       25|    66|      43.5%|
|**Medium**  |    161|    71|       33|    57|      44.1%|
|**Hard**    |    136|    67|       27|    42|      49.3%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      39.8%|  0.000|         3.07/5|
|**Single-Hop**  |      Medium|     78|      42.3%|  0.000|         3.28/5|
|**Single-Hop**  |        Hard|     56|      55.4%|  0.000|         3.72/5|
|**Multi-Hop**   |        Easy|     73|      47.9%|  0.000|         3.60/5|
|**Multi-Hop**   |      Medium|     83|      45.8%|  0.000|         3.44/5|
|**Multi-Hop**   |        Hard|     80|      45.0%|  0.000|         3.40/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  783.0s|
|**Questions/Second**   |    0.58|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.10%|
|**TPM Utilization**    |   0.10%|

## Index & Orchestrator

|Field             |                     Value|
|------------------|-------------------------:|
|**Index/Job ID**  |  bfai__eval66a_g1_1536_tt|
|**Mode**          |                     local|
|**Endpoint**      |                       N/A|

---

*Report generated: 2025-12-26 09:56:01*
*Checkpoint: C055*
*Judge Model: gemini-3-flash-preview*