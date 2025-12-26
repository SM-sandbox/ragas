# Checkpoint Report: C038

**Generated:** 2025-12-25 16:14:03
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 53.9% |
| **Partial Rate** | 16.2% |
| **Fail Rate** | 29.9% |
| **Acceptable Rate** | 70.1% |
| **Overall Score** | 3.75/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 53.9% | -39.3% | ❌ |
| **Acceptable Rate** | 97.6% | 70.1% | -27.5% | ❌ |
| **Fail Rate** | 2.4% | 29.9% | +27.5% | ❌ |
| **MRR** | 0.740 | 0.737 | -0.002 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 3.75/5 | -1.06 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 55.4s | -15.3s | 1.28x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8062 | $-0.0474 |
| **Per Question** | $0.001864 | $0.001760 | $-0.000103 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 62.6% | -33.3% |
| **Multi-Hop** | 90.7% | 45.8% | -44.9% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 53.4% | -35.4% |
| **Medium** | 97.5% | 55.3% | -42.2% |
| **Hard** | 93.4% | 52.9% | -40.4% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 39.3%, acceptable rate dropped 27.5%

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-app-758831001226.us-east1.run.app |
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
|**MRR**         |  0.737|       1.000|      0.491|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.777|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  3.75/5|      3.94/5|     3.57/5|
|**Pass Rate**      |   53.9%|       62.6%|      45.8%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  3.75/5|  3.85/5|  3.62/5|
|**Pass Rate**      |   53.4%|   55.3%|   52.9%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  3.41/5|  3.32/5|  3.49/5|  3.43/5|
|**Completeness**  |  3.51/5|  3.47/5|  3.57/5|  3.49/5|
|**Faithfulness**  |  4.76/5|  4.70/5|  4.83/5|  4.73/5|
|**Relevance**     |  4.12/5|  4.09/5|  4.19/5|  4.09/5|
|**Clarity**       |  4.93/5|  4.93/5|  4.96/5|  4.90/5|
|**Overall**       |  3.75/5|  3.75/5|  3.85/5|  3.62/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  3.41/5|      3.56/5|     3.28/5|
|**Completeness**  |  3.51/5|      3.76/5|     3.28/5|
|**Faithfulness**  |  4.76/5|      4.72/5|     4.79/5|
|**Relevance**     |  4.12/5|      4.16/5|     4.09/5|
|**Clarity**       |  4.93/5|      4.94/5|     4.92/5|
|**Overall**       |  3.75/5|      3.94/5|     3.57/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.78s|        3.2%|
|**Total**       |    55.45s|        100%|

**Min Latency:** 9.34s  |  **Max Latency:** 89.20s

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
|**Single-Hop**  |    222|   139|       11|    72|      62.6%|
|**Multi-Hop**   |    236|   108|       63|    65|      45.8%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|    86|       21|    54|      53.4%|
|**Medium**  |    161|    89|       30|    42|      55.3%|
|**Hard**    |    136|    72|       23|    41|      52.9%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      60.2%|  1.000|         3.92/5|
|**Single-Hop**  |      Medium|     78|      66.7%|  1.000|         4.06/5|
|**Single-Hop**  |        Hard|     56|      60.7%|  1.000|         3.78/5|
|**Multi-Hop**   |        Easy|     73|      45.2%|  0.459|         3.54/5|
|**Multi-Hop**   |      Medium|     83|      44.6%|  0.568|         3.66/5|
|**Multi-Hop**   |        Hard|     80|      47.5%|  0.439|         3.51/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  301.1s|
|**Questions/Second**   |    1.52|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.44%|
|**TPM Utilization**    |   0.44%|

## Index & Orchestrator

|Field             |                                           Value|
|------------------|-----------------------------------------------:|
|**Index/Job ID**  |                        bfai__eval66a_g1_1536_tt|
|**Mode**          |                                           cloud|
|**Endpoint**      |  https://bfai-app-758831001226.us-east1.run.app|
|**Service**       |                                        bfai-api|
|**Project ID**    |                                       bfai-prod|
|**Environment**   |                                      production|
|**Region**        |                                        us-east1|

---

*Report generated: 2025-12-25 16:14:03*
*Checkpoint: C038*
*Judge Model: gemini-3-flash-preview*