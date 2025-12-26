# Checkpoint Report: C059

**Generated:** 2025-12-26 10:38:01
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 0.4% |
| **Partial Rate** | 0.0% |
| **Fail Rate** | 99.6% |
| **Acceptable Rate** | 0.4% |
| **Overall Score** | 1.19/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 0.4% | -92.8% | ❌ |
| **Acceptable Rate** | 97.6% | 0.4% | -97.2% | ❌ |
| **Fail Rate** | 2.4% | 99.6% | +97.2% | ❌ |
| **MRR** | 0.740 | 0.000 | -0.740 | ⚠️ |
| **Recall@100** | 99.1% | 0.0% | -99.1% | ⚠️ |
| **Overall Score** | 4.81/5 | 1.19/5 | -3.62 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 22.7s | -48.1s | 3.12x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.1261 | $-0.7275 |
| **Per Question** | $0.001864 | $0.000275 | $-0.001588 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 0.9% | -95.0% |
| **Multi-Hop** | 90.7% | 0.0% | -90.7% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 1.2% | -87.6% |
| **Medium** | 97.5% | 0.0% | -97.5% |
| **Hard** | 93.4% | 0.0% | -93.4% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 92.8%, acceptable rate dropped 97.2%

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-app-ppfq5ahfsq-ue.a.run.app |
| **Precision@K** | 25 |
| **Recall@K** | 100 |
| **Max Workers** | 100 |
| **Effective Workers** | 50 |

### Generator Model

| Parameter | Value |
|-----------|-------|
| **Model** | cloud |
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
|**Overall Score**  |  1.19/5|      1.27/5|     1.12/5|
|**Pass Rate**      |    0.4%|        0.9%|       0.0%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  1.34/5|  1.13/5|  1.09/5|
|**Pass Rate**      |    1.2%|    0.0%|    0.0%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  1.02/5|  1.05/5|  1.00/5|  1.00/5|
|**Completeness**  |  1.02/5|  1.05/5|  1.00/5|  1.00/5|
|**Faithfulness**  |  1.94/5|  1.72/5|  1.88/5|  2.26/5|
|**Relevance**     |  1.02/5|  1.05/5|  1.00/5|  1.00/5|
|**Clarity**       |  3.57/5|  3.94/5|  3.61/5|  3.09/5|
|**Overall**       |  1.19/5|  1.34/5|  1.13/5|  1.09/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  1.02/5|      1.04/5|     1.00/5|
|**Completeness**  |  1.02/5|      1.04/5|     1.00/5|
|**Faithfulness**  |  1.94/5|      1.81/5|     2.06/5|
|**Relevance**     |  1.02/5|      1.04/5|     1.00/5|
|**Clarity**       |  3.57/5|      4.00/5|     3.17/5|
|**Overall**       |  1.19/5|      1.27/5|     1.12/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     3.43s|       15.1%|
|**Total**       |    22.69s|        100%|

**Min Latency:** 4.11s  |  **Max Latency:** 44.37s

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
|**Single-Hop**  |    222|     2|        0|   220|       0.9%|
|**Multi-Hop**   |    236|     0|        0|   236|       0.0%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|     2|        0|   159|       1.2%|
|**Medium**  |    161|     0|        0|   161|       0.0%|
|**Hard**    |    136|     0|        0|   136|       0.0%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|       2.3%|  0.000|         1.46/5|
|**Single-Hop**  |      Medium|     78|       0.0%|  0.000|         1.16/5|
|**Single-Hop**  |        Hard|     56|       0.0%|  0.000|         1.11/5|
|**Multi-Hop**   |        Easy|     73|       0.0%|  0.000|         1.19/5|
|**Multi-Hop**   |      Medium|     83|       0.0%|  0.000|         1.10/5|
|**Multi-Hop**   |        Hard|     80|       0.0%|  0.000|         1.07/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  135.3s|
|**Questions/Second**   |    3.38|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   1.26%|
|**TPM Utilization**    |   1.26%|

## Index & Orchestrator

|Field             |                                     Value|
|------------------|-----------------------------------------:|
|**Index/Job ID**  |                  bfai__eval66a_g1_1536_tt|
|**Mode**          |                                     cloud|
|**Endpoint**      |  https://bfai-app-ppfq5ahfsq-ue.a.run.app|
|**Service**       |                                  bfai-api|
|**Project ID**    |                                 bfai-prod|
|**Environment**   |                                production|
|**Region**        |                                  us-east1|

---

*Report generated: 2025-12-26 10:38:01*
*Checkpoint: C059*
*Judge Model: gemini-3-flash-preview*