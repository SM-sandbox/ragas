# Checkpoint Report: C037

**Generated:** 2025-12-25 16:09:01
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 53.1% |
| **Partial Rate** | 15.5% |
| **Fail Rate** | 31.4% |
| **Acceptable Rate** | 68.6% |
| **Overall Score** | 3.73/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 53.1% | -40.2% | ❌ |
| **Acceptable Rate** | 97.6% | 68.6% | -29.0% | ❌ |
| **Fail Rate** | 2.4% | 31.4% | +29.0% | ❌ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 3.73/5 | -1.08 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 56.2s | -14.6s | 1.26x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8106 | $-0.0430 |
| **Per Question** | $0.001864 | $0.001770 | $-0.000094 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 60.4% | -35.6% |
| **Multi-Hop** | 90.7% | 46.2% | -44.5% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 53.4% | -35.4% |
| **Medium** | 97.5% | 54.7% | -42.9% |
| **Hard** | 93.4% | 50.7% | -42.6% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 40.2%, acceptable rate dropped 29.0%

## Configuration

### Run Settings

| Parameter | Value |
|-----------|-------|
| **Mode** | cloud |
| **Endpoint** | https://bfai-app-904268163020.us-east1.run.app |
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
|**MRR**         |  0.739|       1.000|      0.493|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.752|   0.784|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  3.73/5|      3.94/5|     3.53/5|
|**Pass Rate**      |   53.1%|       60.4%|      46.2%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  3.73/5|  3.82/5|  3.61/5|
|**Pass Rate**      |   53.4%|   54.7%|   50.7%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  3.39/5|  3.30/5|  3.50/5|  3.38/5|
|**Completeness**  |  3.50/5|  3.47/5|  3.58/5|  3.43/5|
|**Faithfulness**  |  4.76/5|  4.70/5|  4.87/5|  4.72/5|
|**Relevance**     |  4.10/5|  4.15/5|  4.06/5|  4.08/5|
|**Clarity**       |  4.93/5|  4.93/5|  4.98/5|  4.86/5|
|**Overall**       |  3.73/5|  3.73/5|  3.82/5|  3.61/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  3.39/5|      3.52/5|     3.28/5|
|**Completeness**  |  3.50/5|      3.71/5|     3.30/5|
|**Faithfulness**  |  4.76/5|      4.75/5|     4.78/5|
|**Relevance**     |  4.10/5|      4.19/5|     4.01/5|
|**Clarity**       |  4.93/5|      4.95/5|     4.90/5|
|**Overall**       |  3.73/5|      3.94/5|     3.53/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.75s|        3.1%|
|**Total**       |    56.19s|        100%|

**Min Latency:** 9.97s  |  **Max Latency:** 99.19s

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
|**Single-Hop**  |    222|   134|       15|    73|      60.4%|
|**Multi-Hop**   |    236|   109|       56|    71|      46.2%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|    86|       17|    58|      53.4%|
|**Medium**  |    161|    88|       31|    42|      54.7%|
|**Hard**    |    136|    69|       23|    44|      50.7%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      58.0%|  1.000|         3.90/5|
|**Single-Hop**  |      Medium|     78|      66.7%|  1.000|         4.06/5|
|**Single-Hop**  |        Hard|     56|      55.4%|  1.000|         3.82/5|
|**Multi-Hop**   |        Easy|     73|      47.9%|  0.452|         3.52/5|
|**Multi-Hop**   |      Medium|     83|      43.4%|  0.582|         3.60/5|
|**Multi-Hop**   |        Hard|     80|      47.5%|  0.438|         3.46/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  313.3s|
|**Questions/Second**   |    1.46|
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
|**Endpoint**      |  https://bfai-app-904268163020.us-east1.run.app|
|**Service**       |                                        bfai-api|
|**Project ID**    |                                       bfai-prod|
|**Environment**   |                                      production|
|**Region**        |                                        us-east1|

---

*Report generated: 2025-12-25 16:09:01*
*Checkpoint: C037*
*Judge Model: gemini-3-flash-preview*