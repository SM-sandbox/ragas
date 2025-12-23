# Checkpoint Report: C024

**Generated:** 2025-12-22 18:12:23
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 50.7% |
| **Partial Rate** | 40.6% |
| **Fail Rate** | 8.1% |
| **Acceptable Rate** | 91.3% |
| **Overall Score** | 4.20/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 50.7% | -42.1% | ❌ |
| **Acceptable Rate** | 98.5% | 91.3% | -7.2% | ❌ |
| **Fail Rate** | 1.5% | 8.1% | +6.6% | ❌ |
| **MRR** | 0.737 | 0.738 | +0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.20/5 | -0.63 | ⚠️ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 105.2s | +62.5s | 0.41x ❌ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $1.4036 | $+0.5602 |
| **Per Question** | $0.001841 | $0.003065 | $+0.001223 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 64.9% | -30.6% |
| **Multi-Hop** | 90.3% | 37.3% | -53.0% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 50.3% | -40.4% |
| **Medium** | 95.0% | 53.4% | -41.6% |
| **Hard** | 92.6% | 47.8% | -44.9% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 42.1%, acceptable rate dropped 7.2%, latency increased 62.5s

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
| **Model** | gemini-2.5-flash |
| **Reasoning Effort** | low |
| **Temperature** | 0.0 |
| **Seed** | 42 |

### Judge Model

| Parameter | Value |
|-----------|-------|
| **Model** | gemini-2.5-flash |
| **Reasoning Effort** | low |
| **Seed** | 42 |

## Retrieval Metrics

### By Question Type

|Metric          |  Total|  Single-Hop|  Multi-Hop|
|----------------|------:|-----------:|----------:|
|**Recall@100**  |  99.1%|      100.0%|      98.3%|
|**MRR**         |  0.738|       1.000|      0.492|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.752|   0.783|  0.669|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.20/5|      4.39/5|     3.96/5|
|**Pass Rate**      |   50.7%|       64.9%|      37.3%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.10/5|  4.28/5|  4.13/5|
|**Pass Rate**      |   50.3%|   53.4%|   47.8%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.33/5|  4.29/5|  4.40/5|  4.21/5|
|**Completeness**  |  4.45/5|  4.43/5|  4.54/5|  4.26/5|
|**Faithfulness**  |  3.75/5|  3.52/5|  3.75/5|  3.95/5|
|**Relevance**     |  4.87/5|  4.79/5|  4.83/5|  4.90/5|
|**Clarity**       |  4.90/5|  4.88/5|  4.86/5|  4.86/5|
|**Overall**       |  4.20/5|  4.10/5|  4.28/5|  4.13/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.33/5|      4.56/5|     4.06/5|
|**Completeness**  |  4.45/5|      4.69/5|     4.16/5|
|**Faithfulness**  |  3.75/5|      3.92/5|     3.54/5|
|**Relevance**     |  4.87/5|      4.88/5|     4.80/5|
|**Clarity**       |  4.90/5|      4.91/5|     4.82/5|
|**Overall**       |  4.20/5|      4.39/5|     3.96/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.64s|        0.6%|
|**Reranking**   |    17.28s|       16.4%|
|**Generation**  |    79.21s|       75.3%|
|**Judge**       |     8.09s|        7.7%|
|**Total**       |   105.21s|        100%|

**Min Latency:** 40.79s  |  **Max Latency:** 189.38s

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
|**Single-Hop**  |    222|   144|       60|    17|      64.9%|
|**Multi-Hop**   |    236|    88|      126|    20|      37.3%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|    81|       61|    17|      50.3%|
|**Medium**  |    161|    86|       63|    11|      53.4%|
|**Hard**    |    136|    65|       62|     9|      47.8%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      67.0%|  1.000|         4.40/5|
|**Single-Hop**  |      Medium|     78|      61.5%|  1.000|         4.31/5|
|**Single-Hop**  |        Hard|     56|      66.1%|  1.000|         4.50/5|
|**Multi-Hop**   |        Easy|     73|      30.1%|  0.452|         3.73/5|
|**Multi-Hop**   |      Medium|     83|      45.8%|  0.580|         4.25/5|
|**Multi-Hop**   |        Hard|     80|      35.0%|  0.438|         3.87/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  540.7s|
|**Questions/Second**   |    0.85|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.23%|
|**TPM Utilization**    |   0.23%|

## Index & Orchestrator

|Field             |                     Value|
|------------------|-------------------------:|
|**Index/Job ID**  |  bfai__eval66a_g1_1536_tt|
|**Mode**          |                     local|
|**Endpoint**      |                       N/A|

---

*Report generated: 2025-12-22 18:12:23*
*Checkpoint: C024*
*Judge Model: gemini-2.5-flash*