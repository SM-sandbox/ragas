# Checkpoint Report: C027

**Generated:** 2025-12-22 23:33:05
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 88.6% |
| **Partial Rate** | 7.9% |
| **Fail Rate** | 3.5% |
| **Acceptable Rate** | 96.5% |
| **Overall Score** | 4.74/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 88.6% | -4.1% | ⚠️ |
| **Acceptable Rate** | 98.5% | 96.5% | -2.0% | ⚠️ |
| **Fail Rate** | 1.5% | 3.5% | +2.0% | ⚠️ |
| **MRR** | 0.737 | 0.740 | +0.003 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.74/5 | -0.09 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 106.6s | +63.8s | 0.40x ❌ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.9098 | $+0.0664 |
| **Per Question** | $0.001841 | $0.001986 | $+0.000145 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 92.3% | -3.2% |
| **Multi-Hop** | 90.3% | 85.2% | -5.1% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 86.3% | -4.3% |
| **Medium** | 95.0% | 91.9% | -3.1% |
| **Hard** | 92.6% | 87.5% | -5.1% |

### Key Finding

⚠️ **Regressions detected:** pass rate dropped 4.1%, acceptable rate dropped 2.0%, latency increased 63.8s

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
| **Model** | gemini-3-flash-preview |
| **Reasoning Effort** | low |
| **Seed** | 42 |

## Retrieval Metrics

### By Question Type

|Metric          |  Total|  Single-Hop|  Multi-Hop|
|----------------|------:|-----------:|----------:|
|**Recall@100**  |  99.1%|      100.0%|      98.3%|
|**MRR**         |  0.740|       1.000|      0.495|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.751|   0.788|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.74/5|      4.81/5|     4.68/5|
|**Pass Rate**      |   88.6%|       92.3%|      85.2%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.72/5|  4.80/5|  4.70/5|
|**Pass Rate**      |   86.3%|   91.9%|   87.5%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.68/5|  4.65/5|  4.76/5|  4.63/5|
|**Completeness**  |  4.81/5|  4.78/5|  4.87/5|  4.79/5|
|**Faithfulness**  |  4.87/5|  4.88/5|  4.84/5|  4.90/5|
|**Relevance**     |  4.96/5|  4.94/5|  4.96/5|  4.98/5|
|**Clarity**       |  4.99/5|  4.99/5|  4.99/5|  4.99/5|
|**Overall**       |  4.74/5|  4.72/5|  4.80/5|  4.70/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.68/5|      4.77/5|     4.61/5|
|**Completeness**  |  4.81/5|      4.87/5|     4.76/5|
|**Faithfulness**  |  4.87/5|      4.86/5|     4.89/5|
|**Relevance**     |  4.96/5|      4.96/5|     4.96/5|
|**Clarity**       |  4.99/5|      4.99/5|     4.99/5|
|**Overall**       |  4.74/5|      4.81/5|     4.68/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     1.23s|        1.2%|
|**Reranking**   |    18.52s|       17.4%|
|**Generation**  |    85.22s|       80.0%|
|**Judge**       |     1.59s|        1.5%|
|**Total**       |   106.56s|        100%|

**Min Latency:** 13.09s  |  **Max Latency:** 219.43s

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
|**Single-Hop**  |    222|   205|        8|     9|      92.3%|
|**Multi-Hop**   |    236|   201|       28|     7|      85.2%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   139|       15|     7|      86.3%|
|**Medium**  |    161|   148|        8|     5|      91.9%|
|**Hard**    |    136|   119|       13|     4|      87.5%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      92.0%|  1.000|         4.82/5|
|**Single-Hop**  |      Medium|     78|      92.3%|  1.000|         4.79/5|
|**Single-Hop**  |        Hard|     56|      92.9%|  1.000|         4.83/5|
|**Multi-Hop**   |        Easy|     73|      79.5%|  0.452|         4.61/5|
|**Multi-Hop**   |      Medium|     83|      91.6%|  0.588|         4.82/5|
|**Multi-Hop**   |        Hard|     80|      83.8%|  0.439|         4.62/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  545.6s|
|**Questions/Second**   |    0.84|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.27%|
|**TPM Utilization**    |   0.27%|

## Index & Orchestrator

|Field             |                     Value|
|------------------|-------------------------:|
|**Index/Job ID**  |  bfai__eval66a_g1_1536_tt|
|**Mode**          |                     local|
|**Endpoint**      |                       N/A|

---

*Report generated: 2025-12-22 23:33:05*
*Checkpoint: C027*
*Judge Model: gemini-3-flash-preview*