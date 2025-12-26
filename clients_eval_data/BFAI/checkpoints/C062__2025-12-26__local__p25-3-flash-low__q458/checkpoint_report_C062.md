# Checkpoint Report: C062

**Generated:** 2025-12-26 13:02:31
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.6% |
| **Partial Rate** | 6.3% |
| **Fail Rate** | 1.1% |
| **Acceptable Rate** | 98.9% |
| **Overall Score** | 4.84/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 92.6% | -0.7% | ✅ |
| **Acceptable Rate** | 97.6% | 98.9% | +1.3% | ✅ |
| **Fail Rate** | 2.4% | 1.1% | -1.3% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 4.84/5 | +0.03 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 61.6s | -9.2s | 1.15x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8408 | $-0.0128 |
| **Per Question** | $0.001864 | $0.001836 | $-0.000028 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 95.0% | -0.9% |
| **Multi-Hop** | 90.7% | 90.3% | -0.4% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 88.8% | +0.0% |
| **Medium** | 97.5% | 95.7% | -1.9% |
| **Hard** | 93.4% | 93.4% | +0.0% |

### Key Finding

✅ **All metrics within acceptable range of baseline.**

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
|**Recall@100**  |  99.1%|      100.0%|      98.3%|
|**MRR**         |  0.739|       1.000|      0.493|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.751|   0.783|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.84/5|      4.88/5|     4.80/5|
|**Pass Rate**      |   92.6%|       95.0%|      90.3%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.78/5|  4.88/5|  4.85/5|
|**Pass Rate**      |   88.8%|   95.7%|   93.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.77/5|  4.70/5|  4.83/5|  4.76/5|
|**Completeness**  |  4.88/5|  4.84/5|  4.94/5|  4.85/5|
|**Faithfulness**  |  4.94/5|  4.92/5|  4.97/5|  4.93/5|
|**Relevance**     |  4.97/5|  4.96/5|  4.99/5|  4.96/5|
|**Clarity**       |  4.99/5|  4.99/5|  4.99/5|  4.99/5|
|**Overall**       |  4.84/5|  4.78/5|  4.88/5|  4.85/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.77/5|      4.82/5|     4.72/5|
|**Completeness**  |  4.88/5|      4.93/5|     4.83/5|
|**Faithfulness**  |  4.94/5|      4.95/5|     4.94/5|
|**Relevance**     |  4.97/5|      4.97/5|     4.97/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.99/5|
|**Overall**       |  4.84/5|      4.88/5|     4.80/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.65s|        1.0%|
|**Reranking**   |     2.11s|        3.4%|
|**Generation**  |    57.31s|       93.1%|
|**Judge**       |     1.48s|        2.4%|
|**Total**       |    61.55s|        100%|

**Min Latency:** 3.98s  |  **Max Latency:** 165.16s

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
|**Single-Hop**  |    222|   211|        7|     4|      95.0%|
|**Multi-Hop**   |    236|   213|       22|     1|      90.3%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   143|       14|     4|      88.8%|
|**Medium**  |    161|   154|        7|     0|      95.7%|
|**Hard**    |    136|   127|        8|     1|      93.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      93.2%|  1.000|         4.84/5|
|**Single-Hop**  |      Medium|     78|      96.2%|  1.000|         4.90/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.89/5|
|**Multi-Hop**   |        Easy|     73|      83.6%|  0.452|         4.70/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.580|         4.87/5|
|**Multi-Hop**   |        Hard|     80|      91.2%|  0.439|         4.82/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  333.3s|
|**Questions/Second**   |    1.37|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.30%|
|**TPM Utilization**    |   0.30%|

## Index & Orchestrator

|Field             |                     Value|
|------------------|-------------------------:|
|**Index/Job ID**  |  bfai__eval66a_g1_1536_tt|
|**Mode**          |                     local|
|**Endpoint**      |                       N/A|

---

*Report generated: 2025-12-26 13:02:31*
*Checkpoint: C062*
*Judge Model: gemini-3-flash-preview*