# Checkpoint Report: C016

**Generated:** 2025-12-21 13:09:26
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 92.6% |
| **Partial Rate** | 6.1% |
| **Fail Rate** | 1.3% |
| **Acceptable Rate** | 98.7% |
| **Overall Score** | 4.82/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 92.6% | -0.2% | ✅ |
| **Acceptable Rate** | 98.5% | 98.7% | +0.2% | ✅ |
| **Fail Rate** | 1.5% | 1.3% | -0.2% | ✅ |
| **MRR** | 0.737 | 0.739 | +0.002 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.82/5 | -0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 76.5s | +33.8s | 0.56x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.8548 | $+0.0114 |
| **Per Question** | $0.001841 | $0.001866 | $+0.000025 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 94.1% | -1.4% |
| **Multi-Hop** | 90.3% | 91.1% | +0.8% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 90.7% | +0.0% |
| **Medium** | 95.0% | 93.8% | -1.2% |
| **Hard** | 92.6% | 93.4% | +0.7% |

### Key Finding

⚠️ **Regressions detected:** latency increased 33.8s

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

|Metric          |  Total|  Single-Hop|  Multi-Hop|
|----------------|------:|-----------:|----------:|
|**Recall@100**  |  99.1%|      100.0%|      98.3%|
|**MRR**         |  0.739|       1.000|      0.493|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.755|   0.781|  0.670|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.82/5|      4.85/5|     4.79/5|
|**Pass Rate**      |   92.6%|       94.1%|      91.1%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.78/5|  4.85/5|  4.84/5|
|**Pass Rate**      |   90.7%|   93.8%|   93.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.78/5|  4.72/5|  4.83/5|  4.77/5|
|**Completeness**  |  4.86/5|  4.83/5|  4.91/5|  4.85/5|
|**Faithfulness**  |  4.90/5|  4.91/5|  4.89/5|  4.90/5|
|**Relevance**     |  4.96/5|  4.94/5|  4.98/5|  4.97/5|
|**Clarity**       |  4.99/5|  5.00/5|  4.99/5|  4.98/5|
|**Overall**       |  4.82/5|  4.78/5|  4.85/5|  4.84/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.78/5|      4.83/5|     4.72/5|
|**Completeness**  |  4.86/5|      4.91/5|     4.83/5|
|**Faithfulness**  |  4.90/5|      4.89/5|     4.92/5|
|**Relevance**     |  4.96/5|      4.94/5|     4.98/5|
|**Clarity**       |  4.99/5|      5.00/5|     4.98/5|
|**Overall**       |  4.82/5|      4.85/5|     4.79/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.26s|        1.6%|
|**Total**       |    76.53s|        100%|

**Min Latency:** 7.83s  |  **Max Latency:** 138.71s

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
|**Single-Hop**  |    222|   209|       10|     3|      94.1%|
|**Multi-Hop**   |    236|   215|       18|     3|      91.1%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   146|       10|     5|      90.7%|
|**Medium**  |    161|   151|       10|     0|      93.8%|
|**Hard**    |    136|   127|        8|     1|      93.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      94.3%|  1.000|         4.82/5|
|**Single-Hop**  |      Medium|     78|      92.3%|  1.000|         4.84/5|
|**Single-Hop**  |        Hard|     56|      96.4%|  1.000|         4.92/5|
|**Multi-Hop**   |        Easy|     73|      86.3%|  0.459|         4.72/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.576|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      91.2%|  0.439|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  554.6s|
|**Questions/Second**   |    0.83|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.04%|
|**TPM Utilization**    |   0.04%|

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

*Report generated: 2025-12-21 13:09:26*
*Checkpoint: C016*
*Judge Model: gemini-3-flash-preview*