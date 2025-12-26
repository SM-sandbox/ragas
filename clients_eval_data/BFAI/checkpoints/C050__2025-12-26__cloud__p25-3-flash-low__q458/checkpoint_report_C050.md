# Checkpoint Report: C050

**Generated:** 2025-12-26 09:27:44
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 91.7% |
| **Partial Rate** | 7.0% |
| **Fail Rate** | 1.3% |
| **Acceptable Rate** | 98.7% |
| **Overall Score** | 4.81/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 93.2% | 91.7% | -1.5% | ✅ |
| **Acceptable Rate** | 97.6% | 98.7% | +1.1% | ✅ |
| **Fail Rate** | 2.4% | 1.3% | -1.1% | ✅ |
| **MRR** | 0.740 | 0.739 | -0.001 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.81/5 | 4.81/5 | +0.01 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 70.8s | 59.3s | -11.5s | 1.19x ✅ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8536 | $0.8777 | $+0.0242 |
| **Per Question** | $0.001864 | $0.001916 | $+0.000053 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.9% | 92.3% | -3.6% |
| **Multi-Hop** | 90.7% | 91.1% | +0.4% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 88.8% | 88.8% | +0.0% |
| **Medium** | 97.5% | 93.2% | -4.3% |
| **Hard** | 93.4% | 93.4% | +0.0% |

### Key Finding

✅ **All metrics within acceptable range of baseline.**

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
|**Overall Score**  |  4.81/5|      4.85/5|     4.78/5|
|**Pass Rate**      |   91.7%|       92.3%|      91.1%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.76/5|  4.86/5|  4.81/5|
|**Pass Rate**      |   88.8%|   93.2%|   93.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.76/5|  4.73/5|  4.81/5|  4.76/5|
|**Completeness**  |  4.86/5|  4.81/5|  4.91/5|  4.87/5|
|**Faithfulness**  |  4.86/5|  4.84/5|  4.89/5|  4.86/5|
|**Relevance**     |  4.97/5|  4.95/5|  4.98/5|  4.97/5|
|**Clarity**       |  4.98/5|  4.99/5|  4.99/5|  4.96/5|
|**Overall**       |  4.81/5|  4.76/5|  4.86/5|  4.81/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.76/5|      4.82/5|     4.72/5|
|**Completeness**  |  4.86/5|      4.91/5|     4.82/5|
|**Faithfulness**  |  4.86/5|      4.84/5|     4.89/5|
|**Relevance**     |  4.97/5|      4.95/5|     4.98/5|
|**Clarity**       |  4.98/5|      5.00/5|     4.97/5|
|**Overall**       |  4.81/5|      4.85/5|     4.78/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.00s|        0.0%|
|**Reranking**   |     0.00s|        0.0%|
|**Generation**  |     0.00s|        0.0%|
|**Judge**       |     1.81s|        3.1%|
|**Total**       |    59.34s|        100%|

**Min Latency:** 5.74s  |  **Max Latency:** 148.32s

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
|**Single-Hop**  |    222|   205|       13|     4|      92.3%|
|**Multi-Hop**   |    236|   215|       19|     2|      91.1%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   143|       13|     5|      88.8%|
|**Medium**  |    161|   150|       11|     0|      93.2%|
|**Hard**    |    136|   127|        8|     1|      93.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      92.0%|  1.000|         4.83/5|
|**Single-Hop**  |      Medium|     78|      91.0%|  1.000|         4.86/5|
|**Single-Hop**  |        Hard|     56|      94.6%|  1.000|         4.86/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.459|         4.68/5|
|**Multi-Hop**   |      Medium|     83|      95.2%|  0.576|         4.86/5|
|**Multi-Hop**   |        Hard|     80|      92.5%|  0.439|         4.78/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  361.8s|
|**Questions/Second**   |    1.27|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.27%|
|**TPM Utilization**    |   0.27%|

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

*Report generated: 2025-12-26 09:27:44*
*Checkpoint: C050*
*Judge Model: gemini-3-flash-preview*