# Checkpoint Report: C030

**Generated:** 2025-12-23 07:58:21
**Results File:** `results.json`

## Executive Summary

| Metric | Value |
|--------|-------|
| **Pass Rate** | 91.7% |
| **Partial Rate** | 6.1% |
| **Fail Rate** | 2.2% |
| **Acceptable Rate** | 97.8% |
| **Overall Score** | 4.80/5 |
| **Questions** | 458 |

## Comparison to Gold Baseline

### Quality Metrics

| Metric | Baseline | This Run | Delta | Status |
|--------|----------|----------|-------|--------|
| **Pass Rate** | 92.8% | 91.7% | -1.1% | ✅ |
| **Acceptable Rate** | 98.5% | 97.8% | -0.7% | ✅ |
| **Fail Rate** | 1.5% | 2.2% | +0.7% | ✅ |
| **MRR** | 0.737 | 0.742 | +0.005 | ✅ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **Overall Score** | 4.83/5 | 4.80/5 | -0.03 | ✅ |

### Latency

| Metric | Baseline | This Run | Delta | Speedup |
|--------|----------|----------|-------|---------|
| **Avg Latency** | 42.7s | 63.3s | +20.6s | 0.68x ⚠️ |

### Cost

| Metric | Baseline | This Run | Delta |
|--------|----------|----------|-------|
| **Total Cost** | $0.8434 | $0.8308 | $-0.0126 |
| **Per Question** | $0.001841 | $0.001814 | $-0.000028 |

### By Question Type

| Type | Baseline | This Run | Delta |
|------|----------|----------|-------|
| **Single-Hop** | 95.5% | 94.1% | -1.4% |
| **Multi-Hop** | 90.3% | 89.4% | -0.8% |

### By Difficulty

| Difficulty | Baseline | This Run | Delta |
|------------|----------|----------|-------|
| **Easy** | 90.7% | 89.4% | -1.2% |
| **Medium** | 95.0% | 95.0% | +0.0% |
| **Hard** | 92.6% | 90.4% | -2.2% |

### Key Finding

⚠️ **Regressions detected:** latency increased 20.6s

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
|**MRR**         |  0.742|       1.000|      0.499|

### By Difficulty

|Metric          |    Easy|  Medium|   Hard|
|----------------|-------:|-------:|------:|
|**Recall@100**  |  100.0%|   98.8%|  98.5%|
|**MRR**         |   0.766|   0.779|  0.671|

## Quality Scores

### By Question Type

|Metric             |   Total|  Single-Hop|  Multi-Hop|
|-------------------|-------:|-----------:|----------:|
|**Overall Score**  |  4.80/5|      4.82/5|     4.78/5|
|**Pass Rate**      |   91.7%|       94.1%|      89.4%|

### By Difficulty

|Metric             |    Easy|  Medium|    Hard|
|-------------------|-------:|-------:|-------:|
|**Overall Score**  |  4.73/5|  4.89/5|  4.78/5|
|**Pass Rate**      |   89.4%|   95.0%|   90.4%|

### Score Dimensions

#### By Difficulty

|Dimension         |   Total|    Easy|  Medium|    Hard|
|------------------|-------:|-------:|-------:|-------:|
|**Correctness**   |  4.75/5|  4.70/5|  4.85/5|  4.71/5|
|**Completeness**  |  4.86/5|  4.83/5|  4.94/5|  4.80/5|
|**Faithfulness**  |  4.93/5|  4.92/5|  4.96/5|  4.93/5|
|**Relevance**     |  4.94/5|  4.92/5|  4.99/5|  4.92/5|
|**Clarity**       |  4.98/5|  4.99/5|  4.99/5|  4.97/5|
|**Overall**       |  4.80/5|  4.73/5|  4.89/5|  4.78/5|

#### By Question Type

|Dimension         |   Total|  Single-Hop|  Multi-Hop|
|------------------|-------:|-----------:|----------:|
|**Correctness**   |  4.75/5|      4.80/5|     4.71/5|
|**Completeness**  |  4.86/5|      4.90/5|     4.83/5|
|**Faithfulness**  |  4.93/5|      4.95/5|     4.92/5|
|**Relevance**     |  4.94/5|      4.93/5|     4.96/5|
|**Clarity**       |  4.98/5|      5.00/5|     4.97/5|
|**Overall**       |  4.80/5|      4.82/5|     4.78/5|

## Latency Analysis

|Phase           |  Avg Time|  % of Total|
|----------------|---------:|-----------:|
|**Retrieval**   |     0.75s|        1.2%|
|**Reranking**   |     3.00s|        4.7%|
|**Generation**  |    58.08s|       91.7%|
|**Judge**       |     1.48s|        2.3%|
|**Total**       |    63.30s|        100%|

**Min Latency:** 19.64s  |  **Max Latency:** 132.53s

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
|**Single-Hop**  |    222|   209|        6|     7|      94.1%|
|**Multi-Hop**   |    236|   211|       22|     3|      89.4%|

## Breakdown by Difficulty

|Difficulty  |  Total|  Pass|  Partial|  Fail|  Pass Rate|
|------------|------:|-----:|--------:|-----:|----------:|
|**Easy**    |    161|   144|       10|     7|      89.4%|
|**Medium**  |    161|   153|        8|     0|      95.0%|
|**Hard**    |    136|   123|       10|     3|      90.4%|

## Breakdown by Type × Difficulty

|Type            |  Difficulty|  Count|  Pass Rate|    MRR|  Overall Score|
|----------------|-----------:|------:|----------:|------:|--------------:|
|**Single-Hop**  |        Easy|     88|      93.2%|  1.000|         4.76/5|
|**Single-Hop**  |      Medium|     78|      96.2%|  1.000|         4.90/5|
|**Single-Hop**  |        Hard|     56|      92.9%|  1.000|         4.80/5|
|**Multi-Hop**   |        Easy|     73|      84.9%|  0.483|         4.69/5|
|**Multi-Hop**   |      Medium|     83|      94.0%|  0.571|         4.88/5|
|**Multi-Hop**   |        Hard|     80|      88.8%|  0.440|         4.76/5|

## Execution & Throttling

|Metric                 |   Value|
|-----------------------|-------:|
|**Run Duration**       |  349.5s|
|**Questions/Second**   |    1.31|
|**Max Workers**        |     100|
|**Effective Workers**  |      50|
|**Total Requests**     |     458|
|**Total Throttles**    |       0|
|**RPM Utilization**    |   0.27%|
|**TPM Utilization**    |   0.27%|

## Index & Orchestrator

|Field             |                  Value|
|------------------|----------------------:|
|**Index/Job ID**  |  bfai__eval66b_g1536tt|
|**Mode**          |                  local|
|**Endpoint**      |                    N/A|

---

*Report generated: 2025-12-23 07:58:21*
*Checkpoint: C030*
*Judge Model: gemini-3-flash-preview*