# Gold Standard RAG Evaluation Report
## Core Evaluation Results

**Date:** December 18, 2025  
**Run ID:** N/A  
**Corpus:** 458 Gold Standard Questions (Single-hop: 222, Multi-hop: 236)  
**Models:** Generation: gemini-3-flash-preview | Judge: gemini-3-flash-preview  
**Embedding:** gemini-embedding-001 (1536 dim)  
**Index:** `bfai__eval66a_g1_1536_tt` | **Topic Type:** ✅ Enabled  
**Configuration:** Precision@25, Recall@100, Hybrid Search, Reranking Enabled

---

## Run Context

> **Run Type:** Core Evaluation  
> **Comparison:** Comparing current run to baseline (same model: gemini-3-flash-preview)  
> **Status:** ✅ All systems nominal

This is a **standard core evaluation** using the full corpus with default configuration. Results should be directly comparable to previous baselines.

---

## Executive Summary

| Metric | Previous | Current | Δ | Status |
|--------|----------|---------|---|--------|
| **Pass Rate** | 92.4% | 88.9% | -3.5% | ⚠️ |
| **Partial Rate** | 6.6% | 7.4% | +0.9% | ⚠️ |
| **Fail Rate** | 1.1% | 3.7% | +2.6% | ⚠️ |
| **Acceptable Rate** | 98.9% | 96.3% | -2.6% | ⚠️ |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ |
| **MRR** | 0.737 | 0.737 | +0.000 | ✅ |
| **Overall Score** | 4.82/5 | 4.72/5 | -0.10 | ⚠️ |

---

## Mean Reciprocal Rank (MRR) Analysis

### What is MRR?

**Mean Reciprocal Rank (MRR)** measures how well the retrieval system ranks the correct document. It's the average of reciprocal ranks across all queries.

**Formula:** MRR = (1/N) × Σ(1/rank_i) where rank_i is the position of the first relevant document.

**Example:**
- Query 1: Correct doc at position 1 → 1/1 = 1.000
- Query 2: Correct doc at position 3 → 1/3 = 0.333
- Query 3: Correct doc at position 2 → 1/2 = 0.500
- **MRR = (1.000 + 0.333 + 0.500) / 3 = 0.611**

**Interpretation:**
- **1.000** = Perfect - correct document always ranked first
- **0.500** = Correct document typically at position 2
- **< 0.200** = Poor - correct document often buried deep in results

### MRR Matrix (Difficulty × Question Type)

| Difficulty | Single-hop | Multi-hop | **Total** |
|------------|------------|-----------|-----------|
| **Easy** | 1.000 (n=88) | 0.452 (n=73) | **0.752** (n=161) |
| **Medium** | 1.000 (n=78) | 0.574 (n=83) | **0.780** (n=161) |
| **Hard** | 1.000 (n=56) | 0.437 (n=80) | **0.669** (n=136) |
| **Total** | **1.000** (n=222) | **0.490** (n=236) | **0.737** (n=458) |

> **Key Insight:** Single-hop questions achieve perfect MRR (1.000) across all difficulty levels - the correct document is always ranked first. Multi-hop questions average 0.490 because they require multiple documents, making ranking more challenging.

---

## Score Scale Definitions

### CORRECTNESS - Is the answer factually correct vs ground truth?
| Score | Definition |
|-------|------------|
| **5** | Fully correct - All facts match ground truth exactly |
| **4** | Mostly correct - Minor omissions or slight inaccuracies |
| **3** | Partially correct - Some correct info but notable errors/gaps |
| **2** | Mostly incorrect - Major factual errors, limited correct info |
| **1** | Incorrect - Fundamentally wrong or contradicts ground truth |

### COMPLETENESS - Does the answer cover all key points?
| Score | Definition |
|-------|------------|
| **5** | Comprehensive - Covers all key points from ground truth |
| **4** | Mostly complete - Covers most key points, minor gaps |
| **3** | Partially complete - Covers some key points, notable gaps |
| **2** | Incomplete - Missing most key points |
| **1** | Severely incomplete - Fails to address the question substantively |

### FAITHFULNESS - Is the answer faithful to context (no hallucinations)?
| Score | Definition |
|-------|------------|
| **5** | Fully faithful - All claims supported by retrieved context |
| **4** | Mostly faithful - Minor unsupported claims |
| **3** | Partially faithful - Some hallucinated or unsupported content |
| **2** | Mostly unfaithful - Significant hallucinations |
| **1** | Unfaithful - Answer contradicts or ignores context |

### RELEVANCE - Is the answer relevant to the question asked?
| Score | Definition |
|-------|------------|
| **5** | Highly relevant - Directly addresses the question |
| **4** | Mostly relevant - Addresses question with minor tangents |
| **3** | Partially relevant - Some relevant content, some off-topic |
| **2** | Mostly irrelevant - Largely off-topic |
| **1** | Irrelevant - Does not address the question |

### CLARITY - Is the answer clear and well-structured?
| Score | Definition |
|-------|------------|
| **5** | Excellent clarity - Well-organized, easy to understand |
| **4** | Good clarity - Clear with minor structural issues |
| **3** | Adequate clarity - Understandable but could be clearer |
| **2** | Poor clarity - Confusing or poorly organized |
| **1** | Very poor clarity - Incoherent or incomprehensible |

### OVERALL SCORE - Holistic assessment of answer quality
| Score | Definition |
|-------|------------|
| **5** | Excellent - Would fully satisfy a user's information need |
| **4** | Good - Useful answer with minor issues |
| **3** | Acceptable - Adequate but has notable shortcomings |
| **2** | Poor - Significant issues, limited usefulness |
| **1** | Unacceptable - Fails to provide useful information |

---

## Score Distributions

### CORRECTNESS
| Score | Count | % |
|-------|-------|---|
| 5 | 390 | 85.2% |
| 4 | 28 | 6.1% |
| 3 | 18 | 3.9% |
| 2 | 11 | 2.4% |
| 1 | 11 | 2.4% |
| **≥3** | **436** | **95.2%** |
| **Avg** | **4.69** | |

### COMPLETENESS
| Score | Count | % |
|-------|-------|---|
| 5 | 400 | 87.3% |
| 4 | 22 | 4.8% |
| 3 | 12 | 2.6% |
| 2 | 13 | 2.8% |
| 1 | 11 | 2.4% |
| **≥3** | **434** | **94.8%** |
| **Avg** | **4.72** | |

### FAITHFULNESS
| Score | Count | % |
|-------|-------|---|
| 5 | 433 | 94.5% |
| 4 | 10 | 2.2% |
| 3 | 6 | 1.3% |
| 2 | 4 | 0.9% |
| 1 | 5 | 1.1% |
| **≥3** | **449** | **98.0%** |
| **Avg** | **4.88** | |

### RELEVANCE
| Score | Count | % |
|-------|-------|---|
| 5 | 433 | 94.5% |
| 4 | 11 | 2.4% |
| 3 | 4 | 0.9% |
| 2 | 5 | 1.1% |
| 1 | 5 | 1.1% |
| **≥3** | **448** | **97.8%** |
| **Avg** | **4.88** | |

### CLARITY
| Score | Count | % |
|-------|-------|---|
| 5 | 424 | 92.6% |
| 4 | 22 | 4.8% |
| 3 | 4 | 0.9% |
| 2 | 1 | 0.2% |
| 1 | 7 | 1.5% |
| **≥3** | **450** | **98.3%** |
| **Avg** | **4.87** | |

### OVERALL SCORE
| Score | Count | % |
|-------|-------|---|
| 5 | 384 | 83.8% |
| 4 | 22 | 4.8% |
| 3 | 22 | 4.8% |
| 2 | 7 | 1.5% |
| 1 | 9 | 2.0% |
| **≥3** | **428** | **93.4%** |
| **Avg** | **4.72** | |

---

## Latency Analysis

### Total Latency by Difficulty

| Difficulty | Avg | Min | Max | Count |
|------------|-----|-----|-----|-------|
| **Easy** | 19.80s | 5.62s | 66.22s | 161 |
| **Medium** | 24.79s | 6.27s | 62.99s | 161 |
| **Hard** | 29.11s | 9.24s | 60.02s | 136 |
| **Overall** | 24.32s | 5.62s | 66.22s | 458 |

### Total Latency by Question Type

| Type | Avg | Min | Max | Count |
|------|-----|-----|-----|-------|
| **Single-hop** | 20.40s | 5.62s | 66.22s | 222 |
| **Multi-hop** | 28.01s | 7.80s | 62.99s | 236 |

### Phase Breakdown

| Phase | Avg | % of Total |
|-------|-----|------------|
| **Retrieval** | 0.386s | 1.6% |
| **Reranking** | 0.219s | 0.9% |
| **Generation** | 22.570s | 92.8% |
| **Judge** | 1.146s | 4.7% |
| **Total** | 24.32s | 100% |

---

## Token & Cost Analysis

### Token Breakdown

| Token Type | Count | Avg per Question |
|------------|-------|------------------|
| **Prompt (Input)** | 4,946,275 | 10,799 |
| **Completion (Output)** | 171,176 | 373 |
| **Thinking** | 1,269,993 | 2,772 |
| **Cached** | 0 | 0 |
| **Total** | **6,387,444** | **13,946** |

### Cost Breakdown (Gemini 2.5 Flash Pricing)

| Component | Rate | Amount |
|-----------|------|--------|
| **Input** | $0.075/1M tokens | $0.3710 |
| **Output** | $0.30/1M tokens | $0.0514 |
| **Thinking** | $0.30/1M tokens | $0.3810 |
| **Total** | | **$0.8033** |
| **Per Question** | | **$0.001754** |

---

## Breakdown by Question Type

| Type | Total | Pass | Partial | Fail | Pass Rate |
|------|-------|------|---------|------|-----------|
| **Single-hop** | 222 | 209 | 8 | 5 | 94.1% |
| **Multi-hop** | 236 | 198 | 26 | 12 | 83.9% |

## Breakdown by Difficulty

| Difficulty | Total | Pass | Partial | Fail | Pass Rate |
|------------|-------|------|---------|------|-----------|
| **Easy** | 161 | 146 | 10 | 5 | 90.7% |
| **Medium** | 161 | 149 | 10 | 2 | 92.5% |
| **Hard** | 136 | 112 | 14 | 10 | 82.4% |

---

## Retry & Error Statistics

### Retry Stats

| Metric | Value |
|--------|-------|
| **Total Questions** | 458 |
| **Succeeded First Try** | 458 |
| **Recovered After Retry** | 0 |
| **Failed All Retries** | 0 |
| **Avg Attempts** | 1.00 |

### Errors by Phase

| Phase | Count |
|-------|-------|
| **Retrieval** | 0 |
| **Rerank** | 0 |
| **Generation** | 0 |
| **Judge** | 0 |
| **Total Errors** | 0 |

---

## Failure Analysis

### Failed Questions (17 total)

| Question ID | Type | Difficulty | Overall Score |
|-------------|------|------------|---------------|
| sh_med_009 | single_hop | medium | 1 |
| sh_hard_019 | single_hop | hard | 1 |
| sh_hard_020 | single_hop | hard | 2 |
| mh_easy_038 | multi_hop | easy | 2 |
| mh_easy_042 | multi_hop | easy | 1 |
| mh_easy_043 | multi_hop | easy | 3 |
| mh_med_044 | multi_hop | medium | 1 |
| mh_hard_017 | multi_hop | hard | 1 |
| mh_hard_022 | multi_hop | hard | 2 |
| mh_hard_025 | multi_hop | hard | 2 |
| mh_hard_026 | multi_hop | hard | 1 |
| mh_hard_027 | multi_hop | hard | 1 |
| mh_hard_031 | multi_hop | hard | 2 |
| mh_hard_035 | multi_hop | hard | 2 |
| mh_hard_068 | multi_hop | hard | 1 |
| sh_easy_079 | single_hop | easy | 2 |
| sh_easy_083 | single_hop | easy | 1 |

### Partial Answers (34 total)

| Question ID | Type | Difficulty | Overall Score |
|-------------|------|------------|---------------|
| sh_easy_007 | single_hop | easy | 3 |
| sh_easy_021 | single_hop | easy | 3 |
| sh_easy_048 | single_hop | easy | 4 |
| sh_med_073 | single_hop | medium | 3 |
| sh_med_074 | single_hop | medium | 3 |
| sh_hard_033 | single_hop | hard | 4 |
| sh_hard_048 | single_hop | hard | 4 |
| sh_hard_049 | single_hop | hard | 3 |
| mh_easy_005 | multi_hop | easy | 4 |
| mh_easy_012 | multi_hop | easy | 3 |
| ... | ... | ... | ... |
| *(24 more)* | | | |

---

## Execution Details

| Metric | Value |
|--------|-------|
| **Run ID** | N/A |
| **Timestamp** | 2025-12-18T15:55:32.473177 |
| **Duration** | 756s (12.6 min) |
| **Questions/Second** | 0.606 |
| **Workers** | 5 |
| **Mode** | local |

---

## Appendix A: Question Distribution

| Dimension | Distribution |
|-----------|--------------|
| **Question Type** | Single-hop: 222 (48.5%) / Multi-hop: 236 (51.5%) |
| **Difficulty** | Easy: 161 (35.2%) / Medium: 161 (35.2%) / Hard: 136 (29.7%) |

---

*Report generated: December 18, 2025 at 17:11*  
*Corpus: 458 Gold Standard Questions*  
*Evaluation Model: gemini-3-flash-preview (LLM-as-Judge)*
