# Gold Standard Benchmark Report

## R{RUN_NUMBER}: {RUN_TITLE}

**Run ID:** R{RUN_NUMBER}  
**Date:** {DATE}  
**Type:** {RUN_TYPE}  
**Corpus:** {CORPUS_COUNT} Gold Standard Questions (Single-hop: {SH_COUNT}, Multi-hop: {MH_COUNT})  
**Baseline:** {BASELINE_FILE}

---

## Configuration

| Parameter | Value A | Value B | Match |
|-----------|---------|---------|-------|
| **Model** | {MODEL_A} | {MODEL_B} | {MATCH_MODEL} |
| **Reasoning** | {REASONING_A} | {REASONING_B} | {MATCH_REASONING} |
| **Temperature** | {TEMP_A} | {TEMP_B} | {MATCH_TEMP} |
| **Recall Top K** | {RECALL_K_A} | {RECALL_K_B} | {MATCH_RECALL} |
| **Precision Top N** | {PRECISION_N_A} | {PRECISION_N_B} | {MATCH_PRECISION} |
| **Hybrid Search** | {HYBRID_A} | {HYBRID_B} | {MATCH_HYBRID} |
| **Reranking** | {RERANK_A} | {RERANK_B} | {MATCH_RERANK} |
| **RRF Alpha** | {ALPHA_A} | {ALPHA_B} | {MATCH_ALPHA} |
| **Embedding** | {EMBED_A} | {EMBED_B} | {MATCH_EMBED} |
| **Index** | {INDEX_A} | {INDEX_B} | {MATCH_INDEX} |

---

## Executive Summary

| Metric | {ENV_A} | {ENV_B} | Δ | Status |
|--------|---------|---------|---|--------|
| **Pass Rate** | {PASS_RATE_A}% | {PASS_RATE_B}% | {DELTA_PASS}% | {STATUS_PASS} |
| **Partial Rate** | {PARTIAL_RATE_A}% | {PARTIAL_RATE_B}% | {DELTA_PARTIAL}% | {STATUS_PARTIAL} |
| **Fail Rate** | {FAIL_RATE_A}% | {FAIL_RATE_B}% | {DELTA_FAIL}% | {STATUS_FAIL} |
| **Acceptable Rate** | {ACCEPT_RATE_A}% | {ACCEPT_RATE_B}% | {DELTA_ACCEPT}% | {STATUS_ACCEPT} |
| **Recall@100** | {RECALL_A}% | {RECALL_B}% | {DELTA_RECALL}% | {STATUS_RECALL} |
| **MRR** | {MRR_A} | {MRR_B} | {DELTA_MRR} | {STATUS_MRR} |
| **Overall Score** | {SCORE_A}/5 | {SCORE_B}/5 | {DELTA_SCORE} | {STATUS_SCORE} |

### Key Finding

{KEY_FINDING}

---

## Latency Analysis

### Overall Latency

| Environment | Total (with Judge) | Client Experience (excl Judge) | Speedup |
|-------------|--------------------|---------------------------------|---------|
| **{ENV_A}** | {TOTAL_LATENCY_A}s | {CLIENT_LATENCY_A}s | — |
| **{ENV_B}** | {TOTAL_LATENCY_B}s | {CLIENT_LATENCY_B}s | **{SPEEDUP}x** |
| **Δ** | {DELTA_TOTAL_LATENCY}s | {DELTA_CLIENT_LATENCY}s | |

> **Note:** "Client Experience" excludes judge latency since judging is eval-only. This is what end users would experience in production.

### Phase Breakdown ({ENV_A})

| Phase | Avg Time | % of Total |
|-------|----------|------------|
| **Retrieval** | {RETRIEVAL_A}s | {RETRIEVAL_PCT_A}% |
| **Reranking** | {RERANK_A}s | {RERANK_PCT_A}% |
| **Generation** | {GEN_A}s | {GEN_PCT_A}% |
| **Judge** | {JUDGE_A}s | {JUDGE_PCT_A}% |
| **Total** | {TOTAL_A}s | 100% |

### Phase Breakdown ({ENV_B})

| Phase | Avg Time | % of Total |
|-------|----------|------------|
| **Retrieval** | {RETRIEVAL_B}s | {RETRIEVAL_PCT_B}% |
| **Reranking** | {RERANK_B}s | {RERANK_PCT_B}% |
| **Generation** | {GEN_B}s | {GEN_PCT_B}% |
| **Judge** | {JUDGE_B}s | {JUDGE_PCT_B}% |
| **Total** | {TOTAL_B}s | 100% |

---

## Token & Cost Analysis

### Token Breakdown

| Token Type | Total | Per Question |
|------------|-------|--------------|
| **Prompt (Input)** | {PROMPT_TOKENS} | {PROMPT_PER_Q} |
| **Completion (Output)** | {COMPLETION_TOKENS} | {COMPLETION_PER_Q} |
| **Thinking** | {THINKING_TOKENS} | {THINKING_PER_Q} |
| **Cached** | {CACHED_TOKENS} | {CACHED_PER_Q} |
| **Total** | {TOTAL_TOKENS} | {TOTAL_PER_Q} |

> **Note:** {TOKEN_NOTE}

### Cost Estimate ({MODEL_PRICING})

| Component | Rate | Cost |
|-----------|------|------|
| **Input** | $0.075/1M tokens | ${INPUT_COST} |
| **Output** | $0.30/1M tokens | ${OUTPUT_COST} |
| **Thinking** | $0.30/1M tokens | ${THINKING_COST} |
| **Total ({CORPUS_COUNT} questions)** | | **${TOTAL_COST}** |
| **Per Question** | | **${COST_PER_Q}** |
| **Per 1,000 Questions** | | **${COST_PER_1K}** |

> **Note:** {COST_NOTE}

---

## Quality Metrics

### Pass/Fail Distribution

| Verdict | {ENV_A} | {ENV_B} | Δ |
|---------|---------|---------|---|
| **Pass** | {PASS_COUNT_A} ({PASS_RATE_A}%) | {PASS_COUNT_B} ({PASS_RATE_B}%) | {DELTA_PASS_COUNT} |
| **Partial** | {PARTIAL_COUNT_A} ({PARTIAL_RATE_A}%) | {PARTIAL_COUNT_B} ({PARTIAL_RATE_B}%) | {DELTA_PARTIAL_COUNT} |
| **Fail** | {FAIL_COUNT_A} ({FAIL_RATE_A}%) | {FAIL_COUNT_B} ({FAIL_RATE_B}%) | {DELTA_FAIL_COUNT} |

### Score Averages

| Dimension | {ENV_A} | {ENV_B} | Δ |
|-----------|---------|---------|---|
| **Correctness** | {CORRECT_A} | {CORRECT_B} | {DELTA_CORRECT} |
| **Completeness** | {COMPLETE_A} | {COMPLETE_B} | {DELTA_COMPLETE} |
| **Faithfulness** | {FAITH_A} | {FAITH_B} | {DELTA_FAITH} |
| **Relevance** | {RELEV_A} | {RELEV_B} | {DELTA_RELEV} |
| **Clarity** | {CLARITY_A} | {CLARITY_B} | {DELTA_CLARITY} |
| **Overall** | {OVERALL_A} | {OVERALL_B} | {DELTA_OVERALL} |

---

## Retrieval Metrics

| Metric | {ENV_A} | {ENV_B} | Δ |
|--------|---------|---------|---|
| **Recall@100** | {RECALL_A}% | {RECALL_B}% | {DELTA_RECALL}% |
| **MRR** | {MRR_A} | {MRR_B} | {DELTA_MRR} |
| **Retrieval Candidates** | {CANDIDATES_A} | {CANDIDATES_B} | {DELTA_CANDIDATES} |

---

## Breakdown by Question Type

### {ENV_A}

| Type | Total | Pass | Partial | Fail | Pass Rate |
|------|-------|------|---------|------|-----------|
| **Single-hop** | {SH_TOTAL_A} | {SH_PASS_A} | {SH_PARTIAL_A} | {SH_FAIL_A} | {SH_RATE_A}% |
| **Multi-hop** | {MH_TOTAL_A} | {MH_PASS_A} | {MH_PARTIAL_A} | {MH_FAIL_A} | {MH_RATE_A}% |

{QTYPE_NOTE_A}

### {ENV_B}

| Type | Total | Pass | Partial | Fail | Pass Rate |
|------|-------|------|---------|------|-----------|
| **Single-hop** | {SH_TOTAL_B} | {SH_PASS_B} | {SH_PARTIAL_B} | {SH_FAIL_B} | {SH_RATE_B}% |
| **Multi-hop** | {MH_TOTAL_B} | {MH_PASS_B} | {MH_PARTIAL_B} | {MH_FAIL_B} | {MH_RATE_B}% |

---

## Breakdown by Difficulty

### {ENV_A}

| Difficulty | Total | Pass | Partial | Fail | Pass Rate |
|------------|-------|------|---------|------|-----------|
| **Easy** | {EASY_TOTAL_A} | {EASY_PASS_A} | {EASY_PARTIAL_A} | {EASY_FAIL_A} | {EASY_RATE_A}% |
| **Medium** | {MED_TOTAL_A} | {MED_PASS_A} | {MED_PARTIAL_A} | {MED_FAIL_A} | {MED_RATE_A}% |
| **Hard** | {HARD_TOTAL_A} | {HARD_PASS_A} | {HARD_PARTIAL_A} | {HARD_FAIL_A} | {HARD_RATE_A}% |

{DIFF_NOTE_A}

### {ENV_B}

| Difficulty | Total | Pass | Partial | Fail | Pass Rate |
|------------|-------|------|---------|------|-----------|
| **Easy** | {EASY_TOTAL_B} | {EASY_PASS_B} | {EASY_PARTIAL_B} | {EASY_FAIL_B} | {EASY_RATE_B}% |
| **Medium** | {MED_TOTAL_B} | {MED_PASS_B} | {MED_PARTIAL_B} | {MED_FAIL_B} | {MED_RATE_B}% |
| **Hard** | {HARD_TOTAL_B} | {HARD_PASS_B} | {HARD_PARTIAL_B} | {HARD_FAIL_B} | {HARD_RATE_B}% |

---

## Failures

### {ENV_A} Failures ({FAIL_COUNT_A})

| Question ID | Type | Difficulty | Overall Score |
|-------------|------|------------|---------------|
{FAILURES_A}

### {ENV_B} Failures ({FAIL_COUNT_B})

| Question ID | Type | Difficulty | Overall Score |
|-------------|------|------------|---------------|
{FAILURES_B}

---

## Execution Details

| Metric | {ENV_A} | {ENV_B} |
|--------|---------|---------|
| **Timestamp** | {TIMESTAMP_A} | {TIMESTAMP_B} |
| **Duration** | {DURATION_A} | {DURATION_B} |
| **Workers** | {WORKERS_A} | {WORKERS_B} |
| **Mode** | {MODE_A} | {MODE_B} |
| **Endpoint** | {ENDPOINT_A} | {ENDPOINT_B} |

---

## Conclusions

{CONCLUSIONS}

### Recommendation

{RECOMMENDATION}

---

## Files

| File | Description |
|------|-------------|
{FILES}

---

*Report generated: {GENERATED_DATE}*  
*Run ID: R{RUN_NUMBER}*  
*Evaluation Model: {JUDGE_MODEL} (LLM-as-Judge)*
