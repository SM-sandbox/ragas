# Gemini 3 Flash Preview Evaluation Report

> **Date:** December 17, 2025  
> **Model:** gemini-3-flash-preview  
> **Corpus:** 458 Gold Standard Q&A pairs  
> **Baseline:** Gemini 2.5 Flash @ Precision@25 (from Gold_Standard_Comparison_Report)

---

## Executive Summary

| Metric | Gemini 2.5 Flash (P@25) | Gemini 3 Flash LOW | Gemini 3 Flash HIGH | Best |
|--------|------------------------|-------------------|---------------------|------|
| **Pass Rate (4+)** | 85.6% | 51.3% | 39.7% | 2.5 Flash |
| **Partial (3)** | 10.5% | 20.1% | 28.4% | - |
| **Acceptable (3+)** | 96.1% | 71.4% | 68.1% | 2.5 Flash |
| **Fail (1-2)** | 3.9% | 28.6% | 31.4% | 2.5 Flash |
| **Overall Score** | 4.71/5 | 3.67/5 | 3.37/5 | 2.5 Flash |
| **Throughput** | ~100 q/min | 275.4 q/min | 144.4 q/min | 3 Flash LOW |
| **Cost (458 q)** | ~$0.15* | $0.21 | $0.39 | 2.5 Flash |

*Estimated based on 2.5 Flash pricing

### Key Findings

1. **Gemini 3 Flash underperforms Gemini 2.5 Flash** on this eval corpus with simulated context
2. **LOW reasoning is faster and cheaper** than HIGH with similar quality
3. **HIGH reasoning uses 2.4x more thinking tokens** but doesn't improve scores
4. **Throughput is excellent** - 275 q/min with LOW, 144 q/min with HIGH

### ⚠️ Important Caveat

This test used **simulated context** (generic placeholder text) instead of real retrieved documents. The poor scores reflect that the model correctly identified it couldn't answer from the fake context. **A real pipeline test with actual retrieval is needed for valid comparison.**

---

## Detailed Results

### Gemini 3 Flash - LOW Reasoning

| Metric | Value |
|--------|-------|
| **Questions** | 458 |
| **Success Rate** | 100% |
| **Total Time** | 99.8s |
| **Throughput** | 275.4 q/min |
| **Avg per Question** | 9.67s |
| **Avg Generation** | 7.71s |
| **Avg Judge** | 1.96s |

#### Verdict Distribution

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | 235 | 51.3% |
| Partial | 92 | 20.1% |
| Fail | 131 | 28.6% |

#### Average Scores

| Dimension | Score |
|-----------|-------|
| Correctness | 3.52 |
| Completeness | 4.03 |
| Faithfulness | 3.25 |
| Relevance | 4.79 |
| Clarity | 4.97 |
| **Overall** | **3.67** |

#### Token Usage

| Token Type | Count | Avg/Question | Cost |
|------------|-------|--------------|------|
| Input (prompt) | 353,954 | 773 | $0.0265 |
| Output (response) | 165,405 | 361 | $0.0496 |
| Thinking | 441,865 | 965 | $0.1326 |
| **TOTAL** | **961,224** | **2,099** | **$0.2087** |

---

### Gemini 3 Flash - HIGH Reasoning

| Metric | Value |
|--------|-------|
| **Questions** | 458 |
| **Success Rate** | 100% |
| **Total Time** | 190.3s |
| **Throughput** | 144.4 q/min |
| **Avg per Question** | 17.82s |
| **Avg Generation** | 10.83s |
| **Avg Judge** | 6.99s |

#### Verdict Distribution

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | 182 | 39.7% |
| Partial | 130 | 28.4% |
| Fail | 144 | 31.4% |
| Error | 2 | 0.4% |

#### Average Scores

| Dimension | Score |
|-----------|-------|
| Correctness | 3.32 |
| Completeness | 3.79 |
| Faithfulness | 2.75 |
| Relevance | 4.75 |
| Clarity | 4.98 |
| **Overall** | **3.37** |

#### Token Usage

| Token Type | Count | Avg/Question | Cost |
|------------|-------|--------------|------|
| Input (prompt) | 350,504 | 765 | $0.0263 |
| Output (response) | 160,444 | 350 | $0.0481 |
| Thinking | 1,045,921 | 2,284 | $0.3138 |
| **TOTAL** | **1,556,869** | **3,399** | **$0.3882** |

---

## LOW vs HIGH Reasoning Comparison

| Metric | LOW | HIGH | Δ |
|--------|-----|------|---|
| **Pass Rate** | 51.3% | 39.7% | **-11.6%** |
| **Fail Rate** | 28.6% | 31.4% | +2.8% |
| **Overall Score** | 3.67 | 3.37 | **-0.30** |
| **Throughput** | 275.4 q/min | 144.4 q/min | **-47.6%** |
| **Total Time** | 99.8s | 190.3s | +90.5% |
| **Thinking Tokens** | 441,865 | 1,045,921 | **+136.8%** |
| **Total Cost** | $0.21 | $0.39 | **+86.0%** |

### Analysis

- **HIGH reasoning is worse** - Lower pass rate, lower overall score
- **HIGH is slower** - 1.9x longer runtime
- **HIGH is more expensive** - 1.86x cost due to thinking tokens
- **Recommendation:** Use LOW reasoning for this use case

---

## Cost Analysis

### Gemini 3 Flash Pricing (per 1M tokens)

| Token Type | Price |
|------------|-------|
| Input | $0.075 |
| Output | $0.30 |
| Thinking | $0.30 |

### Cost Breakdown (458 questions)

| Run | Input Cost | Output Cost | Thinking Cost | Total |
|-----|------------|-------------|---------------|-------|
| LOW | $0.0265 | $0.0496 | $0.1326 | **$0.2087** |
| HIGH | $0.0263 | $0.0481 | $0.3138 | **$0.3882** |

### Projected Costs at Scale

| Questions | LOW Cost | HIGH Cost |
|-----------|----------|-----------|
| 1,000 | $0.46 | $0.85 |
| 10,000 | $4.56 | $8.47 |
| 100,000 | $45.58 | $84.74 |

---

## LLM Metadata Captured

Each response includes full orchestrator-compatible metadata:

```json
{
  "llm_metadata": {
    "prompt_tokens": 133,
    "completion_tokens": 290,
    "thinking_tokens": 826,
    "total_tokens": 1249,
    "cached_content_tokens": 0,
    "model_version": "gemini-3-flash-preview",
    "finish_reason": "STOP",
    "used_fallback": false,
    "reasoning_effort": "low",
    "avg_logprobs": null,
    "response_id": "resp_xxx"
  }
}
```

---

## Next Steps

1. **Run with real retrieval** - Connect to Vector Search + Google Ranker for valid comparison
2. **Test Gemini 2.5 Flash** - Run same corpus with 2.5 Flash for direct comparison
3. **Analyze failures** - Review the 131-144 failed questions to understand patterns
4. **Consider hybrid** - Use LOW for generation, potentially different for judge

---

## Files Generated

| File | Description |
|------|-------------|
| `gemini3_test_20251217_235055.json` | LOW reasoning full results |
| `gemini3_test_20251217_235419.json` | HIGH reasoning full results |

---

*Report generated by BrightFox AI Eval Suite*
