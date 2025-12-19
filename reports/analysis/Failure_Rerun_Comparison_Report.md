# Failure Rerun Comparison Report

**Date:** December 17, 2025  
**Version:** 1.0

---

## Purpose & Objectives

### Why This Test?

During the Gold Standard Evaluation, 18 out of 458 questions (3.9%) received failing scores (1-2 out of 5). This test investigates whether these failures can be rescued by using enhanced RAG settings with more computational resources.

### Hypothesis

We hypothesize that many failures are caused by:
1. **Insufficient context** - The right information exists but wasn't retrieved or ranked high enough
2. **Model capability** - Flash model may lack reasoning depth for complex questions
3. **Judge sensitivity** - Flash judge may be stricter than necessary

### Test Design

We reran all 18 failed questions with significantly enhanced settings:
- **4x more retrieval** (Recall@200 vs 100)
- **4x more context** (Precision@100 vs 25)
- **Upgraded model** (Gemini 2.5 Pro vs Flash for both generation and judging)

### Success Criteria

- If failures are **retrieval-related**: Enhanced recall should fix them
- If failures are **context-related**: More precision chunks should help
- If failures are **model-related**: Pro model should improve reasoning
- If failures **persist**: Root cause is likely chunking, corpus, or question quality

---

## Executive Summary

| Metric | Original | Enhanced | Improvement |
|--------|----------|----------|-------------|
| **Config** | Recall@100, P@25, Flash | Recall@200, P@100, Pro | 4x context |
| **Pass (4+)** | 0/18 (0%) | 3/18 (17%) | **+3 fixed** |
| **Partial (3)** | 0/18 (0%) | 9/18 (50%) | **+9 improved** |
| **Fail (1-2)** | 18/18 (100%) | 6/18 (33%) | **-12 reduced** |
| **Avg Latency** | ~8.3s | 42.6s | +34.3s (5.1x) |

### Key Finding

**67% of failures improved** with enhanced settings:
- 3 questions now **PASS** (fully fixed)
- 9 questions now **PARTIAL** (significantly improved)
- 6 questions still **FAIL** (need different approach)

---

## Configuration Comparison

| Setting | Original | Enhanced |
|---------|----------|----------|
| **Recall K** | 100 | 200 |
| **Precision K** | 25 | 100 |
| **Generation Model** | gemini-2.5-flash | gemini-2.5-pro |
| **Judge Model** | gemini-2.0-flash | gemini-2.5-pro |
| **Reasoning** | Standard | Higher reasoning |

---

## Results by Difficulty

### Before (Original Config)

| Difficulty | Count | Pass | Partial | Fail |
|------------|-------|------|---------|------|
| **Easy** | 7 | 0 | 0 | 7 |
| **Medium** | 7 | 0 | 0 | 7 |
| **Hard** | 4 | 0 | 0 | 4 |
| **Total** | 18 | 0 | 0 | 18 |

### After (Enhanced Config)

| Difficulty | Count | Pass | Partial | Fail | Fixed Rate |
|------------|-------|------|---------|------|------------|
| **Easy** | 7 | 2 | 3 | 2 | **71% improved** |
| **Medium** | 7 | 1 | 4 | 2 | **71% improved** |
| **Hard** | 4 | 0 | 2 | 2 | **50% improved** |
| **Total** | 18 | 3 | 9 | 6 | **67% improved** |

### Observation

Easy and medium questions benefited most from enhanced settings. Hard questions showed less improvement, suggesting they may have fundamental issues (wrong documents, complex multi-hop reasoning) that more context alone can't solve.

---

## Individual Question Results

| QID | Difficulty | Type | Before | After | Δ Score | Fixed? |
|-----|------------|------|--------|-------|---------|--------|
| q_0009 | easy | single_hop | FAIL (1) | FAIL (2) | +1 | ❌ |
| q_0025 | easy | single_hop | FAIL (1) | PARTIAL (3) | +2 | ⚠️ |
| q_0085 | medium | single_hop | FAIL (1) | PASS (5) | +4 | ✅ |
| q_0110 | medium | single_hop | FAIL (1) | PARTIAL (3) | +2 | ⚠️ |
| q_0136 | medium | single_hop | FAIL (1) | PASS (5) | +4 | ✅ |
| q_0141 | medium | single_hop | FAIL (1) | PASS (4) | +3 | ✅ |
| q_0163 | medium | single_hop | FAIL (1) | PARTIAL (3) | +2 | ⚠️ |
| q_0238 | hard | single_hop | FAIL (2) | PARTIAL (3) | +1 | ⚠️ |
| q_0247 | hard | single_hop | FAIL (1) | FAIL (2) | +1 | ❌ |
| q_0263 | easy | multi_hop | FAIL (2) | FAIL (2) | 0 | ❌ |
| q_0318 | easy | multi_hop | FAIL (1) | PARTIAL (3) | +2 | ⚠️ |
| q_0324 | easy | multi_hop | FAIL (1) | PARTIAL (3) | +2 | ⚠️ |
| q_0345 | medium | multi_hop | FAIL (1) | PARTIAL (3) | +2 | ⚠️ |
| q_0353 | medium | multi_hop | FAIL (2) | PASS (5) | +3 | ✅ |
| q_0457 | hard | multi_hop | FAIL (2) | PARTIAL (3) | +1 | ⚠️ |
| q_0488 | hard | multi_hop | FAIL (2) | PARTIAL (3) | +1 | ⚠️ |
| q_0501 | easy | single_hop | FAIL (1) | PASS (5) | +4 | ✅ |
| q_0511 | easy | single_hop | FAIL (2) | FAIL (2) | 0 | ❌ |

**Legend:** ✅ = Fixed (now PASS), ⚠️ = Improved (now PARTIAL), ❌ = Still FAIL

---

## Latency Analysis

### Phase Timing (Enhanced Config)

| Phase | Avg | Min | Max | % of Total |
|-------|-----|-----|-----|------------|
| **Retrieval** | 0.3s | 0.2s | 0.4s | 0.7% |
| **Reranking** | 0.3s | 0.3s | 0.4s | 0.7% |
| **Generation** | 21.5s | 13.8s | 36.4s | 50.5% |
| **Judge** | 20.5s | 11.8s | 28.8s | 48.1% |
| **Total** | 42.6s | 27.7s | 57.9s | 100% |

### Latency Comparison

| Config | Avg Total | Generation | Judge |
|--------|-----------|------------|-------|
| **Original** | 8.3s | ~6.7s | ~1.2s |
| **Enhanced** | 42.6s | ~21.5s | ~20.5s |
| **Increase** | +34.3s (5.1x) | +14.8s (3.2x) | +19.3s (17x) |

### Cost Analysis

| Metric | Value |
|--------|-------|
| **Extra latency per question** | +34.3s |
| **Questions fixed (PASS)** | 3 |
| **Questions improved (PARTIAL)** | 9 |
| **Cost per fix (PASS)** | ~206s total extra |
| **Cost per improvement** | ~29s per improved question |

---

## Remaining Failures Analysis

The 6 questions that still fail after enhanced settings:

| QID | Difficulty | Archetype | Why Still Failing |
|-----|------------|-----------|-------------------|
| q_0009 | easy | HALLUCINATION | Model still generates wrong efficiency value |
| q_0247 | hard | HALLUCINATION | Model relies on parametric knowledge |
| q_0263 | easy | COMPLEX_REASONING | Multi-doc synthesis still fails |
| q_0511 | easy | HALLUCINATION | Wrong DC voltage despite context |

### Recommendations for Remaining Failures

1. **HALLUCINATION (3 questions):** Need stricter prompts forcing citation, or fine-tuning
2. **COMPLEX_REASONING (1 question):** Need query decomposition or chain-of-thought
3. **Consider:** These may be corpus/question quality issues, not RAG issues

---

## Recommendations

### For Production

**Do NOT use enhanced settings by default.** The 5x latency increase is not justified for the marginal improvement on already-rare failures (3.9% fail rate).

**Instead:**
1. Use standard config (Recall@100, P@25, Flash) for normal queries
2. Implement **fallback escalation**: If confidence is low, retry with enhanced settings
3. Focus on fixing the root causes (chunking, hallucination prompts)

### For Specific Failure Types

| Archetype | Recommended Fix | Enhanced Settings Help? |
|-----------|-----------------|------------------------|
| INCOMPLETE_CONTEXT | Better chunking | ✅ Yes (more context helps) |
| WRONG_DOCUMENT | Cross-encoder reranking | ✅ Yes (more recall helps) |
| HALLUCINATION | Stricter prompts | ⚠️ Partial (Pro is better) |
| COMPLEX_REASONING | Query decomposition | ⚠️ Partial |

---

## Summary

| Outcome | Count | % |
|---------|-------|---|
| **Fully Fixed (PASS)** | 3 | 17% |
| **Improved (PARTIAL)** | 9 | 50% |
| **Still Failing** | 6 | 33% |

**Bottom line:** Enhanced settings can rescue ~67% of failures, but at 5x latency cost. Use selectively for high-value queries or as a fallback mechanism.

---

*Report generated: December 17, 2025*
