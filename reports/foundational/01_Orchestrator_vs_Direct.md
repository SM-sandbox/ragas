# Orchestrator vs Direct API Comparison

**Date:** December 15, 2025  
**Category:** Foundational  
**Status:** ✅ Validated

---

## 1. Objective

Confirm that the BrightFox orchestrator API produces identical retrieval results to direct Google Vertex AI API calls, with no meaningful latency overhead.

---

## 2. Configuration

| Setting | Value |
|---------|-------|
| Job ID | bfai__eval66a_g1_1536_tt |
| Embedding Model | gemini-embedding-001 |
| Dimensions | 1536 |
| Task Type | RETRIEVAL_QUERY |
| Questions Tested | 224 |

---

## 3. Results

### Retrieval Quality

| Metric | Orchestrator | Direct API | Difference |
|--------|--------------|------------|------------|
| Recall@10 | 93.3% | 93.3% | 0.0% |
| MRR@10 | 0.843 | 0.843 | 0.000 |
| Precision@10 | 57.7% | 57.7% | 0.0% |

### Latency Comparison

| Phase | Orchestrator | Direct API | Overhead |
|-------|--------------|------------|----------|
| Retrieval | 0.203s | 0.192s | +0.011s |
| Total | 0.203s | 0.192s | +0.011s |

---

## 4. Key Findings

1. **Identical Results** - The orchestrator produces byte-for-byte identical retrieval results to direct API calls.

2. **Negligible Overhead** - The orchestrator adds only ~11ms latency, which is within network variance.

3. **Consistent Across Runs** - Multiple test runs confirmed 100% consistency.

---

## 5. Recommendation

✅ **Use the orchestrator for all production workloads.** The abstraction layer provides operational benefits (logging, monitoring, configuration management) with no quality or performance penalty.

---

## 6. Related Reports

- [Embedding Model Comparison](02_Embedding_Model_Comparison.md)
- [Semantic Retrieval Test Report](../archive/Semantic_Retrieval_Test_Report.md)
