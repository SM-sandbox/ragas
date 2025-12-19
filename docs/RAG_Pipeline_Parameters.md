# RAG Pipeline Parameters Reference

## Overview

This document catalogs all configurable parameters in the RAG pipeline, comparing **Local** (gRAG_v3 direct) vs **Cloud Run** (bfai-api) configurations.

**Generated:** December 18, 2025  
**Updated:** December 18, 2025 (RESOLVED)  
**Purpose:** Parameter audit for Local vs Cloud comparison

---

## ✅ RESOLVED: Local vs Cloud Now Match!

### Final Results (December 18, 2025)

| Metric | Local (v2) | Cloud | Delta | Status |
|--------|------------|-------|-------|--------|
| **Pass Rate** | 92.4% | 92.6% | +0.2% | ✅ MATCH |
| **Partial Rate** | 6.6% | 6.6% | +0.0% | ✅ MATCH |
| **Fail Rate** | 1.1% | 0.9% | -0.2% | ✅ MATCH |
| **Recall@100** | 99.1% | 99.1% | +0.0% | ✅ MATCH |
| **MRR** | 0.737 | 0.741 | +0.4% | ✅ MATCH |
| **Overall Score** | 4.82/5 | 4.82/5 | -0.00 | ✅ MATCH |

### Root Causes Fixed

1. **Model mismatch:** Cloud was using `gemini-2.5-flash`, now uses `gemini-3-flash-preview`
2. **Recall measurement:** Added `/retrieve` endpoint returning 100 candidates for apples-to-apples comparison
3. **Precision default:** Changed `defaultPrecisionTopN` from 12 to 25
4. **Reasoning effort:** Added `reasoning_effort` parameter passthrough to Cloud Run

### Files Changed

**gRAG_v3 (Cloud Run):**
- `services/api/routes/config.py` - Fixed defaults (model, precision)
- `services/api/routes/query.py` - Added `/retrieve` endpoint + `reasoning_effort` param

**ragas (Eval):**
- `scripts/eval/run_gold_eval.py` - Uses `/retrieve` for recall, passes model/reasoning
- `tests/test_cloud_config_match.py` - Unit tests to validate config before eval

---

## 1. Retrieval Parameters

### 1.1 Vector Search

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `recall_top_k` | 100 | 100 | Number of candidates in initial recall phase |
| `precision_top_n` | 25 | 25 | Number of results after reranking |
| `similarity_threshold` | 0.0 | 0.0 | Minimum similarity score (0 = no threshold) |

### 1.2 Hybrid Search

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `enable_hybrid` | true | true | Enable semantic + keyword search |
| `rrf_ranking_alpha` | 0.5 | 0.5 | RRF blend: 0.5 = 50% semantic, 50% keyword |
| `retrieval_mode` | hybrid | hybrid | Search mode (vector/keyword/hybrid) |

### 1.3 Reranking

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `enable_reranking` | true | true | Use Google Ranking API |
| `ranking_model` | semantic-ranker-default@latest | semantic-ranker-default@latest | Ranking model |

---

## 2. Generation Parameters

### 2.1 Model Configuration

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `ai_model` | gemini-3-flash-preview | gemini-3-flash-preview | Generator model |
| `temperature` | 0.0 | 0.0 | Generation temperature (0 = deterministic) |
| `reasoning_effort` | low | low | Thinking level (low/medium/high) |

### 2.2 Context/Output

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `max_sources_displayed` | 5 | 5 | Max sources shown in response |
| `streaming` | false | false | Stream responses |

---

## 3. Embedding Parameters

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `embedding_model` | gemini-embedding-001 | gemini-embedding-001 | Embedding model |
| `embedding_dimension` | 1536 | 1536 | Vector dimensions |
| `use_task_type` | true | true | Use task-specific embeddings |

---

## 4. Index Configuration

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `job_id` | bfai__eval66a_g1_1536_tt | bfai__eval66a_g1_1536_tt | Index identifier |
| `deployed_index_id` | idx_bfai_eval66a_g1_1536_tt | idx_bfai_eval66a_g1_1536_tt | Deployed index |
| `chunk_count` | 5659 | 5659 | Total chunks indexed |
| `document_count` | 64 | 64 | Total documents |

---

## 5. Chunking Parameters

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `chunk_size` | 500 | 500 | Target chunk size (tokens) |
| `chunk_overlap` | 100 | 100 | Overlap between chunks |

---

## 6. Rate Limits & Timeouts

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `max_embedding_requests_per_min` | 100 | N/A | Embedding rate limit |
| `config_cache_ttl_seconds` | 60 | N/A | Config cache TTL |
| `http_request_timeout` | 10 | N/A | HTTP timeout |
| `query_timeout` | 120 | 120 | Query timeout |

---

## 7. GCP Infrastructure

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `project_id` | civic-athlete-473921-c0 | civic-athlete-473921-c0 | GCP project |
| `region` | us-east1 | us-east1 | GCP region |
| `bucket_name` | brightfoxai-documents | brightfoxai-documents | GCS bucket |

---

## 8. Cloud Run Specific

| Parameter | Value | Description |
|-----------|-------|-------------|
| `service_name` | bfai-api | Cloud Run service |
| `endpoint` | https://bfai-api-ppfq5ahfsq-ue.a.run.app | Public URL |
| `memory` | 512Mi | Container memory |
| `cpu` | 1 | Container CPU |
| `timeout` | 600 | Request timeout |
| `max_instances` | 100 | Max autoscale instances |

---

## 9. Judge Configuration (Eval Only)

| Parameter | Local Value | Cloud Value | Description |
|-----------|-------------|-------------|-------------|
| `judge_model` | gemini-3-flash-preview | gemini-3-flash-preview | Judge model |
| `judge_reasoning` | LOW | LOW | Judge thinking level (hardcoded) |
| `judge_temperature` | 0.0 | 0.0 | Judge temperature |
| `judge_max_output_tokens` | 2048 | 2048 | Max judge output |

---

## Parameter Sources

### Local Configuration
- **File:** `/Users/scottmacon/Documents/GitHub/gRAG_v3/.dev-config.json`
- **Jobs Config:** GCS `gs://brightfoxai-config/jobs_config.json`
- **QueryConfig:** `gRAG_v3/services/api/core/config.py`

### Cloud Configuration
- **Endpoint:** `https://bfai-api-ppfq5ahfsq-ue.a.run.app/config`
- **Service:** Cloud Run `bfai-api` in `us-east1`

---

## Unit Tests

Run config validation before eval:

```bash
python tests/test_cloud_config_match.py
```

This validates:
- Cloud Run `/config` endpoint returns expected defaults
- Cloud Run `/retrieve` endpoint returns 100 candidates
- Cloud Run `/query` endpoint uses correct model and reasoning

---

## Baseline v2 Parameters (Reference)

From `baselines/baseline_BFAI_v2__2025-12-18__q458.json`:

```json
{
  "config": {
    "precision_k": 25,
    "workers": 5,
    "generator_model": "gemini-3-flash-preview",
    "judge_model": "gemini-3-flash-preview"
  },
  "index_metadata": {
    "job_id": "bfai__eval66a_g1_1536_tt",
    "embedding_model": "gemini-embedding-001",
    "embedding_dimension": 1536
  }
}
```
