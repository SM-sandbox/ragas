# Orchestrator Config Alignment

> **Last Updated:** December 17, 2025  
> **Purpose:** Document alignment between eval suite and orchestrator (sm-dev-01) settings

---

## LLM Schema Integration

**Source of Truth:** `sm-dev-01/services/api/core/llm_schema.json`  
**GCS Location:** `gs://brightfoxai-documents/schemas/llm_schema.json`

The eval suite dynamically loads the schema from the orchestrator to ensure we capture all tracked fields.

### LLM Metadata Fields (all tracked by orchestrator)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| prompt_tokens | int | yes | Input tokens |
| completion_tokens | int | yes | Output tokens |
| thinking_tokens | int | yes | Reasoning tokens (Gemini 2.5/3) |
| total_tokens | int | yes | Total tokens |
| cached_content_tokens | int | no | Cached tokens (cost savings) |
| model_version | str | yes | Exact model version from API |
| finish_reason | str | yes | STOP, MAX_TOKENS, SAFETY, RECITATION, OTHER |
| used_fallback | bool | yes | True if fallback model was used |
| reasoning_effort | str | yes | low, medium, high |
| avg_logprobs | float | no | Confidence metric (typically negative) |
| response_id | str | no | Unique response ID for tracing |

---

## Generation Config Alignment

### Current Settings (Eval Suite ↔ Orchestrator)

| Parameter | Orchestrator Default | Eval Suite Default | Status |
|-----------|---------------------|-------------------|--------|
| **model** | `gemini-3-flash-preview` | `gemini-3-flash-preview` | ✅ Aligned |
| **temperature** | `0.0` | `0.0` | ✅ Aligned |
| **top_p** | `0.95` | `0.95` | ✅ Aligned |
| **top_k** | `64` | `64` | ✅ Aligned |
| **max_output_tokens** | `8192` | `8192` | ✅ Aligned |

### Thinking Config

| Model | Config Type | Values |
|-------|-------------|--------|
| **Gemini 3** | `thinking_level` | `LOW`, `HIGH` (no MEDIUM) |
| **Gemini 2.5** | `thinking_budget` | 1024 (low), 8192 (medium), 24576 (high) |

Eval suite uses `thinking_level=LOW` for judge (fast) and `thinking_level=LOW` for generation by default.

---

## Retrieval Config (for real pipeline tests)

These settings are used when running the full RAG pipeline with Vector Search:

| Setting | Orchestrator Value | Description |
|---------|-------------------|-------------|
| **recall_top_k** | `100` | Initial retrieval from Vector Search |
| **precision_top_n** | `25` | After reranking |
| **enable_hybrid** | `true` | Hybrid search (semantic + keyword) |
| **rrf_ranking_alpha** | `0.5` | Reciprocal Rank Fusion weight |
| **enable_reranking** | `true` | Use Google Ranker |
| **ranking_model** | `semantic-ranker-default@latest` | Ranker model |

### Embedding Config

| Setting | Value |
|---------|-------|
| **embedding_model** | `gemini-embedding-001` |
| **embedding_dimension** | `1536` |

---

## Model Fallback Chain

The orchestrator has automatic fallback on 429 rate limit or model unavailability:

| Primary Model | Fallback Model |
|---------------|----------------|
| `gemini-3-flash-preview` | `gemini-2.5-flash` |
| `gemini-3-pro-preview` | `gemini-2.5-pro` |
| `gemini-2.5-flash` | (none) |
| `gemini-2.5-pro` | (none) |

**Note:** Eval suite does NOT use fallback - we want to test the specific model.

---

## Config Sources

| Config Type | Location |
|-------------|----------|
| **LLM Schema** | `sm-dev-01/services/api/core/llm_schema.json` |
| **Model Cards** | `sm-dev-01/services/api/core/approved_models.py` |
| **Query Config** | `sm-dev-01/services/api/core/config.py` |
| **GCS Master Config** | `gs://brightfoxai-documents/config/config.json` |
| **Local Dev Override** | `sm-dev-01/.dev-config.json` |

---

## Validation

The eval suite validates responses against the orchestrator schema:

```python
from schema_loader import validate_llm_metadata, pre_eval_schema_check

# Pre-eval check
pre_eval_schema_check()  # Logs schema version and required fields

# Validate response
missing = validate_llm_metadata(response["llm_metadata"])
if missing:
    logger.warning(f"Missing fields: {missing}")
```

---

## Test Commands

```bash
# Run all tests
python3 -m pytest tests/test_gemini_client.py tests/test_schema_loader.py -v

# Run llm_metadata tests only
python3 -m pytest tests/test_gemini_client.py -v -k "llm_metadata"

# Run schema validation tests
python3 -m pytest tests/test_schema_loader.py -v -k "validate"
```

---

*Document maintained by BrightFox AI Engineering*
