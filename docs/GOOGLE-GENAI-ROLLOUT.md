# Google GenAI SDK Migration Rollout

**Version:** 1.0  
**Date:** December 17, 2025  
**Purpose:** Migrate eval suite from Vertex AI SDK to unified `google-genai` SDK for Gemini 3 support

---

## Executive Summary

This document outlines the migration from `langchain_google_vertexai.ChatVertexAI` and `vertexai.generative_models` to the unified `google-genai` SDK. This migration is **required** to:

1. Access Gemini 3 Flash/Pro models with full feature support
2. Use `ThinkingLevel` (reasoning) settings
3. Future-proof against SDK deprecation (June 2026)
4. Enable separate billing tracking for eval suite

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target State](#2-target-state)
3. [Model Card Reference](#3-model-card-reference)
4. [Generation Config Reference](#4-generation-config-reference)
5. [Migration Strategy](#5-migration-strategy)
6. [File-by-File Changes](#6-file-by-file-changes)
7. [New Files to Create](#7-new-files-to-create)
8. [Testing Plan](#8-testing-plan)
9. [Rollback Plan](#9-rollback-plan)
10. [Timeline](#10-timeline)

---

## 1. Current State Analysis

### 1.1 SDK Usage Summary

| Component | Current SDK | Status |
|-----------|-------------|--------|
| LLM Judge | `langchain_google_vertexai.ChatVertexAI` | ⚠️ Deprecated June 2026 |
| Generator | `google.genai.Client(vertexai=True)` | ⚠️ Vertex mode, no thinking support |
| Preflight | `vertexai.generative_models.GenerativeModel` | ⚠️ Deprecated June 2026 |
| Embeddings | `vertexai.language_models.TextEmbeddingModel` | ✅ Keep (no migration needed) |
| Vector Search | `google.cloud.aiplatform` | ✅ Keep (no migration needed) |
| Ranking | Google Ranking API | ✅ Keep (no migration needed) |

### 1.2 Files Requiring Migration

**Core Source (`src/`):**

| File | Usage | Priority |
|------|-------|----------|
| `src/judge.py` | `ChatVertexAI` for judge LLM | 🔴 HIGH |
| `src/generator.py` | `genai.Client(vertexai=True)` | 🔴 HIGH |
| `src/preflight.py` | `vertexai.generative_models` | 🟡 MEDIUM |
| `src/ragas_evaluator.py` | `VERTEXAI_*` env vars | 🟡 MEDIUM |
| `src/vector_search.py` | `vertexai.init()` for embeddings | ✅ KEEP |

**Eval Scripts (`scripts/eval/`):**

| File | Usage | Priority |
|------|-------|----------|
| `run_gold_eval.py` | `ChatVertexAI` for judge | 🔴 HIGH |
| `preflight_check.py` | `ChatVertexAI` | 🟡 MEDIUM |
| `gold_standard_eval.py` | `ChatVertexAI` | 🟡 MEDIUM |
| `e2e_orchestrator_test.py` | `ChatVertexAI` | 🟡 MEDIUM |

**Setup Scripts (`scripts/setup/`):**

| File | Usage | Priority |
|------|-------|----------|
| `build_knowledge_graph.py` | `ChatVertexAI` | 🟢 LOW |

**Archive Scripts (`scripts/archive/`):**

| File | Usage | Priority |
|------|-------|----------|
| Multiple files | `ChatVertexAI` | ⚪ SKIP (archive only) |

### 1.3 What Stays on Vertex AI

These components use Vertex AI services (not Gemini LLM) and do NOT need migration:

- **Embeddings** - `TextEmbeddingModel` (text-embedding-004)
- **Vector Search** - `MatchingEngineIndexEndpoint`
- **Ranking API** - Google Ranking API
- **aiplatform.init()** - Still needed for Vector Search

---

## 2. Target State

### 2.1 New Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Eval Suite                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │  gemini_client  │    │  Vertex AI      │                 │
│  │  (google-genai) │    │  (unchanged)    │                 │
│  ├─────────────────┤    ├─────────────────┤                 │
│  │ • Generator     │    │ • Embeddings    │                 │
│  │ • Judge         │    │ • Vector Search │                 │
│  │ • Preflight     │    │ • Ranking API   │                 │
│  └────────┬────────┘    └────────┬────────┘                 │
│           │                      │                           │
│           ▼                      ▼                           │
│  ┌─────────────────┐    ┌─────────────────┐                 │
│  │ Secret Manager  │    │ ADC (default)   │                 │
│  │ gemini-api-key  │    │                 │                 │
│  │ -eval           │    │                 │                 │
│  └─────────────────┘    └─────────────────┘                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Authentication

| Component | Auth Method | Source |
|-----------|-------------|--------|
| Gemini LLM | API Key | Secret Manager: `bf-rag-eval-service/gemini-api-key-eval` |
| Embeddings | ADC | `gcloud auth application-default login` |
| Vector Search | ADC | `gcloud auth application-default login` |
| Ranking API | ADC | `gcloud auth application-default login` |

**API Key Created:** ✅
- Project: `bf-rag-eval-service`
- Key Name: `ragas-eval-suite`
- Secret: `gemini-api-key-eval`
- Tested: Working with `gemini-2.5-flash`

### 2.3 Environment Variables

```bash
# Required for Gemini (google-genai SDK)
# API key fetched from Secret Manager at runtime

# Required for Vertex AI services (embeddings, vector search)
GOOGLE_CLOUD_PROJECT=civic-athlete-473921-c0
GOOGLE_CLOUD_LOCATION=us-east1
```

---

## 3. Model Card Reference

### 3.1 Primary Model: Gemini 3 Flash (Preview)

```python
MODEL_GEMINI_3_FLASH = {
    "model_id": "gemini-3-flash-preview",
    "status": "PUBLIC_PREVIEW",
    "use_case": "Fast, cost-effective. Default for eval suite.",
    
    # Token Limits
    "input_token_limit": 1_048_576,   # 1M tokens
    "output_token_limit": 65_536,      # 65K tokens
    
    # Thinking/Reasoning
    "thinking_enabled": True,
    "thinking_levels": ["LOW", "HIGH"],  # NO MEDIUM
    
    # Fallback
    "fallback_model": "gemini-2.5-flash",
}
```

### 3.2 Secondary Model: Gemini 3 Pro (Preview)

```python
MODEL_GEMINI_3_PRO = {
    "model_id": "gemini-3-pro-preview",
    "status": "PUBLIC_PREVIEW",
    "use_case": "Complex reasoning, highest quality. Use for hard questions.",
    
    # Token Limits
    "input_token_limit": 1_048_576,
    "output_token_limit": 65_536,
    
    # Thinking/Reasoning
    "thinking_enabled": True,
    "thinking_levels": ["LOW", "HIGH"],
    
    # Fallback
    "fallback_model": "gemini-2.5-pro",
}
```

### 3.3 Fallback Models (GA - Stable)

| Model | Use Case | Fallback For |
|-------|----------|--------------|
| `gemini-2.5-flash` | Stable Flash tier | `gemini-3-flash-preview` |
| `gemini-2.5-pro` | Stable Pro tier | `gemini-3-pro-preview` |

---

## 4. Generation Config Reference

### 4.1 Eval Suite Settings

| Setting | Judge | Generator | Notes |
|---------|-------|-----------|-------|
| `temperature` | 0.0 | 0.0 | Deterministic for reproducibility |
| `top_p` | 0.95 | 0.95 | Default |
| `top_k` | 64 | 64 | Default |
| `max_output_tokens` | 2048 | 8192 | Judge needs less |
| `thinking_level` | LOW | LOW | Start conservative |
| `response_mime_type` | `application/json` | - | Structured judge output |

### 4.2 ThinkingLevel Options

```python
from google.genai.types import ThinkingLevel

# Available levels (NO MEDIUM)
ThinkingLevel.THINKING_LEVEL_UNSPECIFIED  # Default behavior
ThinkingLevel.LOW                          # Minimal reasoning, faster
ThinkingLevel.HIGH                         # Extended reasoning, slower
```

**Recommendation for Eval:**
- Start with `LOW` for speed
- Use `HIGH` for debugging failures or complex multi-hop questions

### 4.3 Full Config Example

```python
from google.genai import types

# Judge config (structured JSON output)
JUDGE_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    top_p=0.95,
    top_k=64,
    max_output_tokens=2048,
    response_mime_type="application/json",
    thinking_config=types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.LOW,
        include_thoughts=False,  # Don't need reasoning in output
    ),
)

# Generator config (natural language output)
GENERATOR_CONFIG = types.GenerateContentConfig(
    temperature=0.0,
    top_p=0.95,
    top_k=64,
    max_output_tokens=8192,
    thinking_config=types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.LOW,
        include_thoughts=False,
    ),
)
```

---

## 5. Migration Strategy

### 5.1 Approach: Parallel Implementation

1. **Create new `gemini_client.py`** - Standalone client with all features
2. **Add to existing files** - Import new client alongside old
3. **Feature flag** - `USE_GOOGLE_GENAI=true` to switch
4. **Test thoroughly** - Run eval with both backends
5. **Remove old code** - Once validated

### 5.2 Fallback Chain

```
gemini-3-flash-preview
        │
        ▼ (on 429 or 404)
gemini-2.5-flash
        │
        ▼ (on failure)
    RAISE ERROR
```

### 5.3 Rate Limiting

- Exponential backoff: 1s → 2s → 4s → 8s → 16s (max 60s)
- 5 retry attempts
- Automatic fallback to 2.5 on persistent rate limits

---

## 6. File-by-File Changes

### 6.1 `src/gemini_client.py` (NEW)

Create unified client based on cookbook. Key features:
- Secret Manager integration
- Model fallback chain
- Thinking/reasoning support
- Exponential backoff
- JSON output support

### 6.2 `src/judge.py`

**Before:**
```python
from langchain_google_vertexai import ChatVertexAI

self.judge_llm = ChatVertexAI(
    model_name="gemini-2.0-flash",
    project=config.GCP_PROJECT_ID,
    location=config.GCP_LLM_LOCATION,
    temperature=0.0,
)
```

**After:**
```python
from gemini_client import get_client, JUDGE_CONFIG

self.client = get_client()
self.model = "gemini-3-flash-preview"
self.config = JUDGE_CONFIG
```

### 6.3 `src/generator.py`

**Before:**
```python
self.client = genai.Client(
    vertexai=True, 
    project=config.GCP_PROJECT_ID, 
    location=config.GCP_LLM_LOCATION
)
```

**After:**
```python
from gemini_client import get_client, GENERATOR_CONFIG

self.client = get_client()  # Uses API key from Secret Manager
self.config = GENERATOR_CONFIG
```

### 6.4 `scripts/eval/run_gold_eval.py`

**Before:**
```python
from langchain_google_vertexai import ChatVertexAI

self.judge = ChatVertexAI(
    model_name="gemini-2.0-flash",
    project=PROJECT_ID,
    location=LOCATION,
    temperature=0.0,
)
```

**After:**
```python
from src.gemini_client import generate_json

# In _judge_answer method:
result = generate_json(
    prompt=judge_prompt,
    model="gemini-3-flash-preview",
    thinking_level="LOW",
)
```

---

## 7. New Files to Create

### 7.1 `src/gemini_client.py`

Full implementation based on cookbook with:
- `get_client()` - Singleton client with Secret Manager
- `generate()` - Main generation function with fallback
- `generate_json()` - Structured JSON output
- `JUDGE_CONFIG` - Pre-configured for judge
- `GENERATOR_CONFIG` - Pre-configured for generator

### 7.2 `src/model_config.py`

Model cards and configuration:
- `MODEL_GEMINI_3_FLASH`
- `MODEL_GEMINI_3_PRO`
- `MODEL_GEMINI_25_FLASH`
- `MODEL_GEMINI_25_PRO`
- `MODEL_FALLBACK` mapping

---

## 8. Testing Plan

### 8.1 Unit Tests

1. **Secret Manager access** - Verify API key retrieval
2. **Client initialization** - Verify singleton pattern
3. **Basic generation** - Simple prompt/response
4. **JSON output** - Structured response parsing
5. **Fallback** - Simulate 404, verify fallback triggers

### 8.2 Integration Tests

1. **Judge with new client** - Run 10 questions, compare to old
2. **Generator with new client** - Run 10 questions, compare to old
3. **Full pipeline** - End-to-end with new client

### 8.3 Regression Tests

1. **Run TID_10 subset** - 50 questions from gold corpus
2. **Compare metrics** - Pass rate, latency, scores
3. **Acceptance criteria** - Within 2% of baseline

### 8.4 Test Commands

```bash
# Quick test (10 questions)
python scripts/eval/run_gold_eval.py --quick 10 --workers 1

# Parallel test (20 questions)
python scripts/eval/run_gold_eval.py --quick 20 --workers 5

# Full regression (50 questions)
python scripts/eval/run_gold_eval.py --quick 50 --workers 5
```

---

## 9. Rollback Plan

### 9.1 Feature Flag

```python
# In config.py or .env
USE_GOOGLE_GENAI = os.getenv("USE_GOOGLE_GENAI", "false").lower() == "true"
```

### 9.2 Rollback Steps

1. Set `USE_GOOGLE_GENAI=false`
2. Restart eval process
3. Old `ChatVertexAI` code path activates
4. No code changes needed

### 9.3 Keep Old Code Until

- [ ] 100% of eval runs successful with new client
- [ ] No regression in metrics
- [ ] Stable for 1 week in production

---

## 10. Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Create API key & secret | 10 min | ⏳ Manual |
| 2 | Create `gemini_client.py` | 30 min | Pending |
| 3 | Update `src/judge.py` | 15 min | Pending |
| 4 | Update `src/generator.py` | 15 min | Pending |
| 5 | Update `run_gold_eval.py` | 15 min | Pending |
| 6 | Unit tests | 30 min | Pending |
| 7 | Integration tests (10 questions) | 15 min | Pending |
| 8 | Regression tests (50 questions) | 30 min | Pending |
| 9 | Documentation update | 15 min | Pending |

**Total estimated time:** ~3 hours

---

## Appendix A: Secret Manager Setup

### Create API Key

1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Select project: `civic-athlete-473921-c0`
4. Name: `bfai-eval-suite`
5. Copy the key

### Store in Secret Manager

```bash
# Enable API
gcloud services enable secretmanager.googleapis.com --project=civic-athlete-473921-c0

# Create secret
echo -n "YOUR_API_KEY" | gcloud secrets create gemini-api-key-eval \
  --project=civic-athlete-473921-c0 \
  --replication-policy="automatic" \
  --data-file=-

# Verify
gcloud secrets versions access latest --secret=gemini-api-key-eval \
  --project=civic-athlete-473921-c0
```

---

## Appendix B: Dependencies

### New Dependencies

```txt
# requirements.txt additions
google-genai>=1.54.0
google-cloud-secret-manager>=2.25.0
tenacity>=8.2.3  # Already present
```

### Install

```bash
pip install google-genai google-cloud-secret-manager
```

---

## Appendix C: Quick Reference

### Environment Variables

```bash
# For google-genai (API key from Secret Manager)
GOOGLE_CLOUD_PROJECT=civic-athlete-473921-c0

# For Vertex AI services (embeddings, vector search)
GOOGLE_CLOUD_LOCATION=us-east1
```

### Model IDs

```python
# Gemini 3 (Preview) - Primary
GEMINI_3_FLASH = "gemini-3-flash-preview"
GEMINI_3_PRO = "gemini-3-pro-preview"

# Gemini 2.5 (GA) - Fallback
GEMINI_25_FLASH = "gemini-2.5-flash"
GEMINI_25_PRO = "gemini-2.5-pro"
```

### Thinking Levels

```python
from google.genai.types import ThinkingLevel

ThinkingLevel.LOW   # Fast, minimal reasoning
ThinkingLevel.HIGH  # Slow, extended reasoning
```

---

*Document maintained by BrightFox AI Engineering*
