# Speed & Performance Optimization Guide

**Version:** 2.0  
**Date:** December 17, 2025  
**Purpose:** Comprehensive analysis of bottlenecks and strategies to maximize evaluation throughput

---

## ⚠️ CRITICAL: Google GenAI vs Vertex AI Endpoints

> **YOU MUST USE GOOGLE-GENAI, NOT VERTEX AI FOR GEMINI 2.5**

### Why This Matters

| Feature | google-genai | Vertex AI |
|---------|--------------|-----------|
| Gemini 2.5 Flash/Pro | ✅ Full support | ⚠️ Limited |
| Reasoning settings | ✅ `thinking_budget` | ❌ Not available |
| Token budgets | ✅ Configurable | ❌ Not available |
| Deprecation | June 2025 | Ongoing |

### The Problem

Vertex AI endpoint (`aiplatform.googleapis.com`) does **NOT** support:
- `thinking_budget` parameter for reasoning control
- Token budget configuration for Gemini 2.5 models
- Full Gemini 2.5 Flash/Pro feature set

### The Solution

Use the **google-genai** SDK (deprecated June 2025, but currently the ONLY way):

```python
# CORRECT - google-genai SDK
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel(
    model_name="gemini-2.5-flash-preview-05-20",
    generation_config={
        "temperature": 0.0,
        "max_output_tokens": 8192,
        "thinking_config": {"thinking_budget": 1024}  # ONLY available here
    }
)

# WRONG - Vertex AI (missing features)
from vertexai.generative_models import GenerativeModel
model = GenerativeModel("gemini-2.5-flash")  # No thinking_budget support!
```

### API Key Setup

```bash
# Get API key from Google AI Studio (not Cloud Console)
# https://aistudio.google.com/app/apikey

export GOOGLE_API_KEY="your-api-key-here"
```

### Migration Plan

Google-genai is deprecated but functional until June 2025. Monitor for:
1. Vertex AI adding `thinking_budget` support
2. New unified SDK announcement
3. Migration path documentation

---

## Table of Contents

1. [Critical: Google GenAI vs Vertex AI](#️-critical-google-genai-vs-vertex-ai-endpoints)
2. [Current Performance Baseline](#1-current-performance-baseline)
3. [Bottleneck Analysis](#2-bottleneck-analysis)
4. [Parallelism Strategies](#3-parallelism-strategies)
5. [Cloud-Based Scaling](#4-cloud-based-scaling)
6. [Code Optimizations](#5-code-optimizations)
7. [Quota Management](#6-quota-management)
8. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Current Performance Baseline

### Phase Timing (from E2E Orchestrator, n=224)

| Phase | Avg | Min | Max | % of Total |
|-------|-----|-----|-----|------------|
| **Retrieval** | 0.252s | 0.166s | 0.452s | **2.6%** |
| **Reranking** | 0.196s | 0.091s | 1.480s | **2.1%** |
| **Generation** | 7.742s | 1.736s | 46.486s | **81.2%** |
| **LLM Judge** | 1.342s | 0.880s | 2.148s | **14.1%** |
| **Total** | 9.532s | 3.289s | 48.539s | 100% |

### Current Throughput

| Metric | Value |
|--------|-------|
| Questions per minute | ~6.3 |
| Questions per hour | ~378 |
| 458-question corpus | ~72 minutes |
| Workers | 1 (sequential) |

### Key Insight

**Generation is 81% of latency.** Retrieval and reranking are negligible (~5% combined). Any optimization effort should focus on:
1. Generation parallelism
2. Judge parallelism
3. Batching strategies

---

## 2. Bottleneck Analysis

### 2.1 The Real Bottlenecks

```
┌─────────────────────────────────────────────────────────────────┐
│                    BOTTLENECK HIERARCHY                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. GENERATION (81%)  ◄─── PRIMARY BOTTLENECK                   │
│     └── Gemini 2.5 Flash API call                               │
│     └── Token generation time (proportional to answer length)   │
│     └── Rate limit: 60 RPM default, 1500 RPM with quota bump    │
│                                                                 │
│  2. LLM JUDGE (14%)   ◄─── SECONDARY BOTTLENECK                 │
│     └── Gemini 2.0 Flash API call                               │
│     └── JSON parsing and retry overhead                         │
│     └── Rate limit: shared with generation                      │
│                                                                 │
│  3. LOCAL MACHINE     ◄─── TERTIARY BOTTLENECK (at scale)       │
│     └── Python GIL limits true parallelism                      │
│     └── Network I/O becomes saturated at ~50 concurrent         │
│     └── Memory for holding results                              │
│                                                                 │
│  4. RETRIEVAL (2.6%)  ◄─── NOT A BOTTLENECK                     │
│     └── Vector Search is fast (~250ms)                          │
│     └── Could batch but minimal gain                            │
│                                                                 │
│  5. RERANKING (2.1%)  ◄─── NOT A BOTTLENECK                     │
│     └── Google Ranking API is fast (~200ms)                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Rate Limits (The Hard Ceiling)

| API | Default Limit | With Quota Increase | Notes |
|-----|---------------|---------------------|-------|
| Gemini 2.5 Flash | 60 RPM | 1500 RPM | Generation |
| Gemini 2.0 Flash | 60 RPM | 1500 RPM | Judge |
| Vertex AI Ranking | 600 RPM | 6000 RPM | Reranking |
| Vector Search | 1000 QPS | 10000 QPS | Retrieval |

**Current bottleneck:** At 60 RPM, you can only do ~1 question/second (generation + judge = 2 API calls).

**With quota increase to 1500 RPM:** Theoretical max = 750 questions/minute = 12.5 questions/second.

### 2.3 Local Machine Limits

| Resource | Limit | Impact |
|----------|-------|--------|
| Python GIL | 1 thread executing Python | Use multiprocessing, not threading |
| Network sockets | ~1000 concurrent | Rarely hit |
| Memory | 16-32GB typical | Can hold ~10K results in memory |
| CPU | 8-16 cores | Not the bottleneck for I/O-bound work |

**Your local machine is NOT the bottleneck** until you hit ~50+ concurrent workers. The API rate limits will stop you first.

---

## 3. Parallelism Strategies

### 3.1 Strategy Comparison

| Strategy | Complexity | Speedup | Best For |
|----------|------------|---------|----------|
| **ThreadPoolExecutor** | Low | 5-10x | I/O-bound, simple |
| **asyncio + aiohttp** | Medium | 10-20x | High concurrency |
| **multiprocessing** | Medium | 5-10x | CPU-bound (not our case) |
| **Cloud Functions** | High | 50-100x | Massive scale |
| **Cloud Run Jobs** | High | 50-100x | Batch processing |

### 3.2 Recommended: ThreadPoolExecutor (Quick Win)

**Current code (sequential):**
```python
results = []
for q in questions:
    result = run_single(q)
    results.append(result)
```

**Parallel code (15 workers):**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_parallel(questions, workers=15):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single, q): q for q in questions}
        for future in tqdm(as_completed(futures), total=len(questions)):
            try:
                result = future.result(timeout=120)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "question": futures[future]})
    return results
```

**Expected speedup:** 5-10x (limited by rate limits, not workers)

### 3.3 Optimal Worker Count

| Rate Limit (RPM) | Optimal Workers | Throughput |
|------------------|-----------------|------------|
| 60 RPM | 3-5 | ~1 q/s |
| 300 RPM | 10-15 | ~5 q/s |
| 1500 RPM | 50-75 | ~25 q/s |

**Formula:** `workers = (rate_limit_rpm / 60) * avg_latency_seconds * 0.8`

For 1500 RPM with 9.5s avg latency: `workers = (1500/60) * 9.5 * 0.8 = 190`

But practically, 50-75 workers is the sweet spot due to:
- Connection overhead
- Retry storms
- Memory usage

### 3.4 Implementation: Parallel Eval Runner

```python
#!/usr/bin/env python3
"""Parallel Gold Standard Evaluation Runner."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
from tqdm import tqdm

class ParallelEvaluator:
    def __init__(self, workers: int = 15, checkpoint_interval: int = 50):
        self.workers = workers
        self.checkpoint_interval = checkpoint_interval
        self.results = []
        self.lock = Lock()
        self.completed = 0
        
    def run_single(self, question: dict) -> dict:
        """Run single question through pipeline."""
        # ... existing pipeline code ...
        pass
    
    def checkpoint(self, path: Path):
        """Thread-safe checkpoint save."""
        with self.lock:
            with open(path, 'w') as f:
                json.dump(self.results, f)
    
    def run(self, questions: list, checkpoint_path: Path) -> list:
        """Run parallel evaluation with checkpointing."""
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.run_single, q): q for q in questions}
            
            for future in tqdm(as_completed(futures), total=len(questions)):
                try:
                    result = future.result(timeout=120)
                    with self.lock:
                        self.results.append(result)
                        self.completed += 1
                        
                        if self.completed % self.checkpoint_interval == 0:
                            self.checkpoint(checkpoint_path)
                            
                except Exception as e:
                    q = futures[future]
                    with self.lock:
                        self.results.append({
                            "question_id": q.get("question_id"),
                            "error": str(e)
                        })
        
        self.checkpoint(checkpoint_path)
        return self.results
```

---

## 4. Cloud-Based Scaling

### 4.1 Why Cloud?

Your local machine becomes the bottleneck when:
1. You need >50 concurrent workers
2. You're running multiple experiments simultaneously
3. You need to process thousands of questions quickly

### 4.2 Option A: Cloud Functions (Serverless)

**Architecture:**
```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   Local     │────▶│  Cloud Pub/Sub      │────▶│  Cloud      │
│   Script    │     │  (question queue)   │     │  Functions  │
└─────────────┘     └─────────────────────┘     │  (N workers)│
                                                └──────┬──────┘
                                                       │
                                                       ▼
                                                ┌─────────────┐
                                                │  Cloud      │
                                                │  Storage    │
                                                │  (results)  │
                                                └─────────────┘
```

**Pros:**
- Auto-scaling (0 to 1000 instances)
- Pay per invocation
- No server management

**Cons:**
- Cold start latency (~1-2s)
- 9-minute max execution time
- More complex deployment

**Implementation:**
```python
# cloud_function/main.py
import functions_framework
from google.cloud import storage

@functions_framework.cloud_event
def evaluate_question(cloud_event):
    """Process single question from Pub/Sub."""
    question = json.loads(base64.b64decode(cloud_event.data["message"]["data"]))
    
    # Run pipeline
    result = run_single(question)
    
    # Save to GCS
    client = storage.Client()
    bucket = client.bucket("bfai-eval-results")
    blob = bucket.blob(f"results/{question['question_id']}.json")
    blob.upload_from_string(json.dumps(result))
    
    return "OK"
```

### 4.3 Option B: Cloud Run Jobs (Batch)

**Architecture:**
```
┌─────────────┐     ┌─────────────────────┐     ┌─────────────┐
│   Local     │────▶│  Cloud Run Job      │────▶│  Cloud      │
│   Trigger   │     │  (batch container)  │     │  Storage    │
└─────────────┘     │  - 100 tasks        │     │  (results)  │
                    │  - 10 parallelism   │     └─────────────┘
                    └─────────────────────┘
```

**Pros:**
- Simple container deployment
- Up to 24-hour execution
- Built-in task parallelism

**Cons:**
- Less granular scaling
- Container startup time

**Implementation:**
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/eval-runner', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/eval-runner']

# Job definition
gcloud run jobs create eval-job \
  --image gcr.io/$PROJECT_ID/eval-runner \
  --tasks 100 \
  --parallelism 10 \
  --task-timeout 3600 \
  --memory 2Gi
```

### 4.4 Option C: Vertex AI Pipelines (ML-Native)

Best for complex, multi-stage evaluations with experiment tracking.

**Pros:**
- Native Vertex AI integration
- Experiment tracking built-in
- Artifact management

**Cons:**
- Steeper learning curve
- Overkill for simple evals

### 4.5 Recommendation

| Use Case | Recommended Solution |
|----------|---------------------|
| <500 questions | Local + ThreadPoolExecutor |
| 500-5000 questions | Cloud Run Jobs |
| >5000 questions | Cloud Functions + Pub/Sub |
| Complex pipelines | Vertex AI Pipelines |

---

## 5. Code Optimizations

### 5.1 Reduce API Calls

**Current:** 2 API calls per question (generation + judge)

**Optimization:** Batch judge calls
```python
# Instead of judging one at a time
for result in results:
    judgment = judge(result)

# Batch judge (if API supports)
judgments = judge_batch(results, batch_size=10)
```

**Savings:** Reduces overhead, but Gemini doesn't support true batching yet.

### 5.2 Cache Retrieval Results

If running multiple experiments on same questions, cache retrieval:
```python
import hashlib
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_retrieve(question_hash: str, config_hash: str):
    return retriever.retrieve(question, config)

def retrieve_with_cache(question: str, config: QueryConfig):
    q_hash = hashlib.md5(question.encode()).hexdigest()
    c_hash = hashlib.md5(str(config).encode()).hexdigest()
    return cached_retrieve(q_hash, c_hash)
```

**Savings:** ~250ms per question on repeat runs.

### 5.3 Async HTTP Calls

Replace synchronous calls with async:
```python
import asyncio
import aiohttp

async def run_single_async(session: aiohttp.ClientSession, question: dict):
    # Retrieval (sync - fast enough)
    chunks = retriever.retrieve(question["question"], config)
    
    # Generation (async)
    gen_response = await session.post(
        generation_url,
        json={"prompt": prompt, "context": context}
    )
    answer = await gen_response.json()
    
    # Judge (async)
    judge_response = await session.post(
        judge_url,
        json={"question": question, "answer": answer}
    )
    judgment = await judge_response.json()
    
    return {"question_id": question["question_id"], "judgment": judgment}

async def run_parallel_async(questions: list, concurrency: int = 50):
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [run_single_async(session, q) for q in questions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
```

**Savings:** 2-3x improvement over ThreadPoolExecutor for high concurrency.

### 5.4 Reduce Answer Length

Longer answers = more generation time.

**Current prompt:** May produce verbose answers.

**Optimized prompt:**
```
Answer the question concisely in 2-3 sentences. 
Include only the most relevant information.
Cite sources using [1], [2], etc.
```

**Savings:** 20-30% reduction in generation time.

### 5.5 Use Faster Models for Judge

| Model | Latency | Quality | Cost |
|-------|---------|---------|------|
| gemini-2.0-flash | 1.3s | High | $ |
| gemini-1.5-flash | 0.8s | Medium | $ |
| gemini-1.5-flash-8b | 0.5s | Lower | $$ |

**Trade-off:** Faster models may have slightly lower judgment quality.

---

## 6. Quota Management

### 6.1 Current Quotas

Check your quotas:
```bash
gcloud services list --enabled
gcloud alpha services quota list --service=aiplatform.googleapis.com
```

### 6.2 Request Quota Increase

1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Filter by: "Vertex AI API"
3. Select: "Generate content requests per minute per region per base model"
4. Click: "Edit Quotas"
5. Request: 1500 RPM (usually approved within 24h)

### 6.3 Quota-Aware Rate Limiting

```python
from ratelimit import limits, sleep_and_retry

# 1500 RPM = 25 per second
@sleep_and_retry
@limits(calls=25, period=1)
def rate_limited_generate(prompt: str, context: str):
    return generator.generate(prompt, context)

@sleep_and_retry
@limits(calls=25, period=1)
def rate_limited_judge(question: str, answer: str):
    return judge.judge(question, answer)
```

### 6.4 Exponential Backoff

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60)
)
def robust_api_call(func, *args, **kwargs):
    return func(*args, **kwargs)
```

---

## 7. Implementation Roadmap

### Phase 1: Quick Wins (This Week)

| Task | Effort | Speedup | Priority |
|------|--------|---------|----------|
| Add ThreadPoolExecutor (15 workers) | 2h | 5-10x | **HIGH** |
| Request quota increase to 1500 RPM | 1h | 10-25x | **HIGH** |
| Add rate limiting with backoff | 1h | Stability | **HIGH** |

**Expected result:** 458 questions in ~10 minutes (vs 72 minutes)

### Phase 2: Optimization (This Month)

| Task | Effort | Speedup | Priority |
|------|--------|---------|----------|
| Implement retrieval caching | 4h | 1.1x | Medium |
| Optimize judge prompt for speed | 2h | 1.2x | Medium |
| Add async HTTP calls | 8h | 2x | Medium |

**Expected result:** 458 questions in ~5 minutes

### Phase 3: Cloud Scale (Next Month)

| Task | Effort | Speedup | Priority |
|------|--------|---------|----------|
| Deploy Cloud Run Job | 16h | 10-50x | Low |
| Set up Pub/Sub + Cloud Functions | 24h | 50-100x | Low |
| Implement result aggregation | 8h | N/A | Low |

**Expected result:** 5000 questions in ~10 minutes

---

## Quick Reference: Speedup Commands

### Enable Parallel Execution (Immediate)

```python
# In scripts/eval/run_gold_eval.py, change:
# OLD:
for q in questions:
    result = self.run_single(q)
    results.append(result)

# NEW:
from concurrent.futures import ThreadPoolExecutor, as_completed
with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(self.run_single, q): q for q in questions}
    for future in tqdm(as_completed(futures), total=len(questions)):
        results.append(future.result())
```

### Request Quota Increase

```bash
# Check current quota
gcloud alpha services quota list \
  --service=aiplatform.googleapis.com \
  --filter="metric:aiplatform.googleapis.com/generate_content_requests"

# Request increase via console
open "https://console.cloud.google.com/iam-admin/quotas?project=$PROJECT_ID"
```

### Monitor Rate Limits

```bash
# Watch for 429 errors in logs
tail -f logs/eval.log | grep -i "429\|rate\|quota"
```

---

## Summary

| Bottleneck | Current | Optimized | Method |
|------------|---------|-----------|--------|
| **Generation** | 7.7s (81%) | 7.7s | Can't reduce, parallelize instead |
| **Judge** | 1.3s (14%) | 1.3s | Can't reduce, parallelize instead |
| **Throughput** | 6.3 q/min | 150+ q/min | ThreadPool + quota increase |
| **458 questions** | 72 min | 3-5 min | Parallel + rate limit bump |

**Bottom line:** Request the quota increase and add ThreadPoolExecutor. That's 90% of the win with 10% of the effort.

---

## File Locations (Updated Structure)

| Component | Path |
|-----------|------|
| Eval scripts | `scripts/eval/` |
| Generator | `src/generator.py` |
| Judge | `src/judge.py` |
| Retriever | `src/retriever.py` |
| Reranker | `src/reranker.py` |
| Config | `src/config.py` |
| Gold corpus | `clients/BFAI/qa/QA_BFAI_gold_v1-0__q458.json` |
| Test results | `clients/BFAI/tests/TID_XX/data/` |
| GCS bucket | `gs://bfai-eval-suite/BFAI/` |

---

*This document should be updated as optimizations are implemented and new bottlenecks are discovered.*
