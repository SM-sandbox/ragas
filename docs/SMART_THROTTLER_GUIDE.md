# Smart Throttler Guide

**For: Engineers who want intelligent API rate limiting**  
**Last Updated: December 2024**

---

## Overview

The Smart Throttler is an intelligent rate limiter that prevents API rate limit errors while maximizing throughput. It's significantly better than naive "just set workers=10" approaches.

**Location:** `lib/core/smart_throttler/rate_limiter.py`

---

## The Problem with Naive Worker Limits

### Naive Approach:
```python
# "Just use 10 workers"
with ThreadPoolExecutor(max_workers=10) as executor:
    results = executor.map(call_api, questions)
```

**Problems:**
1. **Burst patterns** - All 10 workers fire simultaneously
2. **No awareness of actual limits** - Doesn't know RPM/TPM quotas
3. **Wastes capacity** - If API can handle 1000 RPM, 10 workers is way under
4. **Still hits rate limits** - Bursts can exceed instantaneous limits
5. **No recovery** - When rate limited, all workers fail together

### Result:
- Either too slow (under-utilizing API)
- Or rate limit errors (over-utilizing in bursts)

---

## Smart Throttler Solution

The Smart Throttler solves all these problems:

### Key Features:

| Feature | Benefit |
|---------|---------|
| **Sliding Window Tracking** | Knows exactly how many requests/tokens used in last 60s |
| **Pre-emptive Throttling** | Slows down at 90% capacity, never hits 100% |
| **Semaphore Concurrency** | Limits concurrent requests (e.g., 20) regardless of worker count |
| **Staggered Worker Starts** | Workers start at random intervals, preventing bursts |
| **Model-Specific Limits** | Knows Flash vs Pro have different quotas |
| **Optional Model Rotation** | Can fall back to other models when rate limited |

---

## How It Works

### 1. Sliding Window Tracking

```
Time: ─────────────────────────────────────────────────►
       [60 second window]
       ├─────────────────────────────────────────────┤
       │  Request 1    Request 2    Request 3  ...   │
       │  1000 tokens  1500 tokens  800 tokens       │
       └─────────────────────────────────────────────┘
       
Current RPM: 3 requests
Current TPM: 3300 tokens
```

As old requests age out of the window, capacity becomes available.

### 2. Pre-emptive Throttling (90% Threshold)

```
RPM Limit: 1000
Threshold: 90% = 900

Current RPM: 850  → ✅ Proceed
Current RPM: 920  → ⏳ Wait for capacity
```

By throttling at 90%, we never actually hit the limit and get 429 errors.

### 3. Semaphore Concurrency Control

```python
# Even with 100 workers, only 20 can make API calls simultaneously
self._semaphore = Semaphore(max_concurrent_requests=20)

def acquire_sync(self):
    self._semaphore.acquire()  # Blocks if 20 already in flight
    # ... make API call ...
    
def release(self):
    self._semaphore.release()  # Allow next worker to proceed
```

This is the key insight: **You can have 100 workers for parallelism, but limit concurrent API calls to 20.**

### 4. Staggered Worker Starts

```
Worker 1: Start at t=0.0s
Worker 2: Start at t=0.3s
Worker 3: Start at t=0.7s
Worker 4: Start at t=1.2s
...
```

Random delays prevent the "thundering herd" problem where all workers hit the API at once.

---

## Usage Example

### Basic Usage:

```python
from lib.core.smart_throttler.rate_limiter import SmartRateLimiter, get_limiter_for_model

# Get a limiter configured for your model
limiter = get_limiter_for_model("gemini-3-flash-preview")

# Before each API call
limiter.acquire_sync(estimated_tokens=1000)

try:
    # Make your API call
    response = client.generate(prompt)
finally:
    # Always release after the call
    limiter.release()
```

### With ThreadPoolExecutor:

```python
from concurrent.futures import ThreadPoolExecutor
from lib.core.smart_throttler.rate_limiter import get_limiter_for_model

limiter = get_limiter_for_model("gemini-3-flash-preview")

def process_item(item):
    # Acquire before API call
    limiter.acquire_sync(estimated_tokens=1000)
    try:
        result = call_gemini_api(item)
        return result
    finally:
        limiter.release()

# Can use many workers - semaphore limits actual concurrency
with ThreadPoolExecutor(max_workers=100) as executor:
    results = list(executor.map(process_item, items))
```

---

## Configuration

### Default Configuration:

```python
@dataclass
class RateLimitConfig:
    rpm_limit: int = 1000           # Requests per minute
    tpm_limit: int = 1000000        # Tokens per minute
    threshold: float = 0.9          # Throttle at 90%
    window_size: int = 60           # 60 second sliding window
    min_request_delay: float = 0.05 # 50ms minimum between requests
    max_stagger_delay: float = 2.0  # Up to 2s stagger for workers
    max_concurrent_requests: int = 20  # Semaphore limit
```

### Model-Specific Limits (Tier 3):

| Model | RPM | TPM |
|-------|-----|-----|
| gemini-3-flash-preview | 20,000 | 20,000,000 |
| gemini-2.5-flash | 20,000 | 20,000,000 |
| gemini-2.0-flash | 30,000 | 30,000,000 |
| gemini-3-pro-preview | 2,000 | 8,000,000 |
| gemini-2.5-pro | 2,000 | 8,000,000 |

---

## Comparison: Naive vs Smart

### Scenario: Process 500 items with Gemini API

#### Naive Approach (workers=10):
```
Time: 0s   - 10 requests fire simultaneously
Time: 0.1s - All 10 complete, 10 more fire
Time: 0.2s - Burst pattern continues
Time: 5s   - 429 RATE LIMITED! All workers fail
Time: 35s  - Retry after backoff, more failures
Total: ~15 minutes with errors
```

#### Smart Throttler (workers=100, semaphore=20):
```
Time: 0s   - Workers start staggered over 2s
Time: 2s   - 20 concurrent requests (semaphore limit)
Time: 2.1s - As requests complete, new ones start
Time: 60s  - Smooth throughput, no rate limits
Total: ~5 minutes, zero errors
```

### Results:

| Metric | Naive (10 workers) | Smart (100 workers) |
|--------|-------------------|---------------------|
| **Throughput** | ~50 req/min (with errors) | ~300 req/min |
| **Rate Limit Errors** | Many | Zero |
| **Total Time** | 15+ minutes | 5 minutes |
| **Reliability** | Poor | Excellent |

---

## Monitoring

### Get Current Stats:

```python
limiter = get_limiter_for_model("gemini-3-flash-preview")

stats = limiter.get_stats()
print(f"Current RPM: {stats.current_rpm}/{stats.rpm_limit}")
print(f"Current TPM: {stats.current_tpm}/{stats.tpm_limit}")
print(f"RPM Utilization: {stats.rpm_utilization:.1%}")
print(f"Is Throttled: {stats.is_throttled}")

usage = limiter.get_usage()
print(f"Total Requests: {usage['total_requests']}")
print(f"Total Throttles: {usage['total_throttles']}")
```

---

## When to Use This

**Use Smart Throttler when:**
- Making many API calls (>50)
- Running parallel workers
- Need reliable throughput without errors
- Processing batches of documents/questions

**Probably overkill for:**
- Single API calls
- Interactive/real-time use cases
- Low-volume applications

---

## Getting the Code

The Smart Throttler is available in this repo at:
```
lib/core/smart_throttler/rate_limiter.py
```

If you want to use it in your pipeline code, ask Scott and he'll send you the directory. It's a single file with no external dependencies beyond Python stdlib.

---

## Summary

| Naive Approach | Smart Throttler |
|----------------|-----------------|
| Fixed worker count | Dynamic concurrency |
| Burst patterns | Staggered starts |
| No quota awareness | Sliding window tracking |
| Rate limit errors | Pre-emptive throttling |
| Under or over utilization | Optimal throughput |

**The Smart Throttler lets you use 100 workers while only making 20 concurrent API calls, with zero rate limit errors.**

---

## Questions?

Ask Scott for:
- The Smart Throttler code
- Help integrating it into your pipeline
- Tuning configuration for your use case
