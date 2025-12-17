# Google AI Platform Overview

> **Last Updated:** December 17, 2025  
> **Author:** Generated for BrightFox AI  
> **Purpose:** Business logic and platform understanding for developers

---

## Table of Contents

1. [Platform Comparison: AI Studio vs Vertex AI](#platform-comparison)
2. [SDK Deprecation Timeline](#sdk-deprecation-timeline)
3. [The Unified google-genai SDK](#the-unified-google-genai-sdk)
4. [Available Gemini 3 Models](#available-gemini-3-models)
5. [Quota Systems](#quota-systems)
6. [Pricing Structure](#pricing-structure)
7. [Authentication & Secrets Management](#authentication--secrets-management)
8. [Key Decisions for Our Stack](#key-decisions-for-our-stack)

---

## Platform Comparison

Google offers two ways to access Gemini models:

| Feature | Google AI Studio | Vertex AI |
|---------|------------------|-----------|
| **Target Audience** | Developers, startups, prototyping | Enterprise, production workloads |
| **Authentication** | API Key | Google Cloud IAM (ADC); also supports API key in "express mode" |
| **Billing** | Pay-as-you-go via linked Cloud project | Google Cloud billing |
| **Compliance** | Standard | SOC2, HIPAA, FedRAMP available |
| **VPC/Private** | No | Yes |
| **SLA** | Best effort | Enterprise SLAs available |
| **Provisioned Throughput** | No | Yes (guaranteed capacity) |
| **Model Garden** | Gemini only | Gemini + 3rd party (Claude, Mistral, etc.) |
| **Fine-tuning** | Not available (no tunable models) | Full support (supervised & preference tuning) |
| **Pricing** | **Same per-token rates as Vertex AI** | **Same per-token rates as AI Studio** |

### Key Insight

**Per-token pricing is identical** between platforms. The main differences are enterprise features, compliance, private networking, and guaranteed throughput. For most development and even production workloads, AI Studio is sufficient.

---

## SDK Deprecation Timeline

### ⚠️ CRITICAL: SDK Migration Required

| SDK | Package Name | Status | End-of-Life |
|-----|--------------|--------|-------------|
| Google AI Python SDK (OLD) | `google-generativeai` | **DEPRECATED** | Nov 30, 2025 ❌ |
| Google AI JS SDK (OLD) | `@google/generative-ai` | **DEPRECATED** | Aug 31, 2025 ❌ |
| Vertex AI SDK (Python) | `vertexai.generative_models` | **DEPRECATED** | June 24, 2026 |
| Vertex AI SDK (JS) | `@google-cloud/vertexai` | **DEPRECATED** | June 24, 2026 |
| Vertex AI SDK (Java) | `com.google.cloud:google-cloud-vertexai` | **DEPRECATED** | June 24, 2026 |
| Vertex AI SDK (Go) | `cloud.google.com/go/vertexai/genai` | **DEPRECATED** | June 24, 2026 |

### ✅ What to Use Now

| Language | New Package | Install |
|----------|-------------|---------|
| Python | `google-genai` | `pip install google-genai` |
| JavaScript | `@google/genai` | `npm install @google/genai` |
| Go | `google.golang.org/genai` | `go get google.golang.org/genai` |
| Java | `com.google.genai:google-genai` | Maven/Gradle |

---

## The Unified google-genai SDK

The new SDK is **unified** - same code works for both AI Studio and Vertex AI:

```python
from google import genai

# Option 1: AI Studio (API Key)
client = genai.Client(api_key='YOUR_API_KEY')

# Option 2: Vertex AI (Project-based)
client = genai.Client(
    vertexai=True,
    project='bfai-prod',
    location='us-central1'
)

# Option 3: Gemini Developer API via environment variables
# Set: GEMINI_API_KEY or GOOGLE_API_KEY (either works, GEMINI_API_KEY takes precedence)
client = genai.Client()

# Option 4: Vertex AI via environment variables
# Set: GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION
client = genai.Client()
```

### Why This Matters

- **One codebase** for both platforms
- **Easy migration** between AI Studio and Vertex AI
- **Future-proof** - this is Google's go-forward SDK
- **Full feature parity** with deprecated SDKs

---

## Available Gemini 3 Models

As of December 17, 2025:

| Model ID | Type | Input Tokens | Output Tokens | Thinking | Status |
|----------|------|--------------|---------------|----------|--------|
| `gemini-3-flash-preview` | Fast/Cheap | 1,048,576 (1M) | 65,536 (65K) | ✅ Yes | PUBLIC_PREVIEW |
| `gemini-3-pro-preview` | Powerful | 1,048,576 (1M) | 65,536 (65K) | ✅ Yes | PUBLIC_PREVIEW |
| `gemini-3-pro-image-preview` | Image Gen | 65,536 (65K) | 32,768 (32K) | ✅ Yes | PUBLIC_PREVIEW |

### Fallback Models (Gemini 2.5)

| Model ID | Type | Input Tokens | Output Tokens | Thinking | Status |
|----------|------|--------------|---------------|----------|--------|
| `gemini-2.5-flash` | Fast/Cheap | 1,048,576 (1M) | 65,536 (65K) | ✅ Yes | GA |
| `gemini-2.5-pro` | Powerful | 1,048,576 (1M) | 65,536 (65K) | ✅ Yes | GA |
| `gemini-2.5-flash-lite` | Ultra-fast | 1,048,576 (1M) | 65,536 (65K) | ✅ Yes | GA |

---

## Quota Systems

### Gemini Developer API (AI Studio) Rate Limits

Rate limits are measured by:
- **RPM** - Requests per minute
- **TPM** - Tokens per minute
- **RPD** - Requests per day

**View your project's rate limits:** [AI Studio Usage Dashboard](https://aistudio.google.com/usage)

> **Note:** Specific limits vary by model and account. Check the dashboard for your actual limits. You can request higher limits if needed.

### Vertex AI: Dynamic Shared Quota (DSQ)

For Gemini 2.0+ and Gemini 3 models, Vertex AI uses **Dynamic Shared Quota**:

- Capacity dynamically distributed among all customers (shared pool)
- No predefined quota allocation per project
- No quota increase requests needed
- **Trade-off:** You compete with everyone else for capacity
- **Note:** Requests can still be constrained by system rate limits

### ⚠️ Critical Implication

**Exponential backoff is MANDATORY** with DSQ. You will get 429 errors during high-demand periods. There is no guaranteed throughput unless you pay for Provisioned Throughput.

---

## Pricing Structure

### Gemini 3 Pricing (Current - December 2025)

| Model | Context | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|---------|----------------------|------------------------|
| Gemini 3 Pro | ≤200K | $2.00 | $12.00 |
| Gemini 3 Pro | >200K | $4.00 | $18.00 |
| Gemini 3 Flash | ≤200K | See pricing page | See pricing page |
| Gemini 3 Flash | >200K | See pricing page | See pricing page |

### Gemini 2.5 Pricing (Reference)

| Model | Context | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|---------|----------------------|------------------------|
| Gemini 2.5 Flash | ≤200K | ~$0.075 | ~$0.30 |
| Gemini 2.5 Flash | >200K | ~$0.15 | ~$0.60 |
| Gemini 2.5 Pro | ≤200K | ~$1.25 | ~$5.00 |
| Gemini 2.5 Pro | >200K | ~$2.50 | ~$10.00 |

### Cost Optimization

- **Batch API:** 50% discount
- **Context Caching:** Reduces cost for repeated context
- **Grounding with Google Search:** FREE for Gemini 3 until Jan 5, 2026

---

## Authentication & Secrets Management

### Our Setup (bfai-prod)

API keys are stored in **Google Cloud Secret Manager**:

```bash
# Access via CLI
gcloud secrets versions access latest --secret=gemini-api-key --project=bfai-prod
```

```python
# Access via Python
from google.cloud import secretmanager

client = secretmanager.SecretManagerServiceClient()
response = client.access_secret_version(
    request={'name': 'projects/bfai-prod/secrets/gemini-api-key/versions/latest'}
)
api_key = response.payload.data.decode('UTF-8')
```

### Security Best Practices

1. **Never hardcode API keys** in source code
2. **Use Secret Manager** for all credentials
3. **Rotate keys** periodically
4. **Use IAM** for Vertex AI mode (no API key needed)

---

## Key Decisions for Our Stack

### Recommended Approach

1. **Use `google-genai` SDK** - unified, future-proof
2. **Use AI Studio mode** (API key) for Gemini 3 access today
3. **Store API key in Secret Manager** - already configured
4. **Implement exponential backoff** - mandatory for reliability
5. **Set up model fallback chain** (see below)
6. **Monitor costs**

### Migration Priority

| Priority | Action | Deadline |
|----------|--------|----------|
| 🔴 HIGH | Migrate from `google-generativeai` | **ASAP** (already EOL) |
| 🟡 MEDIUM | Migrate from `vertexai.generative_models` | Before June 24, 2026 |
| 🟢 LOW | Evaluate Provisioned Throughput | When hitting rate limits |

### Model Fallback Strategy

All models (Gemini 3 and 2.5) are available via the same `google-genai` SDK and AI Studio API. No platform switching needed.

**Fallback is same-tier:** Flash falls back to Flash, Pro falls back to Pro.

| Primary Model | Fallback Model |
|---------------|----------------|
| `gemini-3-flash-preview` | `gemini-2.5-flash` |
| `gemini-3-pro-preview` | `gemini-2.5-pro` |

**Fallback is configurable:** Enable/disable via config. Only triggers on 429/rate limit or model unavailable.

---

## Quick Reference

### Environment Variables

```bash
# For Gemini Developer API (AI Studio)
# Either works; GEMINI_API_KEY takes precedence if both set
export GEMINI_API_KEY="your-api-key"
# OR
export GOOGLE_API_KEY="your-api-key"

# For Vertex AI
export GOOGLE_CLOUD_PROJECT="bfai-prod"
export GOOGLE_CLOUD_LOCATION="us-central1"
export GOOGLE_GENAI_USE_VERTEXAI="true"
```

### Model IDs

```python
# Gemini 3 (Preview)
GEMINI_3_FLASH = "gemini-3-flash-preview"
GEMINI_3_PRO = "gemini-3-pro-preview"
GEMINI_3_PRO_IMAGE = "gemini-3-pro-image-preview"

# Gemini 2.5 (Stable - Fallback)
GEMINI_25_FLASH = "gemini-2.5-flash"
GEMINI_25_PRO = "gemini-2.5-pro"
GEMINI_25_FLASH_LITE = "gemini-2.5-flash-lite"
```

### Links

- [AI Studio](https://aistudio.google.com)
- [Vertex AI Console](https://console.cloud.google.com/vertex-ai)
- [google-genai SDK Docs](https://googleapis.github.io/python-genai/)
- [Migration Guide](https://cloud.google.com/vertex-ai/generative-ai/docs/deprecations/genai-vertexai-sdk)
- [Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)

---

*Document maintained by BrightFox AI Engineering*
