# Gemini API Rate Limits

**Project:** `bf-rag-eval-service`  
**Service:** `generativelanguage.googleapis.com` (AI Studio)  
**Last Updated:** December 2024

---

## Current Tier: Paid Tier 3

### How to Verify Your Tier

#### Option 1: gcloud CLI
```bash
gcloud alpha quotas info list \
  --service=generativelanguage.googleapis.com \
  --project=bf-rag-eval-service \
  | grep -B 150 "GenerateRequestsPerMinutePerProjectPerModel-PaidTier3" \
  | grep -B 5 "gemini-3-flash"
```

**Expected output for Tier 3:**
```yaml
details:
  value: '20000'
dimensions:
  model: gemini-3-flash
```

If you see `20000` RPM for gemini-3-flash, you're on Tier 3.

#### Option 2: Google Cloud Console

1. Go to: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
2. Select project: `bf-rag-eval-service`
3. Look for "Generate content requests per model (paid tier 3)"
4. Check the limit values for each model

#### Option 3: AI Studio Console

1. Go to: https://aistudio.google.com/
2. Click **Settings** (gear icon)
3. Look for **API Key** or **Usage** section
4. Your tier is based on billing - Tier 3 requires significant monthly spend

---

## Rate Limits by Tier

### Gemini 3 Flash

| Metric | Free | Tier 1 | Tier 2 | Tier 3 |
|--------|------|--------|--------|--------|
| **RPM** (requests/min) | 15 | 500 | 2,000 | **20,000** |
| **TPM** (tokens/min) | 1M | 4M | 10M | **20M** |
| **RPD** (requests/day) | 1,500 | 10,000 | 50,000 | **Unlimited** |

### Gemini 3 Pro

| Metric | Free | Tier 1 | Tier 2 | Tier 3 |
|--------|------|--------|--------|--------|
| **RPM** | 2 | 100 | 500 | **2,000** |
| **TPM** | 32K | 1M | 4M | **8M** |
| **RPD** | 50 | 1,000 | 5,000 | **Unlimited** |

### Gemini 2.5 Flash

| Metric | Free | Tier 1 | Tier 2 | Tier 3 |
|--------|------|--------|--------|--------|
| **RPM** | 15 | 1,000 | 4,000 | **20,000** |
| **TPM** | 1M | 4M | 10M | **20M** |

### Gemini 2.0 Flash

| Metric | Free | Tier 1 | Tier 2 | Tier 3 |
|--------|------|--------|--------|--------|
| **RPM** | 15 | 2,000 | 10,000 | **30,000** |
| **TPM** | 1M | 4M | 10M | **20M** |

---

## Batch API Limits (Tier 3)

| Quota | Gemini 3 Flash |
|-------|----------------|
| **Concurrent Batches** | 100 |
| **Concurrent Tokens** | Unlimited (-1) |

---

## Quick Check Commands

### List all quotas:
```bash
gcloud alpha quotas info list \
  --service=generativelanguage.googleapis.com \
  --project=bf-rag-eval-service
```

### Check specific model RPM:
```bash
gcloud alpha quotas info describe \
  GenerateRequestsPerMinutePerProjectPerModel-PaidTier3 \
  --service=generativelanguage.googleapis.com \
  --project=bf-rag-eval-service \
  | grep -A 5 "gemini-3-flash"
```

### Check batch limits:
```bash
gcloud alpha quotas info list \
  --service=generativelanguage.googleapis.com \
  --project=bf-rag-eval-service \
  | grep -i "batch"
```

---

## How Tiers Are Determined

Tiers are based on **monthly billing spend** on Gemini API:

| Tier | Monthly Spend |
|------|---------------|
| Free | $0 |
| Tier 1 | $0+ (billing enabled) |
| Tier 2 | ~$250+ |
| Tier 3 | ~$1,000+ |

> Note: Exact thresholds may vary. Google doesn't publish exact amounts.

---

## Console Links

- **API Quotas:** https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas?project=bf-rag-eval-service
- **API Metrics:** https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/metrics?project=bf-rag-eval-service
- **AI Studio:** https://aistudio.google.com/
- **Billing:** https://console.cloud.google.com/billing?project=bf-rag-eval-service

---

## Our Current Limits (bf-rag-eval-service)

From `gcloud alpha quotas info list`:

```
Model: gemini-3-flash
├── RPM (Tier 3): 20,000
├── TPM (Tier 3): 20,000,000
├── Batch Concurrent Batches: 100
└── Batch Concurrent Tokens: Unlimited

Model: gemini-3-pro
├── RPM (Tier 3): 2,000
├── TPM (Tier 3): 8,000,000
└── Batch Concurrent Batches: 100
```

---

## Requesting Quota Increases

If you need higher limits:

1. Go to: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
2. Find the quota you want to increase
3. Click the pencil icon to edit
4. Request a higher limit
5. Provide justification

Or use gcloud:
```bash
gcloud alpha quotas preferences create \
  --service=generativelanguage.googleapis.com \
  --project=bf-rag-eval-service \
  --quota-id=GenerateRequestsPerMinutePerProjectPerModel-PaidTier3 \
  --preferred-value=50000 \
  --dimensions=model=gemini-3-flash
```
