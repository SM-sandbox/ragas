# Google Gen AI SDK & AI Studio Migration Guide

**For: Engineers migrating from Vertex AI to AI Studio**  
**Last Updated: December 2024**

---

## Overview

The Gemini 3 models (e.g., `gemini-3-flash-preview`) are **only available through Google AI Studio**, not Vertex AI. This means you need to:

1. Use the **`google-genai` SDK** (not `google-cloud-aiplatform`)
2. Create an **API key** in AI Studio
3. Store the key in **Secret Manager**

This guide explains the setup and how to verify everything works.

---

## Key Differences: Vertex AI vs AI Studio

| Feature | Vertex AI | AI Studio (Gen AI SDK) |
|---------|-----------|------------------------|
| **SDK** | `google-cloud-aiplatform` | `google-genai` |
| **Auth** | Service Account / ADC | API Key |
| **Endpoint** | `{region}-aiplatform.googleapis.com` | `generativelanguage.googleapis.com` |
| **Gemini 3 Models** | ❌ Not available | ✅ Available |
| **Gemini 2.x Models** | ✅ Available | ✅ Available |

**Bottom line:** If you need Gemini 3 models, you must use AI Studio with an API key.

---

## Step 1: Create an API Key in AI Studio

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Sign in with your Google account
3. Click **"Get API Key"** in the left sidebar
4. Click **"Create API Key"**
5. Select your GCP project (e.g., `bf-rag-eval-service`)
6. Copy the generated key

> ⚠️ **Important:** The API key is shown only once. Copy it immediately.

---

## Step 2: Store the Key in Secret Manager

Store the API key in GCP Secret Manager so your code can access it securely.

### Using gcloud CLI:

```bash
# Set your project
gcloud config set project YOUR_PROJECT_ID

# Create the secret
echo -n "YOUR_API_KEY" | gcloud secrets create gemini-api-key-eval \
    --data-file=- \
    --replication-policy="automatic"

# Or if the secret already exists, add a new version:
echo -n "YOUR_API_KEY" | gcloud secrets versions add gemini-api-key-eval \
    --data-file=-
```

### Naming Convention:

Use a descriptive name that indicates:
- What it's for: `gemini-api-key-`
- The use case: `eval`, `pipeline`, `prod`, etc.

Example: `gemini-api-key-pipeline`

---

## Step 3: Grant Access to the Secret

Your service account needs permission to access the secret:

```bash
# Grant access to your service account
gcloud secrets add-iam-policy-binding gemini-api-key-eval \
    --member="serviceAccount:YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

---

## Step 4: Install the SDK

```bash
pip install google-genai google-cloud-secret-manager
```

Or add to your `requirements.txt`:
```
google-genai>=1.0.0
google-cloud-secret-manager>=2.0.0
```

---

## Step 5: Code Implementation

Here's how to use the Gen AI SDK with Secret Manager:

```python
from google import genai
from google.genai import types
from google.cloud import secretmanager

# Configuration
PROJECT_ID = "your-project-id"
SECRET_NAME = "gemini-api-key-pipeline"  # Your secret name

def get_api_key() -> str:
    """Fetch API key from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        request={"name": f"projects/{PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest"}
    )
    return response.payload.data.decode("UTF-8")

def get_gemini_client() -> genai.Client:
    """Create Gemini client with API key."""
    api_key = get_api_key()
    return genai.Client(api_key=api_key)

# Usage
client = get_gemini_client()
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Hello, world!",
    config=types.GenerateContentConfig(
        temperature=0.0,
        max_output_tokens=1024,
    ),
)
print(response.text)
```

---

## Step 6: Verify Setup

### Quick Health Check:

```python
from google import genai
from google.cloud import secretmanager

PROJECT_ID = "your-project-id"
SECRET_NAME = "gemini-api-key-pipeline"

# Get API key
sm = secretmanager.SecretManagerServiceClient()
resp = sm.access_secret_version(
    request={"name": f"projects/{PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest"}
)
api_key = resp.payload.data.decode("UTF-8")

# Test client
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Say 'OK' if you can hear me.",
)
print(f"Response: {response.text}")
print("✅ Setup verified!")
```

### Run this to verify:
```bash
python -c "
from google import genai
from google.cloud import secretmanager

PROJECT_ID = 'your-project-id'
SECRET_NAME = 'gemini-api-key-pipeline'

sm = secretmanager.SecretManagerServiceClient()
resp = sm.access_secret_version(
    request={'name': f'projects/{PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest'}
)
api_key = resp.payload.data.decode('UTF-8')

client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model='gemini-3-flash-preview',
    contents='Say OK if you can hear me.',
)
print(f'Response: {response.text}')
print('Setup verified!')
"
```

---

## Checklist for Windsurf/AI Assistant

Have your AI assistant (Windsurf) read this document and then:

1. **Check SDK Installation:**
   ```bash
   pip show google-genai
   pip show google-cloud-secret-manager
   ```

2. **Verify Secret Exists:**
   ```bash
   gcloud secrets describe YOUR_SECRET_NAME --project=YOUR_PROJECT_ID
   ```

3. **Test Secret Access:**
   ```bash
   gcloud secrets versions access latest --secret=YOUR_SECRET_NAME --project=YOUR_PROJECT_ID
   ```

4. **Run Health Check:**
   - Create a simple test script using the code above
   - Verify it returns a response from `gemini-3-flash-preview`

5. **Update Your Code:**
   - Replace `google-cloud-aiplatform` imports with `google-genai`
   - Update client initialization to use API key
   - Test with a simple generation call

6. **Run Unit Tests:**
   - Ensure all existing tests still pass
   - Add a test for the new client initialization

---

## Common Issues

### "API key not found"
- Check the secret name matches exactly
- Verify the secret exists: `gcloud secrets list --project=YOUR_PROJECT`

### "Permission denied"
- Grant `secretmanager.secretAccessor` role to your service account
- Check you're authenticated: `gcloud auth application-default login`

### "Model not found"
- Gemini 3 models require AI Studio API key
- Verify you're using `google-genai` SDK, not `google-cloud-aiplatform`

### "Rate limited (429)"
- See the Smart Throttler documentation for handling rate limits
- Consider implementing exponential backoff

---

## Reference: Current Setup in ragas/

This repo uses the Gen AI SDK in `lib/clients/gemini_client.py`:

```python
# Configuration
PROJECT_ID = "bf-rag-eval-service"
SECRET_NAME = "gemini-api-key-eval"
DEFAULT_MODEL = "gemini-3-flash-preview"

# Client initialization
from google import genai
from google.cloud import secretmanager

client = genai.Client(api_key=get_api_key())
```

You can use this as a reference implementation.

---

## Questions?

Ask Scott for help with:
- Secret Manager setup
- API key creation
- Access to the Smart Throttler code (see `SMART_THROTTLER_GUIDE.md`)
