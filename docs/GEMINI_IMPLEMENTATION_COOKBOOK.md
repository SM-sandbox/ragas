# Gemini Implementation Cookbook

> **Last Updated:** December 17, 2025  
> **Purpose:** Drop this into any codebase for standardized Gemini integration  
> **SDK:** `google-genai` (unified SDK)

---

## Quick Start

```bash
pip install google-genai google-cloud-secret-manager tenacity
```

```python
from google import genai
from google.genai import types

# Initialize client (pulls API key from Secret Manager)
from google.cloud import secretmanager

def get_api_key():
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(
        request={'name': 'projects/bfai-prod/secrets/gemini-api-key/versions/latest'}
    )
    return response.payload.data.decode('UTF-8')

client = genai.Client(api_key=get_api_key())
```

---

## Model Cards

### Primary Models (Gemini 3 - Preview)

#### gemini-3-flash-preview

```python
MODEL_CARD_GEMINI_3_FLASH = {
    "model_id": "gemini-3-flash-preview",
    "version": "3-flash-preview-12-2025",
    "display_name": "Gemini 3 Flash Preview",
    "status": "PUBLIC_PREVIEW",
    "use_case": "Fast, cost-effective tasks. Default for most workloads.",
    
    # Token Limits
    "input_token_limit": 1_048_576,   # 1M tokens
    "output_token_limit": 65_536,      # 65K tokens
    
    # Default Generation Settings
    "defaults": {
        "temperature": 1.0,
        "max_temperature": 2.0,
        "top_p": 0.95,
        "top_k": 64,
    },
    
    # Thinking/Reasoning
    "thinking_enabled": True,
    "thinking_levels": ["LOW", "HIGH"],  # No MEDIUM
    
    # Supported Actions
    "supported_actions": [
        "generateContent",
        "countTokens",
        "createCachedContent",
        "batchGenerateContent"
    ],
    
    # Pricing (per 1M tokens)
    "pricing": {
        "input_per_1m_under_200k": "See pricing page",
        "output_per_1m_under_200k": "See pricing page",
        "input_per_1m_over_200k": "See pricing page",
        "output_per_1m_over_200k": "See pricing page",
    },
    
    # Fallback (same-tier)
    "fallback_model": "gemini-2.5-flash",  # Falls back to 2.5 Flash
}
```

#### gemini-3-pro-preview

```python
MODEL_CARD_GEMINI_3_PRO = {
    "model_id": "gemini-3-pro-preview",
    "version": "3-pro-preview-11-2025",
    "display_name": "Gemini 3 Pro Preview",
    "status": "PUBLIC_PREVIEW",
    "use_case": "Complex reasoning, agentic workflows, highest quality output.",
    
    # Token Limits
    "input_token_limit": 1_048_576,   # 1M tokens
    "output_token_limit": 65_536,      # 65K tokens
    
    # Default Generation Settings
    "defaults": {
        "temperature": 1.0,
        "max_temperature": 2.0,
        "top_p": 0.95,
        "top_k": 64,
    },
    
    # Thinking/Reasoning
    "thinking_enabled": True,
    "thinking_levels": ["LOW", "HIGH"],
    
    # Supported Actions
    "supported_actions": [
        "generateContent",
        "countTokens",
        "createCachedContent",
        "batchGenerateContent"
    ],
    
    # Pricing (per 1M tokens)
    "pricing": {
        "input_per_1m_under_200k": "$2.00",
        "output_per_1m_under_200k": "$12.00",
        "input_per_1m_over_200k": "$4.00",
        "output_per_1m_over_200k": "$18.00",
    },
    
    # Fallback (same-tier)
    "fallback_model": "gemini-2.5-pro",  # Falls back to 2.5 Pro
}
```

### Fallback Models (Gemini 2.5 - Stable)

#### gemini-2.5-flash

```python
MODEL_CARD_GEMINI_25_FLASH = {
    "model_id": "gemini-2.5-flash",
    "version": "001",
    "display_name": "Gemini 2.5 Flash",
    "status": "GA",
    "use_case": "Primary fallback. Fast, reliable, cost-effective.",
    
    # Token Limits
    "input_token_limit": 1_048_576,
    "output_token_limit": 65_536,
    
    # Default Generation Settings
    "defaults": {
        "temperature": 1.0,
        "max_temperature": 2.0,
        "top_p": 0.95,
        "top_k": 64,
    },
    
    # Thinking/Reasoning
    "thinking_enabled": True,
    "thinking_levels": ["LOW", "HIGH"],
    
    # Supported Actions
    "supported_actions": [
        "generateContent",
        "countTokens",
        "createCachedContent",
        "batchGenerateContent"
    ],
    
    # Pricing (per 1M tokens)
    "pricing": {
        "input_per_1m_under_200k": "$0.075",
        "output_per_1m_under_200k": "$0.30",
        "input_per_1m_over_200k": "$0.15",
        "output_per_1m_over_200k": "$0.60",
    },
    
    # Fallback
    "fallback_model": None,  # GA model - no fallback needed
}
```

#### gemini-2.5-pro

```python
MODEL_CARD_GEMINI_25_PRO = {
    "model_id": "gemini-2.5-pro",
    "version": "2.5",
    "display_name": "Gemini 2.5 Pro",
    "status": "GA",
    "use_case": "Secondary fallback for complex tasks when Gemini 3 unavailable.",
    
    # Token Limits
    "input_token_limit": 1_048_576,
    "output_token_limit": 65_536,
    
    # Default Generation Settings
    "defaults": {
        "temperature": 1.0,
        "max_temperature": 2.0,
        "top_p": 0.95,
        "top_k": 64,
    },
    
    # Thinking/Reasoning
    "thinking_enabled": True,
    "thinking_levels": ["LOW", "HIGH"],
    
    # Pricing (per 1M tokens)
    "pricing": {
        "input_per_1m_under_200k": "$1.25",
        "output_per_1m_under_200k": "$5.00",
        "input_per_1m_over_200k": "$2.50",
        "output_per_1m_over_200k": "$10.00",
    },
    
    # Fallback
    "fallback_model": None,  # GA model - no fallback needed
}
```

---

## Generation Config Reference

### All Available Parameters

```python
from google.genai import types

config = types.GenerateContentConfig(
    # === CORE GENERATION SETTINGS ===
    temperature=0.7,              # float: 0.0-2.0, controls randomness
    top_p=0.95,                   # float: nucleus sampling threshold
    top_k=64,                     # float: top-k sampling
    max_output_tokens=8192,       # int: max response length
    stop_sequences=["END"],       # list[str]: stop generation on these
    seed=42,                      # int: for reproducibility
    
    # === PENALTIES ===
    presence_penalty=0.0,         # float: penalize repeated topics
    frequency_penalty=0.0,        # float: penalize repeated tokens
    
    # === THINKING/REASONING ===
    thinking_config=types.ThinkingConfig(
        thinking_level=types.ThinkingLevel.HIGH,  # LOW or HIGH (no MEDIUM)
        include_thoughts=True,     # bool: return reasoning in response
        thinking_budget=None,      # int: optional token budget for thinking
    ),
    
    # === OUTPUT FORMAT ===
    response_mime_type="application/json",  # str: force JSON output
    response_schema=MyPydanticModel,        # Schema for structured output
    
    # === SYSTEM INSTRUCTION ===
    system_instruction="You are a helpful assistant...",
    
    # === TOOLS & FUNCTION CALLING ===
    tools=[my_tool_function],
    tool_config=types.ToolConfig(...),
    automatic_function_calling=types.AutomaticFunctionCallingConfig(...),
    
    # === SAFETY ===
    safety_settings=[
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_ONLY_HIGH"
        )
    ],
    
    # === CACHING ===
    cached_content="cached-content-id",  # str: use cached context
    
    # === MULTIMODAL ===
    response_modalities=["TEXT"],  # list[str]: output types
    media_resolution=types.MediaResolution.MEDIUM,
    
    # === AUDIO/SPEECH ===
    speech_config=types.SpeechConfig(...),
    audio_timestamp=False,
    
    # === IMAGE GENERATION ===
    image_config=types.ImageConfig(...),
)
```

### ThinkingLevel Enum

```python
from google.genai.types import ThinkingLevel

# Available levels (NO MEDIUM)
ThinkingLevel.THINKING_LEVEL_UNSPECIFIED  # Default behavior
ThinkingLevel.LOW                          # Minimal reasoning
ThinkingLevel.HIGH                         # Extended reasoning
```

---

## Standard Implementation

### Base Client with Secret Manager

```python
"""
gemini_client.py - Standard Gemini client for all projects
"""
from google import genai
from google.genai import types
from google.cloud import secretmanager
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ID = "bfai-prod"
SECRET_NAME = "gemini-api-key"

# Model fallback mapping (same-tier fallback)
MODEL_FALLBACK = {
    "gemini-3-flash-preview": "gemini-2.5-flash",  # Flash -> Flash
    "gemini-3-pro-preview": "gemini-2.5-pro",      # Pro -> Pro
    "gemini-2.5-flash": None,                       # GA - no fallback
    "gemini-2.5-pro": None,                         # GA - no fallback
}

# Default model
DEFAULT_MODEL = "gemini-3-flash-preview"

# Default generation config
DEFAULT_CONFIG = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# Fallback enabled by default (set to False to disable)
ENABLE_FALLBACK = True

# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

_client = None
_api_key = None

def _get_api_key() -> str:
    """Fetch API key from Secret Manager (cached)."""
    global _api_key
    if _api_key is None:
        sm_client = secretmanager.SecretManagerServiceClient()
        response = sm_client.access_secret_version(
            request={'name': f'projects/{PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest'}
        )
        _api_key = response.payload.data.decode('UTF-8')
    return _api_key

def get_client() -> genai.Client:
    """Get or create the Gemini client (singleton)."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=_get_api_key())
    return _client

# =============================================================================
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# =============================================================================

class RateLimitError(Exception):
    """Raised when rate limited (429)."""
    pass

class ModelUnavailableError(Exception):
    """Raised when model returns 404 or is unavailable."""
    pass

def _is_rate_limit_error(exception):
    """Check if exception is a rate limit error."""
    error_str = str(exception).lower()
    return "429" in error_str or "rate" in error_str or "quota" in error_str

def _is_model_unavailable(exception):
    """Check if model is unavailable (404)."""
    error_str = str(exception).lower()
    return "404" in error_str or "not found" in error_str

# =============================================================================
# CORE GENERATION FUNCTION
# =============================================================================

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(RateLimitError),
    before_sleep=lambda retry_state: logger.warning(
        f"Rate limited, retrying in {retry_state.next_action.sleep} seconds..."
    )
)
def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig
):
    """Generate content with retry logic for rate limits."""
    try:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            raise RateLimitError(str(e))
        elif _is_model_unavailable(e):
            raise ModelUnavailableError(str(e))
        raise

def generate(
    prompt: str,
    model: str = None,
    thinking_level: str = None,
    include_thoughts: bool = False,
    temperature: float = None,
    max_output_tokens: int = None,
    system_instruction: str = None,
    response_mime_type: str = None,
    response_schema = None,
    enable_fallback: bool = None,
    **kwargs
) -> dict:
    """
    Generate content with optional same-tier fallback.
    
    Args:
        prompt: The input prompt
        model: Model ID (default: DEFAULT_MODEL)
        thinking_level: "LOW" or "HIGH" (optional)
        include_thoughts: Return reasoning in response
        temperature: Override default temperature
        max_output_tokens: Override default max tokens
        system_instruction: System prompt
        response_mime_type: e.g., "application/json"
        response_schema: Pydantic model or dict for structured output
        enable_fallback: Override ENABLE_FALLBACK setting (True/False)
        **kwargs: Additional config parameters
    
    Returns:
        dict with 'text', 'model_used', 'thoughts' (if requested), 'used_fallback'
    """
    client = get_client()
    
    # Determine if fallback is enabled
    use_fallback = enable_fallback if enable_fallback is not None else ENABLE_FALLBACK
    
    # Build config
    config_params = {**DEFAULT_CONFIG}
    
    if temperature is not None:
        config_params["temperature"] = temperature
    if max_output_tokens is not None:
        config_params["max_output_tokens"] = max_output_tokens
    if system_instruction:
        config_params["system_instruction"] = system_instruction
    if response_mime_type:
        config_params["response_mime_type"] = response_mime_type
    if response_schema:
        config_params["response_schema"] = response_schema
    
    # Add thinking config if specified
    if thinking_level:
        level = types.ThinkingLevel.HIGH if thinking_level.upper() == "HIGH" else types.ThinkingLevel.LOW
        config_params["thinking_config"] = types.ThinkingConfig(
            thinking_level=level,
            include_thoughts=include_thoughts
        )
    
    config_params.update(kwargs)
    config = types.GenerateContentConfig(**config_params)
    
    # Determine models to try
    primary_model = model or DEFAULT_MODEL
    fallback_model = MODEL_FALLBACK.get(primary_model) if use_fallback else None
    
    models_to_try = [primary_model]
    if fallback_model:
        models_to_try.append(fallback_model)
    
    # Try each model
    last_error = None
    for i, current_model in enumerate(models_to_try):
        is_fallback = i > 0
        try:
            logger.info(f"Attempting generation with {current_model}" + (" (fallback)" if is_fallback else ""))
            response = _generate_with_retry(client, current_model, prompt, config)
            
            # Parse response
            result = {
                "text": response.text,
                "model_used": current_model,
                "thoughts": None,
                "used_fallback": is_fallback
            }
            
            # Extract thoughts if requested
            if include_thoughts and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'thought') and part.thought:
                        result["thoughts"] = part.text
                        break
            
            return result
            
        except ModelUnavailableError as e:
            logger.warning(f"Model {current_model} unavailable: {e}")
            last_error = e
            continue
        except RateLimitError as e:
            logger.warning(f"Rate limit hit for {current_model}, trying fallback..." if fallback_model and not is_fallback else f"Rate limit exhausted for {current_model}: {e}")
            last_error = e
            continue
        except Exception as e:
            logger.error(f"Unexpected error with {current_model}: {e}")
            last_error = e
            continue
    
    raise Exception(f"All models failed. Last error: {last_error}")

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_fast(prompt: str, **kwargs) -> str:
    """Quick generation with Gemini 3 Flash (or fallback)."""
    result = generate(prompt, model="gemini-3-flash-preview", **kwargs)
    return result["text"]

def generate_smart(prompt: str, **kwargs) -> str:
    """High-quality generation with Gemini 3 Pro (or fallback)."""
    result = generate(prompt, model="gemini-3-pro-preview", **kwargs)
    return result["text"]

def generate_with_reasoning(prompt: str, thinking_level: str = "HIGH", **kwargs) -> dict:
    """Generate with visible reasoning/thinking."""
    return generate(
        prompt,
        thinking_level=thinking_level,
        include_thoughts=True,
        **kwargs
    )

def generate_json(prompt: str, schema=None, **kwargs) -> dict:
    """Generate structured JSON output."""
    import json
    result = generate(
        prompt,
        response_mime_type="application/json",
        response_schema=schema,
        **kwargs
    )
    return json.loads(result["text"])
```

---

## Usage Examples

### Basic Generation

```python
from gemini_client import generate, generate_fast, generate_smart

# Simple generation (auto-fallback)
response = generate("Explain quantum computing in simple terms")
print(response["text"])
print(f"Model used: {response['model_used']}")

# Fast generation (Gemini 3 Flash preferred)
text = generate_fast("Summarize this article: ...")

# High-quality generation (Gemini 3 Pro preferred)
text = generate_smart("Write a detailed analysis of...")
```

### With Reasoning/Thinking

```python
from gemini_client import generate_with_reasoning

result = generate_with_reasoning(
    "What is 15 * 23? Show your work.",
    thinking_level="HIGH"
)

print("REASONING:", result["thoughts"])
print("ANSWER:", result["text"])
```

### Structured JSON Output

```python
from gemini_client import generate_json
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

result = generate_json(
    "Extract person info: John Smith is a 35-year-old software engineer.",
    schema=Person
)
print(result)  # {"name": "John Smith", "age": 35, "occupation": "software engineer"}
```

### With System Instruction

```python
from gemini_client import generate

response = generate(
    "What's the weather like?",
    system_instruction="You are a helpful weather assistant. Always be concise.",
    temperature=0.3
)
```

### Override Model

```python
from gemini_client import generate

# Force specific model
response = generate(
    "Complex analysis task...",
    model="gemini-3-pro-preview",
    thinking_level="HIGH",
    max_output_tokens=16384
)
```

---

## Worker/Batch Processing Pattern

For high-volume workloads with multiple workers:

```python
"""
batch_processor.py - Pattern for parallel Gemini processing
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from gemini_client import generate
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

MAX_WORKERS = 10          # Concurrent workers
BATCH_SIZE = 50           # Items per batch
RATE_LIMIT_DELAY = 0.1    # Seconds between requests per worker

# =============================================================================
# BATCH PROCESSOR
# =============================================================================

def process_item(item: dict) -> dict:
    """Process a single item with rate limiting."""
    import time
    
    try:
        result = generate(
            prompt=item["prompt"],
            model=item.get("model"),
            thinking_level=item.get("thinking_level"),
            **item.get("config", {})
        )
        time.sleep(RATE_LIMIT_DELAY)  # Basic rate limiting
        return {
            "id": item["id"],
            "success": True,
            "result": result
        }
    except Exception as e:
        logger.error(f"Failed to process item {item['id']}: {e}")
        return {
            "id": item["id"],
            "success": False,
            "error": str(e)
        }

def process_batch(items: list[dict], max_workers: int = MAX_WORKERS) -> list[dict]:
    """Process a batch of items in parallel."""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item) for item in items]
        for future in futures:
            results.append(future.result())
    
    return results

# =============================================================================
# ASYNC VERSION (for higher throughput)
# =============================================================================

async def process_item_async(item: dict, semaphore: asyncio.Semaphore) -> dict:
    """Process item with semaphore-based concurrency control."""
    async with semaphore:
        # Run sync generate in thread pool
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: generate(
                    prompt=item["prompt"],
                    model=item.get("model"),
                    **item.get("config", {})
                )
            )
            await asyncio.sleep(RATE_LIMIT_DELAY)
            return {"id": item["id"], "success": True, "result": result}
        except Exception as e:
            return {"id": item["id"], "success": False, "error": str(e)}

async def process_batch_async(items: list[dict], max_concurrent: int = MAX_WORKERS) -> list[dict]:
    """Process batch with async concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [process_item_async(item, semaphore) for item in items]
    return await asyncio.gather(*tasks)

# =============================================================================
# USAGE
# =============================================================================

if __name__ == "__main__":
    # Example batch
    items = [
        {"id": i, "prompt": f"Summarize topic {i}", "model": "gemini-3-flash-preview"}
        for i in range(100)
    ]
    
    # Sync processing
    results = process_batch(items)
    
    # Async processing
    # results = asyncio.run(process_batch_async(items))
    
    success_count = sum(1 for r in results if r["success"])
    print(f"Processed {success_count}/{len(items)} successfully")
```

---

## Environment Setup

### Required Environment Variables

```bash
# For Secret Manager access (uses default credentials)
export GOOGLE_CLOUD_PROJECT="bfai-prod"

# OR for direct API key (not recommended for production)
export GOOGLE_API_KEY="your-api-key"

# For Vertex AI mode (alternative to AI Studio)
export GOOGLE_GENAI_USE_VERTEXAI="True"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### Dependencies

```txt
# requirements.txt
google-genai>=1.54.0
google-cloud-secret-manager>=2.25.0
tenacity>=8.2.3
pydantic>=2.0.0
```

---

## Troubleshooting

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `429 Too Many Requests` | Rate limited | Exponential backoff (built-in) |
| `404 Not Found` | Model unavailable | Fallback chain handles this |
| `403 Permission Denied` | Auth issue | Check API key / IAM |
| `400 Bad Request` | Invalid params | Check config values |

### Debugging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable SDK debug logging
import google.genai
google.genai._api_client.logger.setLevel(logging.DEBUG)
```

---

## Checklist for New Projects

- [ ] Install dependencies: `pip install google-genai google-cloud-secret-manager tenacity`
- [ ] Copy `gemini_client.py` to project
- [ ] Verify Secret Manager access: `gcloud secrets versions access latest --secret=gemini-api-key --project=bfai-prod`
- [ ] Test basic generation: `python -c "from gemini_client import generate_fast; print(generate_fast('Hello'))"`
- [ ] Configure logging
- [ ] Set up monitoring for 429 errors
- [ ] Review model chain for your use case

---

*Document maintained by BrightFox AI Engineering*
