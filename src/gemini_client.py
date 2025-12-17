"""
Gemini Client for RAG Eval Suite

Uses google-genai SDK with API key from Secret Manager.
Model: gemini-3-flash-preview (no fallback)
Project: bf-rag-eval-service

Usage:
    from src.gemini_client import generate, generate_json, get_client
    
    # Simple generation
    result = generate("What is 2+2?")
    print(result["text"])
    
    # JSON output
    data = generate_json("Extract: John is 30 years old", schema=PersonSchema)
"""

import json
import logging
from typing import Optional, Any

from google import genai
from google.genai import types
from google.cloud import secretmanager
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ID = "bf-rag-eval-service"
SECRET_NAME = "gemini-api-key-eval"

# Model - Gemini 3 Flash Preview ONLY (no fallback)
DEFAULT_MODEL = "gemini-3-flash-preview"

# Default generation config
DEFAULT_CONFIG = {
    "temperature": 0.0,  # Deterministic for eval
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# Judge-specific config (structured JSON output)
JUDGE_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 2048,
    "response_mime_type": "application/json",
}

# Generator-specific config
GENERATOR_CONFIG = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
}

# =============================================================================
# CLIENT INITIALIZATION
# =============================================================================

_client: Optional[genai.Client] = None
_api_key: Optional[str] = None


def _get_api_key() -> str:
    """Fetch API key from Secret Manager (cached)."""
    global _api_key
    if _api_key is None:
        logger.info(f"Fetching API key from Secret Manager: {PROJECT_ID}/{SECRET_NAME}")
        sm_client = secretmanager.SecretManagerServiceClient()
        response = sm_client.access_secret_version(
            request={"name": f"projects/{PROJECT_ID}/secrets/{SECRET_NAME}/versions/latest"}
        )
        _api_key = response.payload.data.decode("UTF-8")
        logger.info("API key retrieved successfully")
    return _api_key


def get_client() -> genai.Client:
    """Get or create the Gemini client (singleton)."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=_get_api_key())
        logger.info(f"Gemini client initialized with model: {DEFAULT_MODEL}")
    return _client


def reset_client():
    """Reset the client (useful for testing)."""
    global _client, _api_key
    _client = None
    _api_key = None


# =============================================================================
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# =============================================================================


class RateLimitError(Exception):
    """Raised when rate limited (429)."""
    pass


class APIError(Exception):
    """Raised for other API errors."""
    pass


def _is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error."""
    error_str = str(exception).lower()
    return "429" in error_str or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str


# =============================================================================
# CORE GENERATION FUNCTION
# =============================================================================


@retry(
    wait=wait_exponential(multiplier=1, min=1, max=60),
    stop=stop_after_attempt(10),  # More retries, we're committed
    retry=retry_if_exception_type(RateLimitError),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)
def _generate_with_retry(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
) -> Any:
    """Generate content with retry logic for rate limits."""
    try:
        return client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
    except Exception as e:
        if _is_rate_limit_error(e):
            logger.warning(f"Rate limited: {e}")
            raise RateLimitError(str(e))
        logger.error(f"API error: {e}")
        raise APIError(str(e))


def generate(
    prompt: str,
    model: str = None,
    thinking_level: str = None,
    include_thoughts: bool = False,
    temperature: float = None,
    max_output_tokens: int = None,
    system_instruction: str = None,
    response_mime_type: str = None,
    response_schema: Any = None,
    **kwargs,
) -> dict:
    """
    Generate content with Gemini 3 Flash Preview.
    
    Args:
        prompt: The input prompt
        model: Model ID (default: gemini-3-flash-preview)
        thinking_level: "LOW" or "HIGH" (optional)
        include_thoughts: Return reasoning in response
        temperature: Override default temperature
        max_output_tokens: Override default max tokens
        system_instruction: System prompt
        response_mime_type: e.g., "application/json"
        response_schema: Pydantic model or dict for structured output
        **kwargs: Additional config parameters
    
    Returns:
        dict with 'text', 'model_used', 'thoughts' (if requested)
    """
    client = get_client()
    
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
            include_thoughts=include_thoughts,
        )
    
    config_params.update(kwargs)
    config = types.GenerateContentConfig(**config_params)
    
    # Use specified model or default
    target_model = model or DEFAULT_MODEL
    
    logger.debug(f"Generating with {target_model}, thinking={thinking_level}")
    response = _generate_with_retry(client, target_model, prompt, config)
    
    # Parse response
    result = {
        "text": response.text,
        "model_used": target_model,
        "thoughts": None,
    }
    
    # Extract thoughts if requested
    if include_thoughts and response.candidates:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "thought") and part.thought:
                result["thoughts"] = part.text
                break
    
    return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def generate_fast(prompt: str, **kwargs) -> str:
    """Quick generation with Gemini 3 Flash."""
    result = generate(prompt, **kwargs)
    return result["text"]


def generate_with_reasoning(prompt: str, thinking_level: str = "LOW", **kwargs) -> dict:
    """Generate with visible reasoning/thinking."""
    return generate(
        prompt,
        thinking_level=thinking_level,
        include_thoughts=True,
        **kwargs,
    )


def generate_json(prompt: str, schema: Any = None, **kwargs) -> dict:
    """Generate structured JSON output."""
    result = generate(
        prompt,
        response_mime_type="application/json",
        response_schema=schema,
        **kwargs,
    )
    return json.loads(result["text"])


def generate_for_judge(prompt: str, **kwargs) -> dict:
    """Generate with judge-specific config (JSON output, lower tokens)."""
    config = {**JUDGE_CONFIG, **kwargs}
    result = generate(
        prompt,
        response_mime_type=config.get("response_mime_type"),
        max_output_tokens=config.get("max_output_tokens"),
        temperature=config.get("temperature"),
        thinking_level="LOW",  # Fast for judge
    )
    return json.loads(result["text"])


def generate_for_rag(prompt: str, **kwargs) -> str:
    """Generate with RAG-specific config (natural language output)."""
    config = {**GENERATOR_CONFIG, **kwargs}
    result = generate(
        prompt,
        max_output_tokens=config.get("max_output_tokens"),
        temperature=config.get("temperature"),
        thinking_level="LOW",  # Fast for generation
    )
    return result["text"]


# =============================================================================
# MODEL INFO
# =============================================================================

MODEL_INFO = {
    "model_id": DEFAULT_MODEL,
    "display_name": "Gemini 3 Flash Preview",
    "status": "PUBLIC_PREVIEW",
    "input_token_limit": 1_048_576,
    "output_token_limit": 65_536,
    "thinking_levels": ["LOW", "HIGH"],
    "project": PROJECT_ID,
    "secret": SECRET_NAME,
}


def get_model_info() -> dict:
    """Return model configuration info."""
    return MODEL_INFO.copy()


# =============================================================================
# HEALTH CHECK
# =============================================================================


def health_check() -> dict:
    """Verify client is working."""
    try:
        result = generate("Say 'OK' if you can hear me.", max_output_tokens=10)
        return {
            "status": "healthy",
            "model": result["model_used"],
            "response": result["text"],
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    print("Testing Gemini client...")
    
    result = health_check()
    print(f"Health check: {result}")
    
    if result["status"] == "healthy":
        print("\nTesting JSON output...")
        json_result = generate_json("Return a JSON object with name='test' and value=42")
        print(f"JSON result: {json_result}")
        
        print("\nTesting reasoning...")
        reasoning_result = generate_with_reasoning("What is 15 * 23?", thinking_level="LOW")
        print(f"Answer: {reasoning_result['text']}")
        if reasoning_result["thoughts"]:
            print(f"Thoughts: {reasoning_result['thoughts']}")
