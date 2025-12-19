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
# CORE GENERATION FUNCTION
# =============================================================================


class APIError(Exception):
    """Raised for API errors."""
    pass


def _generate_content(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries: int = 5,
) -> Any:
    """
    Generate content using google-genai SDK with retry on rate limits.
    
    The SDK uses a global endpoint (generativelanguage.googleapis.com)
    and Google handles routing automatically. No region management needed.
    """
    import time
    
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff: 2, 4, 8, 16, 32 seconds
                wait_time = 2 ** (attempt + 1)
                logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
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
    seed: int = None,
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
    if seed is not None:
        config_params["seed"] = seed
    
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
    response = _generate_content(client, target_model, prompt, config)
    
    # Parse response with full llm_metadata (matches orchestrator schema)
    result = {
        "text": response.text,
        "model_used": target_model,
        "thoughts": None,
        "llm_metadata": None,
    }
    
    # Build llm_metadata matching orchestrator schema
    llm_metadata = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "thinking_tokens": 0,
        "total_tokens": 0,
        "cached_content_tokens": 0,
        "model_version": getattr(response, "model_version", target_model),
        "finish_reason": None,
        "used_fallback": False,  # We don't use fallback in eval suite
        "reasoning_effort": thinking_level.lower() if thinking_level else "low",
        "avg_logprobs": None,
        "response_id": getattr(response, "response_id", None),
    }
    
    # Extract usage metadata (token counts)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        llm_metadata["prompt_tokens"] = getattr(usage, "prompt_token_count", 0) or 0
        llm_metadata["completion_tokens"] = getattr(usage, "candidates_token_count", 0) or 0
        llm_metadata["thinking_tokens"] = getattr(usage, "thoughts_token_count", 0) if hasattr(usage, "thoughts_token_count") else 0
        llm_metadata["total_tokens"] = getattr(usage, "total_token_count", 0) or 0
        llm_metadata["cached_content_tokens"] = getattr(usage, "cached_content_token_count", 0) or 0
    
    # Extract finish_reason and avg_logprobs from candidate
    if response.candidates and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if hasattr(candidate, "finish_reason"):
            llm_metadata["finish_reason"] = str(candidate.finish_reason) if candidate.finish_reason else "STOP"
        if hasattr(candidate, "avg_logprobs"):
            llm_metadata["avg_logprobs"] = candidate.avg_logprobs
    
    result["llm_metadata"] = llm_metadata
    
    # Also keep "usage" for backward compatibility
    result["usage"] = {
        "prompt_tokens": llm_metadata["prompt_tokens"],
        "response_tokens": llm_metadata["completion_tokens"],
        "total_tokens": llm_metadata["total_tokens"],
        "thinking_tokens": llm_metadata["thinking_tokens"],
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


def generate_for_judge(
    prompt: str,
    model: str = None,
    temperature: float = None,
    reasoning_effort: str = None,
    seed: int = None,
    return_metadata: bool = False,
    **kwargs
) -> dict:
    """Generate with judge-specific config (JSON output, lower tokens).
    
    Args:
        prompt: The judge prompt
        model: Model to use (default: from config or DEFAULT_MODEL)
        temperature: Temperature (default: 0.0)
        reasoning_effort: "low" or "high" (default: "low")
        seed: Random seed for reproducibility
        return_metadata: If True, return {"judgment": ..., "tokens": ..., "metadata": ...}
        **kwargs: Additional parameters
    
    Returns:
        Parsed JSON response from judge, or dict with judgment + metadata if return_metadata=True
    """
    config = {**JUDGE_CONFIG, **kwargs}
    
    # Use provided values or defaults
    target_model = model or DEFAULT_MODEL
    target_temp = temperature if temperature is not None else config.get("temperature", 0.0)
    target_reasoning = (reasoning_effort or "low").upper()
    
    result = generate(
        prompt,
        model=target_model,
        response_mime_type=config.get("response_mime_type"),
        max_output_tokens=config.get("max_output_tokens"),
        temperature=target_temp,
        thinking_level=target_reasoning,
        seed=seed,
    )
    
    # Parse JSON, handling list responses
    text = result["text"]
    if text is None:
        error_result = {"error": "No response text", "verdict": "error"}
        if return_metadata:
            return {"judgment": error_result, "tokens": {}, "metadata": {}}
        return error_result
    
    try:
        parsed = json.loads(text)
        # If response is a list, take first element
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            error_result = {"error": f"Unexpected type: {type(parsed)}", "verdict": "error"}
            if return_metadata:
                return {"judgment": error_result, "tokens": {}, "metadata": {}}
            return error_result
        
        if return_metadata:
            # Extract tokens from llm_metadata
            llm_meta = result.get("llm_metadata", {})
            return {
                "judgment": parsed,
                "tokens": {
                    "prompt": llm_meta.get("prompt_tokens", 0),
                    "completion": llm_meta.get("completion_tokens", 0),
                    "thinking": llm_meta.get("thinking_tokens", 0),
                    "total": llm_meta.get("total_tokens", 0),
                    "cached": llm_meta.get("cached_content_tokens", 0),
                },
                "metadata": {
                    "model": result.get("model_used"),
                    "model_version": llm_meta.get("model_version"),
                    "finish_reason": llm_meta.get("finish_reason"),
                    "response_id": llm_meta.get("response_id"),
                },
            }
        return parsed
    except json.JSONDecodeError as e:
        error_result = {"error": f"JSON parse error: {e}", "verdict": "error"}
        if return_metadata:
            return {"judgment": error_result, "tokens": {}, "metadata": {}}
        return error_result


def generate_for_rag(
    prompt: str,
    model: str = None,
    temperature: float = None,
    reasoning_effort: str = None,
    max_output_tokens: int = None,
    seed: int = None,
    **kwargs
) -> dict:
    """Generate with RAG-specific config (natural language output).
    
    Args:
        prompt: The RAG prompt
        model: Model to use (default: from config or DEFAULT_MODEL)
        temperature: Temperature (default: 0.0)
        reasoning_effort: "low" or "high" (default: "low")
        max_output_tokens: Max tokens (default: 8192)
        seed: Random seed for reproducibility
        **kwargs: Additional parameters
    
    Returns:
        Full result dict with text, llm_metadata, usage, etc.
    """
    config = {**GENERATOR_CONFIG, **kwargs}
    
    # Use provided values or defaults
    target_model = model or DEFAULT_MODEL
    target_temp = temperature if temperature is not None else config.get("temperature", 0.0)
    target_reasoning = (reasoning_effort or "low").upper()
    target_max_tokens = max_output_tokens or config.get("max_output_tokens", 8192)
    
    result = generate(
        prompt,
        model=target_model,
        max_output_tokens=target_max_tokens,
        temperature=target_temp,
        thinking_level=target_reasoning,
        seed=seed,
    )
    return result


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
