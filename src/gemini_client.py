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
import time as time_module

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ID = "bf-rag-eval-service"
SECRET_NAME = "gemini-api-key-eval"

# Model - Gemini 3 Flash Preview ONLY (no fallback)
DEFAULT_MODEL = "gemini-3-flash-preview"

# US Regions for failover (cycle through on capacity errors)
US_REGIONS = [
    "us-central1",
    "us-east1", 
    "us-east4",
    "us-west1",
    "us-west4",
    "us-south1",
]

# Current region index (for round-robin failover)
_current_region_idx = 0

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


class CapacityError(Exception):
    """Raised when region is at capacity (503, overloaded)."""
    pass


class APIError(Exception):
    """Raised for other API errors."""
    pass


def _is_rate_limit_error(exception: Exception) -> bool:
    """Check if exception is a rate limit error."""
    error_str = str(exception).lower()
    return "429" in error_str or "rate" in error_str or "quota" in error_str or "resource_exhausted" in error_str


def _is_capacity_error(exception: Exception) -> bool:
    """Check if exception is a capacity/overload error."""
    error_str = str(exception).lower()
    return ("503" in error_str or "capacity" in error_str or 
            "overload" in error_str or "unavailable" in error_str or
            "timeout" in error_str)


def _get_next_region() -> str:
    """Get next region in round-robin fashion."""
    global _current_region_idx
    region = US_REGIONS[_current_region_idx]
    _current_region_idx = (_current_region_idx + 1) % len(US_REGIONS)
    return region


def _get_current_region() -> str:
    """Get current region without advancing."""
    return US_REGIONS[_current_region_idx]


# =============================================================================
# CORE GENERATION FUNCTION
# =============================================================================


def _generate_with_region_failover(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
) -> Any:
    """
    Generate content with region failover on capacity errors.
    
    Strategy:
    - Try current region with 2 retries (exponential backoff)
    - On capacity error after 2 retries, hop to next region
    - Cycle through all US regions before giving up
    """
    regions_tried = 0
    max_regions = len(US_REGIONS)
    last_error = None
    
    while regions_tried < max_regions:
        current_region = _get_current_region()
        retries_in_region = 0
        max_retries_per_region = 2
        
        while retries_in_region < max_retries_per_region:
            try:
                logger.debug(f"Attempting generation in region {current_region} (attempt {retries_in_region + 1})")
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                last_error = e
                
                if _is_rate_limit_error(e):
                    # Rate limit - exponential backoff within region
                    wait_time = (2 ** retries_in_region) * 1  # 1s, 2s
                    logger.warning(f"Rate limited in {current_region}, waiting {wait_time}s (attempt {retries_in_region + 1}/{max_retries_per_region})")
                    time_module.sleep(wait_time)
                    retries_in_region += 1
                    
                elif _is_capacity_error(e):
                    # Capacity error - hop to next region immediately
                    logger.warning(f"Capacity error in {current_region}: {e}. Hopping to next region.")
                    _get_next_region()  # Advance to next region
                    regions_tried += 1
                    break  # Exit inner loop, try next region
                    
                else:
                    # Other error - raise immediately
                    logger.error(f"API error in {current_region}: {e}")
                    raise APIError(str(e))
        
        # Exhausted retries in this region, move to next
        if retries_in_region >= max_retries_per_region:
            logger.warning(f"Exhausted {max_retries_per_region} retries in {current_region}, hopping to next region")
            _get_next_region()
            regions_tried += 1
    
    # All regions exhausted
    logger.error(f"All {max_regions} regions exhausted. Last error: {last_error}")
    raise APIError(f"All regions exhausted. Last error: {last_error}")


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
    response = _generate_with_region_failover(client, target_model, prompt, config)
    
    # Parse response with token counts
    result = {
        "text": response.text,
        "model_used": target_model,
        "thoughts": None,
        "usage": None,
    }
    
    # Extract usage metadata (token counts)
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        usage = response.usage_metadata
        result["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_token_count", 0),
            "response_tokens": getattr(usage, "candidates_token_count", 0),
            "total_tokens": getattr(usage, "total_token_count", 0),
            "thinking_tokens": getattr(usage, "thoughts_token_count", 0) if hasattr(usage, "thoughts_token_count") else 0,
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
    
    # Parse JSON, handling list responses
    text = result["text"]
    if text is None:
        return {"error": "No response text", "verdict": "error"}
    
    try:
        parsed = json.loads(text)
        # If response is a list, take first element
        if isinstance(parsed, list) and len(parsed) > 0:
            parsed = parsed[0]
        if not isinstance(parsed, dict):
            return {"error": f"Unexpected type: {type(parsed)}", "verdict": "error"}
        return parsed
    except json.JSONDecodeError as e:
        return {"error": f"JSON parse error: {e}", "verdict": "error"}


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
