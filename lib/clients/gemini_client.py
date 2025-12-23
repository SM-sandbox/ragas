"""
Gemini Client for RAG Eval Suite

Enterprise-grade multi-region Vertex AI client with AI Studio fallback.

Architecture:
    1. Primary: Vertex AI us-east1 (closest to data)
    2. Fallback: Vertex AI us-east4, us-central1, us-west1
    3. Last Resort: AI Studio (global endpoint, API key auth)

Project: bfai-prod (Vertex AI)
Fallback Project: bf-rag-eval-service (AI Studio API key)

Usage:
    from lib.clients.gemini_client import generate, generate_json, get_client
    
    # Simple generation (auto-cascades on failure)
    result = generate("What is 2+2?")
    print(result["text"])
    
    # JSON output
    data = generate_json("Extract: John is 30 years old", schema=PersonSchema)
    
    # Check which endpoint was used
    print(result["endpoint_used"])  # e.g., "vertex_ai:us-east1" or "ai_studio"
"""

import json
import logging
import time
from typing import Optional, Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum

from google import genai
from google.genai import types
from google.cloud import secretmanager

logger = logging.getLogger(__name__)


# =============================================================================
# ENDPOINT TYPES
# =============================================================================

class EndpointType(Enum):
    """Type of endpoint used for generation."""
    VERTEX_AI = "vertex_ai"
    AI_STUDIO = "ai_studio"


@dataclass
class EndpointResult:
    """Result from an endpoint attempt."""
    success: bool
    endpoint_type: EndpointType
    region: Optional[str] = None
    response: Any = None
    error: Optional[Exception] = None
    latency_ms: float = 0.0
    
    @property
    def endpoint_name(self) -> str:
        """Human-readable endpoint name."""
        if self.endpoint_type == EndpointType.VERTEX_AI:
            return f"vertex_ai:{self.region}"
        return "ai_studio"


@dataclass 
class CascadeStats:
    """Statistics for cascade execution."""
    total_attempts: int = 0
    vertex_ai_attempts: int = 0
    ai_studio_attempts: int = 0
    successful_endpoint: Optional[str] = None
    failed_endpoints: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0

# =============================================================================
# CONFIGURATION
# =============================================================================

# Vertex AI Configuration (Primary)
VERTEX_PROJECT_ID = "bfai-prod"
# Gemini 3 Flash Preview ONLY works on global endpoint (not regional)
# See: https://cloud.google.com/vertex-ai/generative-ai/docs/start/get-started-with-gemini-3
VERTEX_AI_REGIONS = ["global"]  # Only global - no regional failover needed

# AI Studio Configuration (Fallback)
AI_STUDIO_PROJECT_ID = "bf-rag-eval-service"
AI_STUDIO_SECRET_NAME = "gemini-api-key-eval"
ENABLE_AI_STUDIO_FALLBACK = True

# Model Configuration
# gemini-3-flash-preview is available on BOTH Vertex AI (global endpoint) and AI Studio
DEFAULT_MODEL = "gemini-3-flash-preview"  # Available on Vertex AI global + AI Studio
AI_STUDIO_MODEL = "gemini-3-flash-preview"  # Same model on AI Studio

# Retry configuration
MAX_RETRIES_PER_ENDPOINT = 2  # Retries within a single endpoint before cascading
RETRY_BACKOFF_BASE = 1.0  # Base seconds for exponential backoff

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
# CLIENT INITIALIZATION (Lazy caching)
# =============================================================================

# Client cache: region -> client for Vertex AI, "ai_studio" -> client for AI Studio
_client_cache: Dict[str, genai.Client] = {}
_ai_studio_api_key: Optional[str] = None


def _get_ai_studio_api_key() -> str:
    """Fetch AI Studio API key from Secret Manager (cached)."""
    global _ai_studio_api_key
    if _ai_studio_api_key is None:
        logger.info(f"Fetching AI Studio API key from Secret Manager: {AI_STUDIO_PROJECT_ID}/{AI_STUDIO_SECRET_NAME}")
        sm_client = secretmanager.SecretManagerServiceClient()
        response = sm_client.access_secret_version(
            request={"name": f"projects/{AI_STUDIO_PROJECT_ID}/secrets/{AI_STUDIO_SECRET_NAME}/versions/latest"}
        )
        _ai_studio_api_key = response.payload.data.decode("UTF-8")
        logger.info("AI Studio API key retrieved successfully")
    return _ai_studio_api_key


def get_vertex_client(region: str) -> genai.Client:
    """Get or create a Vertex AI client for a specific region (lazy cached)."""
    cache_key = f"vertex:{region}"
    if cache_key not in _client_cache:
        logger.info(f"Initializing Vertex AI client: project={VERTEX_PROJECT_ID}, region={region}")
        _client_cache[cache_key] = genai.Client(
            vertexai=True,
            project=VERTEX_PROJECT_ID,
            location=region,
        )
    return _client_cache[cache_key]


def get_ai_studio_client() -> genai.Client:
    """Get or create the AI Studio client (lazy cached)."""
    cache_key = "ai_studio"
    if cache_key not in _client_cache:
        logger.info("Initializing AI Studio client")
        _client_cache[cache_key] = genai.Client(api_key=_get_ai_studio_api_key())
    return _client_cache[cache_key]


def get_client() -> genai.Client:
    """Get the primary client (first Vertex AI region). For backward compatibility."""
    return get_vertex_client(VERTEX_AI_REGIONS[0])


def reset_client():
    """Reset all cached clients (useful for testing)."""
    global _client_cache, _ai_studio_api_key
    _client_cache = {}
    _ai_studio_api_key = None
    logger.info("All clients reset")


# =============================================================================
# ERROR CLASSIFICATION
# =============================================================================


class APIError(Exception):
    """Raised for API errors."""
    pass


class CascadeExhaustedError(APIError):
    """Raised when all endpoints in the cascade have failed."""
    def __init__(self, message: str, stats: CascadeStats):
        super().__init__(message)
        self.stats = stats


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger cascade to next endpoint.
    
    Retryable (cascade to next region):
        - 429 (Rate Limited)
        - 503 (Service Unavailable)
        - Timeout
        - Quota Exceeded
        - 404 (Model not found - might be in another region)
    
    Non-retryable (fail fast):
        - 400 (Bad Request) - your request is wrong
        - 401/403 (Auth) - fix your credentials
        - Other client errors
    """
    error_str = str(error).lower()
    
    # Retryable errors - cascade to next endpoint
    retryable_patterns = [
        "429",
        "resource_exhausted",
        "quota",
        "503",
        "service unavailable",
        "unavailable",
        "timeout",
        "timed out",
        "deadline exceeded",
        "504",
        "gateway timeout",
        "404",  # Model might not be in this region
        "not found",
    ]
    
    for pattern in retryable_patterns:
        if pattern in error_str:
            return True
    
    # Non-retryable errors - fail fast
    non_retryable_patterns = [
        "400",
        "bad request",
        "invalid",
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "permission denied",
    ]
    
    for pattern in non_retryable_patterns:
        if pattern in error_str:
            return False
    
    # Default: retry on unknown errors (be resilient)
    return True


# =============================================================================
# CORE GENERATION WITH CASCADE
# =============================================================================


def _generate_content_single(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries: int = MAX_RETRIES_PER_ENDPOINT,
) -> Any:
    """
    Generate content with retries on a single endpoint.
    
    This handles transient errors within one endpoint before we cascade.
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Only retry rate limits within the same endpoint
            is_rate_limit = "429" in error_str or "resource_exhausted" in error_str
            
            if is_rate_limit and attempt < max_retries - 1:
                wait_time = RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(f"Rate limited, waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            # Not a rate limit or exhausted retries - let cascade handle it
            raise
    
    raise last_error


def _generate_content_with_cascade(
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
) -> tuple[Any, str, CascadeStats]:
    """
    Generate content with multi-region Vertex AI cascade + AI Studio fallback.
    
    Cascade order:
        1. Vertex AI us-east1 (primary)
        2. Vertex AI us-east4 (east coast backup)
        3. Vertex AI us-central1 (central hub)
        4. Vertex AI us-west1 (west coast backup)
        5. AI Studio (global, API key auth) - last resort
    
    Returns:
        tuple of (response, endpoint_name, cascade_stats)
    """
    stats = CascadeStats()
    last_error = None
    
    # Phase 1: Vertex AI regional cascade
    for region in VERTEX_AI_REGIONS:
        stats.total_attempts += 1
        stats.vertex_ai_attempts += 1
        endpoint_name = f"vertex_ai:{region}"
        
        start_time = time.time()
        try:
            logger.debug(f"Trying Vertex AI region: {region}")
            client = get_vertex_client(region)
            response = _generate_content_single(client, model, contents, config)
            
            latency_ms = (time.time() - start_time) * 1000
            stats.total_latency_ms += latency_ms
            stats.successful_endpoint = endpoint_name
            
            logger.info(f"Success on {endpoint_name} ({latency_ms:.0f}ms)")
            return response, endpoint_name, stats
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            stats.total_latency_ms += latency_ms
            stats.failed_endpoints.append(endpoint_name)
            last_error = e
            
            if is_retryable_error(e):
                logger.warning(f"Vertex AI {region} failed ({latency_ms:.0f}ms): {e}")
                continue  # Try next region
            else:
                # Non-retryable error - fail fast
                logger.error(f"Non-retryable error on {endpoint_name}: {e}")
                raise APIError(f"Non-retryable error: {e}")
    
    # Phase 2: AI Studio fallback (if enabled)
    if ENABLE_AI_STUDIO_FALLBACK:
        stats.total_attempts += 1
        stats.ai_studio_attempts += 1
        endpoint_name = "ai_studio"
        
        # Use AI Studio model (same as Vertex AI for gemini-3-flash-preview)
        ai_studio_model = AI_STUDIO_MODEL
        
        start_time = time.time()
        try:
            logger.warning("All Vertex AI regions failed, falling back to AI Studio")
            client = get_ai_studio_client()
            response = _generate_content_single(client, ai_studio_model, contents, config)
            
            latency_ms = (time.time() - start_time) * 1000
            stats.total_latency_ms += latency_ms
            stats.successful_endpoint = endpoint_name
            
            logger.info(f"Success on AI Studio fallback ({latency_ms:.0f}ms)")
            return response, endpoint_name, stats
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            stats.total_latency_ms += latency_ms
            stats.failed_endpoints.append(endpoint_name)
            last_error = e
            logger.error(f"AI Studio fallback also failed: {e}")
    
    # All endpoints exhausted
    error_msg = f"All endpoints exhausted after {stats.total_attempts} attempts. Last error: {last_error}"
    logger.error(error_msg)
    raise CascadeExhaustedError(error_msg, stats)


def _generate_content(
    client: genai.Client,
    model: str,
    contents: str,
    config: types.GenerateContentConfig,
    max_retries: int = 5,
) -> Any:
    """
    Legacy wrapper for backward compatibility.
    Now uses cascade internally.
    """
    response, endpoint_name, stats = _generate_content_with_cascade(model, contents, config)
    return response


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
        dict with 'text', 'model_used', 'endpoint_used', 'cascade_stats', 'thoughts' (if requested)
    """
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
    
    # Add thinking config if specified AND model supports it
    # Only gemini-3-* and gemini-2.0-flash-thinking-* models support thinking_level
    # gemini-2.5-flash does NOT support thinking_level
    target_model = model or DEFAULT_MODEL
    supports_thinking = "gemini-3" in target_model or "thinking" in target_model
    
    if thinking_level and supports_thinking:
        level = types.ThinkingLevel.HIGH if thinking_level.upper() == "HIGH" else types.ThinkingLevel.LOW
        config_params["thinking_config"] = types.ThinkingConfig(
            thinking_level=level,
            include_thoughts=include_thoughts,
        )
    elif thinking_level and not supports_thinking:
        logger.debug(f"Model {target_model} does not support thinking_level, skipping")
    
    config_params.update(kwargs)
    config = types.GenerateContentConfig(**config_params)
    
    # Use specified model or default
    target_model = model or DEFAULT_MODEL
    
    logger.debug(f"Generating with {target_model}, thinking={thinking_level}")
    
    # Use cascade for generation
    response, endpoint_used, cascade_stats = _generate_content_with_cascade(target_model, prompt, config)
    
    # Parse response with full llm_metadata (matches orchestrator schema)
    result = {
        "text": response.text,
        "model_used": target_model,
        "endpoint_used": endpoint_used,
        "cascade_stats": {
            "total_attempts": cascade_stats.total_attempts,
            "vertex_ai_attempts": cascade_stats.vertex_ai_attempts,
            "ai_studio_attempts": cascade_stats.ai_studio_attempts,
            "failed_endpoints": cascade_stats.failed_endpoints,
            "total_latency_ms": cascade_stats.total_latency_ms,
        },
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
        "used_fallback": endpoint_used == "ai_studio",  # True if AI Studio fallback was used
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
    max_retries: int = 3,
    **kwargs
) -> dict:
    """Generate with judge-specific config (JSON output, lower tokens).
    
    Includes retry logic for JSON parse errors since LLMs occasionally
    return malformed JSON.
    
    Args:
        prompt: The judge prompt
        model: Model to use (default: from config or DEFAULT_MODEL)
        temperature: Temperature (default: 0.0)
        reasoning_effort: "low" or "high" (default: "low")
        seed: Random seed for reproducibility
        return_metadata: If True, return {"judgment": ..., "tokens": ..., "metadata": ...}
        max_retries: Max retries for JSON parse errors (default: 3)
        **kwargs: Additional parameters
    
    Returns:
        Parsed JSON response from judge, or dict with judgment + metadata if return_metadata=True
    """
    config = {**JUDGE_CONFIG, **kwargs}
    
    # Use provided values or defaults
    target_model = model or DEFAULT_MODEL
    target_temp = temperature if temperature is not None else config.get("temperature", 0.0)
    
    # Only pass thinking_level for models that support it (gemini-3-* models)
    supports_thinking = "gemini-3" in target_model or "thinking" in target_model
    target_reasoning = (reasoning_effort or "low").upper() if supports_thinking else None
    
    last_error = None
    for attempt in range(max_retries):
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
            last_error = "No response text"
            logger.warning(f"Judge attempt {attempt + 1}/{max_retries}: No response text, retrying...")
            continue
        
        try:
            parsed = json.loads(text)
            # If response is a list, take first element
            if isinstance(parsed, list) and len(parsed) > 0:
                parsed = parsed[0]
            if not isinstance(parsed, dict):
                last_error = f"Unexpected type: {type(parsed)}"
                logger.warning(f"Judge attempt {attempt + 1}/{max_retries}: {last_error}, retrying...")
                continue
            
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
                        "judge_retries": attempt,
                    },
                }
            return parsed
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            logger.warning(f"Judge attempt {attempt + 1}/{max_retries}: {last_error}, retrying...")
            continue
    
    # All retries exhausted
    logger.error(f"Judge failed after {max_retries} attempts: {last_error}")
    error_result = {"error": last_error, "verdict": "error"}
    if return_metadata:
        return {"judgment": error_result, "tokens": {}, "metadata": {"judge_retries": max_retries}}
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
    target_max_tokens = max_output_tokens or config.get("max_output_tokens", 8192)
    
    # Only pass thinking_level for models that support it (gemini-3-* models)
    supports_thinking = "gemini-3" in target_model or "thinking" in target_model
    target_reasoning = (reasoning_effort or "low").upper() if supports_thinking else None
    
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
    "vertex_project": VERTEX_PROJECT_ID,
    "vertex_regions": VERTEX_AI_REGIONS,
    "ai_studio_project": AI_STUDIO_PROJECT_ID,
    "ai_studio_fallback_enabled": ENABLE_AI_STUDIO_FALLBACK,
}


def get_model_info() -> dict:
    """Return model configuration info."""
    return MODEL_INFO.copy()


# =============================================================================
# HEALTH CHECK
# =============================================================================


def health_check() -> dict:
    """
    Verify client is working and report cascade status.
    
    Returns:
        dict with status, endpoint info, and cascade details
    """
    try:
        result = generate("Say 'OK' if you can hear me.", max_output_tokens=10)
        return {
            "status": "healthy",
            "model": result["model_used"],
            "endpoint_used": result.get("endpoint_used", "unknown"),
            "response": result["text"],
            "cascade_stats": result.get("cascade_stats", {}),
            "used_fallback": result.get("llm_metadata", {}).get("used_fallback", False),
        }
    except CascadeExhaustedError as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "cascade_stats": {
                "total_attempts": e.stats.total_attempts,
                "failed_endpoints": e.stats.failed_endpoints,
            },
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def health_check_all_endpoints() -> dict:
    """
    Test each endpoint individually and report status.
    
    Useful for diagnosing which endpoints are healthy.
    
    Returns:
        dict with status of each endpoint
    """
    results = {
        "overall_status": "healthy",
        "vertex_ai_regions": {},
        "ai_studio": None,
        "summary": {
            "healthy_count": 0,
            "unhealthy_count": 0,
        }
    }
    
    # Test each Vertex AI region
    for region in VERTEX_AI_REGIONS:
        try:
            client = get_vertex_client(region)
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=10,
            )
            start = time.time()
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents="Say 'OK'",
                config=config,
            )
            latency_ms = (time.time() - start) * 1000
            results["vertex_ai_regions"][region] = {
                "status": "healthy",
                "latency_ms": round(latency_ms, 1),
                "response": response.text[:50] if response.text else None,
            }
            results["summary"]["healthy_count"] += 1
        except Exception as e:
            results["vertex_ai_regions"][region] = {
                "status": "unhealthy",
                "error": str(e)[:100],
            }
            results["summary"]["unhealthy_count"] += 1
    
    # Test AI Studio
    if ENABLE_AI_STUDIO_FALLBACK:
        try:
            client = get_ai_studio_client()
            config = types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=10,
            )
            start = time.time()
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents="Say 'OK'",
                config=config,
            )
            latency_ms = (time.time() - start) * 1000
            results["ai_studio"] = {
                "status": "healthy",
                "latency_ms": round(latency_ms, 1),
                "response": response.text[:50] if response.text else None,
            }
            results["summary"]["healthy_count"] += 1
        except Exception as e:
            results["ai_studio"] = {
                "status": "unhealthy",
                "error": str(e)[:100],
            }
            results["summary"]["unhealthy_count"] += 1
    
    # Set overall status
    if results["summary"]["unhealthy_count"] > 0:
        if results["summary"]["healthy_count"] == 0:
            results["overall_status"] = "unhealthy"
        else:
            results["overall_status"] = "degraded"
    
    return results


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
