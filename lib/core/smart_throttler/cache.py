"""
Response Cache for Smart Throttler.

Optional local cache for eval reruns. Cache hits bypass throttling entirely.
Keyed by model, step, normalized prompt, and output-affecting parameters.
"""

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class CacheConfig:
    """Configuration for response cache."""
    enabled: bool = False
    ttl_s: int = 86400  # 24 hours
    max_entries: int = 50000


@dataclass
class CacheEntry:
    """A cached response entry."""
    key: str
    response: Any
    created_at: float
    expires_at: float
    hits: int = 0
    
    # Metadata for debugging
    model: str = ""
    step_name: str = ""
    prompt_hash: str = ""


@dataclass
class CacheStats:
    """Statistics for the cache."""
    enabled: bool
    entries: int
    max_entries: int
    hits: int
    misses: int
    hit_rate: float
    evictions: int
    expirations: int


class ResponseCache:
    """
    LRU cache for API responses with TTL.
    
    Designed for eval reruns where the same prompts are sent repeatedly.
    Cache hits bypass throttling entirely, saving API quota.
    
    Cache key includes:
    - Model name
    - Step name
    - Normalized prompt hash
    - Parameters that affect output (temperature, etc.)
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize the response cache.
        
        Args:
            config: Cache configuration
        """
        self._config = config or CacheConfig()
        self._lock = threading.Lock()
        
        # LRU cache using OrderedDict
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0
    
    @property
    def enabled(self) -> bool:
        """Check if cache is enabled."""
        return self._config.enabled
    
    def _make_key(
        self,
        model: str,
        step_name: str,
        prompt: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Create a cache key from request parameters.
        
        Args:
            model: Model name
            step_name: Step name
            prompt: Full prompt text
            temperature: Temperature setting
            **kwargs: Other parameters that affect output
            
        Returns:
            Cache key string
        """
        # Normalize prompt (strip whitespace, lowercase for comparison)
        normalized_prompt = prompt.strip()
        
        # Create hash of prompt
        prompt_hash = hashlib.sha256(normalized_prompt.encode()).hexdigest()[:16]
        
        # Include output-affecting parameters
        params = {
            "model": model,
            "step": step_name,
            "temp": temperature,
            **{k: v for k, v in kwargs.items() if v is not None},
        }
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:8]
        
        return f"{model}:{step_name}:{prompt_hash}:{params_hash}"
    
    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache is full."""
        while len(self._cache) >= self._config.max_entries:
            # Remove oldest (first) entry
            self._cache.popitem(last=False)
            self._evictions += 1
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if entry.expires_at < now
        ]
        for key in expired_keys:
            del self._cache[key]
            self._expirations += 1
    
    def get(
        self,
        model: str,
        step_name: str,
        prompt: str,
        temperature: float = 0.0,
        **kwargs,
    ) -> Tuple[bool, Any]:
        """
        Get a cached response.
        
        Args:
            model: Model name
            step_name: Step name
            prompt: Full prompt text
            temperature: Temperature setting
            **kwargs: Other parameters
            
        Returns:
            Tuple of (hit, response) - response is None on miss
        """
        if not self._config.enabled:
            return False, None
        
        key = self._make_key(model, step_name, prompt, temperature, **kwargs)
        
        with self._lock:
            # Cleanup expired entries periodically
            if len(self._cache) > 0 and self._misses % 100 == 0:
                self._cleanup_expired()
            
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return False, None
            
            # Check expiration
            if entry.expires_at < time.time():
                del self._cache[key]
                self._expirations += 1
                self._misses += 1
                return False, None
            
            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            
            return True, entry.response
    
    def put(
        self,
        model: str,
        step_name: str,
        prompt: str,
        response: Any,
        temperature: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Store a response in the cache.
        
        Args:
            model: Model name
            step_name: Step name
            prompt: Full prompt text
            response: Response to cache
            temperature: Temperature setting
            **kwargs: Other parameters
        """
        if not self._config.enabled:
            return
        
        key = self._make_key(model, step_name, prompt, temperature, **kwargs)
        prompt_hash = hashlib.sha256(prompt.strip().encode()).hexdigest()[:16]
        
        now = time.time()
        entry = CacheEntry(
            key=key,
            response=response,
            created_at=now,
            expires_at=now + self._config.ttl_s,
            model=model,
            step_name=step_name,
            prompt_hash=prompt_hash,
        )
        
        with self._lock:
            # Evict if needed
            self._evict_if_needed()
            
            # Store entry
            self._cache[key] = entry
            self._cache.move_to_end(key)
    
    def invalidate(
        self,
        model: Optional[str] = None,
        step_name: Optional[str] = None,
    ) -> int:
        """
        Invalidate cache entries.
        
        Args:
            model: Invalidate entries for this model (None = all)
            step_name: Invalidate entries for this step (None = all)
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if model is None and step_name is None:
                count = len(self._cache)
                self._cache.clear()
                return count
            
            keys_to_remove = []
            for key, entry in self._cache.items():
                if model and entry.model != model:
                    continue
                if step_name and entry.step_name != step_name:
                    continue
                keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
            
            return len(keys_to_remove)
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0
            
            return CacheStats(
                enabled=self._config.enabled,
                entries=len(self._cache),
                max_entries=self._config.max_entries,
                hits=self._hits,
                misses=self._misses,
                hit_rate=hit_rate,
                evictions=self._evictions,
                expirations=self._expirations,
            )
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
    
    def reset_stats(self) -> None:
        """Reset statistics without clearing cache."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
    
    def update_config(self, config: CacheConfig) -> None:
        """Update cache configuration."""
        with self._lock:
            self._config = config
            # Evict if new max is smaller
            while len(self._cache) > self._config.max_entries:
                self._cache.popitem(last=False)
                self._evictions += 1
