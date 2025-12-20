"""
GCP Quota Checker for Gemini API.

Fetches and caches Gemini API quotas from GCP.

Usage:
    from lib.core.smart_throttler import get_quotas, get_tier, load_cached_quotas
    
    # Fetch fresh quotas from GCP
    quotas = get_quotas(project="bfai-prod")
    
    # Get tier based on quotas
    tier = get_tier(project="bfai-prod")
    
    # Load from cache (fast)
    cached = load_cached_quotas()
"""

import subprocess
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Cache file location
QUOTA_CACHE_FILE = Path(__file__).parent.parent.parent / "config" / "gemini_quotas.json"


def get_current_project() -> str:
    """Get the currently active GCP project from gcloud config."""
    cmd = ["gcloud", "config", "get-value", "project"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get current project: {result.stderr}")
    return result.stdout.strip()


def get_quotas(project: Optional[str] = None, save_cache: bool = True) -> Dict[str, Any]:
    """
    Fetch Gemini API quotas from GCP.
    
    Args:
        project: GCP project ID (defaults to current gcloud project)
        save_cache: Whether to save results to cache file
        
    Returns:
        Dict with quota info per model
    """
    if project is None:
        project = get_current_project()
    
    cmd = [
        "gcloud", "alpha", "services", "quota", "list",
        "--service=generativelanguage.googleapis.com",
        f"--consumer=projects/{project}",
        "--format=json"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to fetch quotas: {result.stderr}")
    
    data = json.loads(result.stdout)
    
    # Models we care about
    target_models = [
        "gemini-3-flash",
        "gemini-3-pro", 
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ]
    
    results = {}
    
    for q in data:
        metric = q.get("metric", "")
        if "generate_content" not in metric:
            continue
            
        limits = q.get("consumerQuotaLimits", [])
        for lim in limits:
            unit = lim.get("unit", "")
            buckets = lim.get("quotaBuckets", [])
            
            for b in buckets:
                eff = b.get("effectiveLimit")
                dims = b.get("dimensions", {})
                model = dims.get("model", "")
                
                if not model or not eff:
                    continue
                
                # Check if this is a model we care about
                for target in target_models:
                    if model == target or model.startswith(target + "-"):
                        if model not in results:
                            results[model] = {"model": model, "rpm": 0, "tpm": 0}
                        if "request" in metric and "/min" in unit:
                            results[model]["rpm"] = max(results[model]["rpm"], int(eff))
                        elif "token" in metric and "/min" in unit:
                            results[model]["tpm"] = max(results[model]["tpm"], int(eff))
    
    # Save to cache if requested
    if save_cache:
        cache_data = {
            "project": project,
            "tier": _determine_tier(results),
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "quotas": results,
        }
        QUOTA_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(QUOTA_CACHE_FILE, "w") as f:
            json.dump(cache_data, f, indent=2)
    
    return results


def _determine_tier(quotas: Dict[str, Any]) -> str:
    """Determine the pricing tier based on quota levels."""
    # Check flash model RPM to determine tier
    flash_rpm = 0
    for model, vals in quotas.items():
        if "flash" in model and "exp" not in model and "lite" not in model:
            flash_rpm = max(flash_rpm, vals.get("rpm", 0))
    
    if flash_rpm >= 20000:
        return "Paid Tier 3 (Scale)"
    elif flash_rpm >= 2000:
        return "Paid Tier 2 (Pay-as-you-go)"
    elif flash_rpm >= 100:
        return "Paid Tier 1 (Starter)"
    else:
        return "Free Tier"


def get_tier(project: Optional[str] = None) -> str:
    """
    Determine the pricing tier based on quota levels.
    
    Tier 1: Free tier (low limits)
    Tier 2: Pay-as-you-go (medium limits)
    Tier 3: Scale tier (high limits, 20K+ RPM for flash)
    
    Returns:
        Tier name string
    """
    quotas = get_quotas(project)
    return _determine_tier(quotas)


def load_cached_quotas() -> Optional[Dict[str, Any]]:
    """
    Load quotas from cache file.
    
    Returns:
        Cached quota data or None if cache doesn't exist
    """
    if not QUOTA_CACHE_FILE.exists():
        return None
    
    try:
        with open(QUOTA_CACHE_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def refresh_quotas(project: Optional[str] = None) -> Dict[str, Any]:
    """
    Force refresh quotas from GCP and update cache.
    
    Args:
        project: GCP project ID (defaults to current gcloud project)
        
    Returns:
        Fresh quota data
    """
    return get_quotas(project, save_cache=True)


def get_quota_for_model(model: str, project: Optional[str] = None) -> Optional[Dict[str, int]]:
    """
    Get quota for a specific model.
    
    Args:
        model: Model name (e.g., "gemini-2.5-flash")
        project: GCP project ID
        
    Returns:
        Dict with rpm and tpm, or None if not found
    """
    # Try cache first
    cached = load_cached_quotas()
    if cached:
        quotas = cached.get("quotas", {})
        if model in quotas:
            return {"rpm": quotas[model]["rpm"], "tpm": quotas[model]["tpm"]}
    
    # Fetch fresh if not in cache
    quotas = get_quotas(project)
    if model in quotas:
        return {"rpm": quotas[model]["rpm"], "tpm": quotas[model]["tpm"]}
    
    return None
