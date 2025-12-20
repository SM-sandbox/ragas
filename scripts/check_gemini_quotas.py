#!/usr/bin/env python3
"""
Check Gemini API quotas and tier for a GCP project.

Usage:
    python scripts/check_gemini_quotas.py
    python scripts/check_gemini_quotas.py --project bfai-prod
    python scripts/check_gemini_quotas.py --json
"""

import subprocess
import json
import argparse
from typing import Dict, Any


def get_current_project() -> str:
    """Get the currently active GCP project from gcloud config."""
    cmd = ["gcloud", "config", "get-value", "project"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get current project: {result.stderr}")
    return result.stdout.strip()


def get_quotas(project: str = None) -> Dict[str, Any]:  # type: ignore
    """
    Fetch Gemini API quotas from GCP.
    
    Args:
        project: GCP project ID (defaults to current gcloud project)
        
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
    
    return results


def get_tier(project: str = None) -> str:  # type: ignore
    """
    Determine the pricing tier based on quota levels.
    
    Tier 1: Free tier (low limits)
    Tier 2: Pay-as-you-go (medium limits)
    Tier 3: Scale tier (high limits, 20K+ RPM for flash)
    
    Returns:
        Tier name string
    """
    quotas = get_quotas(project)
    
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


def main():
    parser = argparse.ArgumentParser(description="Check Gemini API quotas")
    parser.add_argument("--project", default=None, help="GCP project ID (defaults to current gcloud project)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--model", help="Filter to specific model")
    args = parser.parse_args()
    
    try:
        # Get project (auto-detect if not specified)
        project = args.project if args.project else get_current_project()
        
        quotas = get_quotas(project)
        tier = get_tier(project)
        
        if args.json:
            output = {
                "project": project,
                "tier": tier,
                "quotas": quotas
            }
            print(json.dumps(output, indent=2))
        else:
            print("=" * 60)
            print(f"GEMINI API QUOTAS - {project}")
            print(f"Tier: {tier}")
            print("=" * 60)
            print()
            
            # Sort by model name
            for model in sorted(quotas.keys()):
                if args.model and args.model not in model:
                    continue
                vals = quotas[model]
                rpm = vals.get("rpm", 0)
                tpm = vals.get("tpm", 0)
                print(f"{model}:")
                print(f"  RPM: {rpm:,}")
                print(f"  TPM: {tpm:,}")
                print()
                
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
