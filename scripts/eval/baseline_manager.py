"""
Baseline Manager for Core Eval

Handles loading, saving, and comparing baselines for RAG evaluation.
Baselines are versioned JSON files stored in the baselines/ directory.

Naming convention: baseline_{CLIENT}_v{VERSION}__{DATE}__q{COUNT}.json
Example: baseline_BFAI_v1__2025-12-17__q458.json
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Default baselines directory
BASELINES_DIR = Path(__file__).parent.parent.parent / "baselines"


def get_baseline_path(client: str, version: str, date: str, question_count: int) -> Path:
    """Generate baseline file path."""
    filename = f"baseline_{client}_v{version}__{date}__q{question_count}.json"
    return BASELINES_DIR / filename


def parse_baseline_filename(filename: str) -> Optional[Dict[str, str]]:
    """
    Parse baseline filename to extract metadata.
    
    Returns:
        Dict with client, version, date, question_count or None if invalid
    """
    pattern = r"baseline_(\w+)_v(\d+)__(\d{4}-\d{2}-\d{2})__q(\d+)\.json"
    match = re.match(pattern, filename)
    if match:
        return {
            "client": match.group(1),
            "version": match.group(2),
            "date": match.group(3),
            "question_count": int(match.group(4)),
        }
    return None


def list_baselines(client: Optional[str] = None) -> List[Dict]:
    """
    List all available baselines.
    
    Args:
        client: Filter by client name (optional)
        
    Returns:
        List of baseline metadata dicts, sorted by date descending
    """
    if not BASELINES_DIR.exists():
        return []
    
    baselines = []
    for f in BASELINES_DIR.glob("baseline_*.json"):
        meta = parse_baseline_filename(f.name)
        if meta:
            if client is None or meta["client"].upper() == client.upper():
                meta["path"] = str(f)
                baselines.append(meta)
    
    # Sort by date descending (newest first)
    baselines.sort(key=lambda x: x["date"], reverse=True)
    return baselines


def get_latest_baseline(client: str) -> Optional[Dict]:
    """
    Load the latest baseline for a client.
    
    Args:
        client: Client name (e.g., "BFAI")
        
    Returns:
        Baseline data dict or None if not found
    """
    baselines = list_baselines(client)
    if not baselines:
        return None
    
    latest = baselines[0]
    with open(latest["path"], "r") as f:
        return json.load(f)


def load_baseline(path: str) -> Dict:
    """Load a specific baseline file."""
    with open(path, "r") as f:
        return json.load(f)


def save_baseline(
    data: Dict,
    client: str,
    version: Optional[str] = None,
    date: Optional[str] = None,
) -> Path:
    """
    Save a new baseline.
    
    Args:
        data: Baseline data dict
        client: Client name
        version: Version number (auto-incremented if None)
        date: Date string (today if None)
        
    Returns:
        Path to saved baseline file
    """
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Auto-increment version if not specified
    if version is None:
        existing = list_baselines(client)
        if existing:
            max_version = max(int(b["version"]) for b in existing)
            version = str(max_version + 1)
        else:
            version = "1"
    
    # Use today's date if not specified
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    
    # Get question count from data
    question_count = data.get("corpus", {}).get("question_count", 0)
    
    # Generate path and save
    path = get_baseline_path(client, version, date, question_count)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    return path


def compare_to_baseline(current: Dict, baseline: Dict) -> Dict:
    """
    Compare current run results to baseline.
    
    Args:
        current: Current run metrics
        baseline: Baseline metrics
        
    Returns:
        Dict with deltas for each metric
    """
    comparison = {
        "baseline_version": baseline.get("baseline_version", "unknown"),
        "baseline_date": baseline.get("created_date", "unknown"),
        "deltas": {},
        "regressions": [],
        "improvements": [],
    }
    
    # Compare metrics
    baseline_metrics = baseline.get("metrics", {})
    current_metrics = current.get("metrics", {})
    
    metric_keys = [
        ("pass_rate", 0.02),      # 2% threshold
        ("partial_rate", 0.05),
        ("fail_rate", 0.02),
        ("acceptable_rate", 0.02),
        ("recall_at_100", 0.01),
        ("mrr", 0.02),
        ("overall_score_avg", 0.1),
    ]
    
    for key, threshold in metric_keys:
        baseline_val = baseline_metrics.get(key, 0)
        current_val = current_metrics.get(key, 0)
        delta = current_val - baseline_val
        
        comparison["deltas"][key] = {
            "baseline": baseline_val,
            "current": current_val,
            "delta": round(delta, 4),
            "delta_pct": round(delta * 100, 2) if baseline_val else 0,
        }
        
        # Check for regressions (negative delta for positive metrics)
        if key in ["pass_rate", "acceptable_rate", "recall_at_100", "mrr", "overall_score_avg"]:
            if delta < -threshold:
                comparison["regressions"].append(key)
            elif delta > threshold:
                comparison["improvements"].append(key)
        # For fail_rate, positive delta is bad
        elif key == "fail_rate":
            if delta > threshold:
                comparison["regressions"].append(key)
            elif delta < -threshold:
                comparison["improvements"].append(key)
    
    # Compare latency
    baseline_latency = baseline.get("latency", {})
    current_latency = current.get("latency", {})
    
    if baseline_latency.get("total_avg_s") and current_latency.get("total_avg_s"):
        latency_delta = current_latency["total_avg_s"] - baseline_latency["total_avg_s"]
        comparison["deltas"]["latency_avg_s"] = {
            "baseline": baseline_latency["total_avg_s"],
            "current": current_latency["total_avg_s"],
            "delta": round(latency_delta, 2),
        }
    
    # Compare cost
    baseline_cost = baseline.get("cost", {})
    current_cost = current.get("cost", {})
    
    if baseline_cost.get("total_cost_usd") is not None and current_cost.get("total_cost_usd") is not None:
        cost_delta = current_cost["total_cost_usd"] - baseline_cost["total_cost_usd"]
        comparison["deltas"]["cost_usd"] = {
            "baseline": baseline_cost["total_cost_usd"],
            "current": current_cost["total_cost_usd"],
            "delta": round(cost_delta, 4),
        }
    
    return comparison


def format_comparison_report(comparison: Dict) -> str:
    """Format comparison as a markdown report section."""
    lines = [
        "## Comparison to Baseline",
        "",
        f"**Baseline:** v{comparison['baseline_version']} ({comparison['baseline_date']})",
        "",
    ]
    
    # Regressions
    if comparison["regressions"]:
        lines.append("### ⚠️ Regressions")
        for key in comparison["regressions"]:
            delta = comparison["deltas"][key]
            lines.append(f"- **{key}**: {delta['baseline']:.3f} → {delta['current']:.3f} ({delta['delta']:+.3f})")
        lines.append("")
    
    # Improvements
    if comparison["improvements"]:
        lines.append("### ✅ Improvements")
        for key in comparison["improvements"]:
            delta = comparison["deltas"][key]
            lines.append(f"- **{key}**: {delta['baseline']:.3f} → {delta['current']:.3f} ({delta['delta']:+.3f})")
        lines.append("")
    
    # Full delta table
    lines.append("### Metric Deltas")
    lines.append("")
    lines.append("| Metric | Baseline | Current | Δ |")
    lines.append("|--------|----------|---------|---|")
    
    for key, delta in comparison["deltas"].items():
        baseline_val = delta["baseline"]
        current_val = delta["current"]
        delta_val = delta["delta"]
        
        # Format based on type
        if "rate" in key or key in ["mrr", "recall_at_100"]:
            lines.append(f"| {key} | {baseline_val:.1%} | {current_val:.1%} | {delta_val:+.1%} |")
        elif "cost" in key:
            lines.append(f"| {key} | ${baseline_val:.4f} | ${current_val:.4f} | ${delta_val:+.4f} |")
        elif "latency" in key:
            lines.append(f"| {key} | {baseline_val:.2f}s | {current_val:.2f}s | {delta_val:+.2f}s |")
        else:
            lines.append(f"| {key} | {baseline_val:.3f} | {current_val:.3f} | {delta_val:+.3f} |")
    
    lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test baseline manager
    print("Baseline Manager Test")
    print("=" * 50)
    
    # List baselines
    baselines = list_baselines()
    print(f"\nFound {len(baselines)} baselines:")
    for b in baselines:
        print(f"  - {b['client']} v{b['version']} ({b['date']}) - {b['question_count']} questions")
    
    # Load latest BFAI baseline
    latest = get_latest_baseline("BFAI")
    if latest:
        print(f"\nLatest BFAI baseline:")
        print(f"  Version: {latest.get('baseline_version')}")
        print(f"  Date: {latest.get('created_date')}")
        print(f"  Pass rate: {latest.get('metrics', {}).get('pass_rate', 0):.1%}")
