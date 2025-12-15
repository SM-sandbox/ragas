#!/usr/bin/env python3
"""
Generate Latency Report

Extracts timing data from all evaluation runs and produces a report showing:
- Retrieval latency (seconds)
- Generation latency (seconds)  
- Judge/LLM call latency (seconds)
- Total latency

For all embedding configurations tested.
"""

import json
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


def extract_timing_from_gcp_file(filepath: str) -> Dict[str, Dict]:
    """Extract timing data from GCP embedding comparison files."""
    with open(filepath) as f:
        data = json.load(f)
    
    results = {}
    
    if 'detailed_results' in data:
        for config_name, config_data in data['detailed_results'].items():
            if isinstance(config_data, dict) and 'results' in config_data:
                timings = []
                for r in config_data['results']:
                    if isinstance(r.get('timing'), dict):
                        timings.append(r['timing'])
                    elif isinstance(r.get('timing'), str):
                        try:
                            timings.append(eval(r['timing']))
                        except:
                            pass
                
                if timings:
                    results[config_name] = {
                        'retrieval_seconds': [t.get('retrieval_seconds', 0) for t in timings if t.get('retrieval_seconds')],
                        'generation_seconds': [t.get('generation_seconds', 0) for t in timings if t.get('generation_seconds')],
                        'judge_seconds': [t.get('judge_seconds', 0) for t in timings if t.get('judge_seconds')],
                        'total_seconds': [t.get('total_seconds', 0) for t in timings if t.get('total_seconds')],
                        'count': len(timings)
                    }
    
    return results


def extract_timing_from_azure_file(filepath: str) -> Dict[str, Dict]:
    """Extract timing data from Azure evaluation files."""
    with open(filepath) as f:
        data = json.load(f)
    
    results = {}
    config_name = 'azure-text-embedding-3-large'
    
    if 'results' in data:
        timings = []
        for r in data['results']:
            if isinstance(r.get('timing'), dict):
                timings.append(r['timing'])
        
        if timings:
            results[config_name] = {
                'retrieval_seconds': [t.get('retrieval_seconds', 0) for t in timings if t.get('retrieval_seconds')],
                'generation_seconds': [t.get('generation_seconds', 0) for t in timings if t.get('generation_seconds')],
                'judge_seconds': [t.get('judge_seconds', 0) for t in timings if t.get('judge_seconds')],
                'total_seconds': [t.get('total_seconds', 0) for t in timings if t.get('total_seconds')],
                'count': len(timings)
            }
    
    return results


def aggregate_timings(all_timings: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Aggregate timing data across multiple runs."""
    aggregated = {}
    
    for config_name, timing_list in all_timings.items():
        retrieval = []
        generation = []
        judge = []
        total = []
        
        for t in timing_list:
            retrieval.extend(t.get('retrieval_seconds', []))
            generation.extend(t.get('generation_seconds', []))
            judge.extend(t.get('judge_seconds', []))
            total.extend(t.get('total_seconds', []))
        
        def safe_stats(values):
            if not values:
                return {'avg': 0, 'min': 0, 'max': 0, 'count': 0}
            return {
                'avg': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        
        aggregated[config_name] = {
            'retrieval': safe_stats(retrieval),
            'generation': safe_stats(generation),
            'judge': safe_stats(judge),
            'total': safe_stats(total)
        }
    
    return aggregated


def generate_report(aggregated: Dict[str, Dict]) -> str:
    """Generate markdown report."""
    lines = [
        "# Latency Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This report shows timing data for retrieval, generation, and LLM judge calls across all embedding configurations.",
        "",
        "## Summary Table",
        "",
        "| Config | Retrieval (s) | Generation (s) | Judge (s) | Total (s) | Samples |",
        "|--------|---------------|----------------|-----------|-----------|---------|",
    ]
    
    # Sort by config name
    for config_name in sorted(aggregated.keys()):
        stats = aggregated[config_name]
        r = stats['retrieval']
        g = stats['generation']
        j = stats['judge']
        t = stats['total']
        
        lines.append(
            f"| {config_name} | {r['avg']:.2f} | {g['avg']:.2f} | {j['avg']:.2f} | {t['avg']:.2f} | {t['count']} |"
        )
    
    lines.extend([
        "",
        "## Detailed Statistics",
        "",
    ])
    
    for config_name in sorted(aggregated.keys()):
        stats = aggregated[config_name]
        lines.extend([
            f"### {config_name}",
            "",
            "| Metric | Avg (s) | Min (s) | Max (s) | Samples |",
            "|--------|---------|---------|---------|---------|",
        ])
        
        for metric_name, metric_key in [('Retrieval', 'retrieval'), ('Generation', 'generation'), 
                                         ('Judge/LLM', 'judge'), ('Total', 'total')]:
            s = stats[metric_key]
            lines.append(f"| {metric_name} | {s['avg']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {s['count']} |")
        
        lines.append("")
    
    # Add observations
    lines.extend([
        "## Observations",
        "",
    ])
    
    # Find fastest/slowest
    if aggregated:
        by_retrieval = sorted(aggregated.items(), key=lambda x: x[1]['retrieval']['avg'] if x[1]['retrieval']['avg'] > 0 else float('inf'))
        by_generation = sorted(aggregated.items(), key=lambda x: x[1]['generation']['avg'] if x[1]['generation']['avg'] > 0 else float('inf'))
        by_total = sorted(aggregated.items(), key=lambda x: x[1]['total']['avg'] if x[1]['total']['avg'] > 0 else float('inf'))
        
        if by_retrieval and by_retrieval[0][1]['retrieval']['avg'] > 0:
            lines.append(f"- **Fastest Retrieval:** {by_retrieval[0][0]} ({by_retrieval[0][1]['retrieval']['avg']:.2f}s avg)")
        if by_generation and by_generation[0][1]['generation']['avg'] > 0:
            lines.append(f"- **Fastest Generation:** {by_generation[0][0]} ({by_generation[0][1]['generation']['avg']:.2f}s avg)")
        if by_total and by_total[0][1]['total']['avg'] > 0:
            lines.append(f"- **Fastest Total:** {by_total[0][0]} ({by_total[0][1]['total']['avg']:.2f}s avg)")
    
    return "\n".join(lines)


def main():
    """Generate the latency report."""
    print("=" * 70)
    print("LATENCY REPORT GENERATOR")
    print("=" * 70)
    
    # Collect all timing data
    all_timings = defaultdict(list)
    
    # GCP embedding comparison files
    gcp_files = list(glob.glob(str(EXPERIMENTS_DIR / "2024-12-14_embedding_model_comparison/data/embedding_comparison*.json")))
    gcp_files += list(glob.glob(str(EXPERIMENTS_DIR / "2024-12-14_embedding_dimension_test/data/embedding*.json")))
    
    print(f"Found {len(gcp_files)} GCP evaluation files")
    
    for f in gcp_files:
        try:
            timings = extract_timing_from_gcp_file(f)
            for config_name, timing_data in timings.items():
                all_timings[config_name].append(timing_data)
        except Exception as e:
            print(f"  Warning: Could not parse {f}: {e}")
    
    # Azure files
    azure_files = list(glob.glob(str(EXPERIMENTS_DIR / "2024-12-14_azure_vs_gcp/data/azure_evaluation*.json")))
    print(f"Found {len(azure_files)} Azure evaluation files")
    
    for f in azure_files:
        try:
            timings = extract_timing_from_azure_file(f)
            for config_name, timing_data in timings.items():
                all_timings[config_name].append(timing_data)
        except Exception as e:
            print(f"  Warning: Could not parse {f}: {e}")
    
    print(f"\nConfigs found: {list(all_timings.keys())}")
    
    # Aggregate
    aggregated = aggregate_timings(all_timings)
    
    # Generate report
    report = generate_report(aggregated)
    
    # Save report
    REPORTS_DIR.mkdir(exist_ok=True)
    output_file = REPORTS_DIR / "Latency_Report.md"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved to: {output_file}")
    print("\n" + "=" * 70)
    print(report)


if __name__ == "__main__":
    main()
