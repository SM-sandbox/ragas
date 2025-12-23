#!/usr/bin/env python3
"""
Failure Analysis Script for Checkpoint Results

This script analyzes checkpoint results to identify and classify failures (scores 1-3)
into archetypes for pipeline debugging. It generates a deliverable package for the
pipeline engineer to investigate chunking/metadata issues.

Archetypes:
- INCOMPLETE_CONTEXT: Retrieved chunks missing full answer
- WRONG_DOCUMENT: Relevant doc ranked poorly (low MRR)
- HALLUCINATION: Generated plausible but incorrect info
- NUMERICAL_PRECISION: Exact numbers wrong or missing
- COMPLEX_REASONING: Multi-step reasoning failed
- NO_FAILURE: Judge disagreement (actually correct)

Usage:
    python scripts/analyze_failures.py <checkpoint_dir> [--output-dir <dir>]
    python scripts/analyze_failures.py --checkpoint C020
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.clients.gemini_client import generate_json

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "clients_eval_data" / "BFAI" / "checkpoints"

# Classification prompt
ARCHETYPE_PROMPT = """You are an expert at analyzing RAG system failures. Given a failed question evaluation, classify the failure into one of these archetypes:

**ARCHETYPES:**
1. **INCOMPLETE_CONTEXT** - Retrieved chunks were relevant but missing key information needed for complete answer
2. **WRONG_DOCUMENT** - Correct document was recalled but ranked poorly (low MRR), or wrong document was prioritized
3. **HALLUCINATION** - LLM generated plausible but incorrect information not supported by context
4. **NUMERICAL_PRECISION** - Exact numbers, product codes, or specifications were wrong or missing
5. **COMPLEX_REASONING** - Multi-hop or synthesis reasoning failed despite good retrieval
6. **NO_FAILURE** - The answer appears correct; judge may have been too strict

**QUESTION DATA:**
- Question ID: {question_id}
- Question Type: {question_type}
- Difficulty: {difficulty}
- Question: {question}
- Ground Truth Answer: {ground_truth}
- RAG Answer: {rag_answer}

**EVALUATION SCORES:**
- Correctness: {correctness}/5
- Completeness: {completeness}/5
- Faithfulness: {faithfulness}/5
- Relevance: {relevance}/5
- Clarity: {clarity}/5
- Overall Score: {overall_score}/5
- Verdict: {verdict}

**RETRIEVAL METRICS:**
- Recall Hit: {recall_hit}
- MRR: {mrr}
- Source Files: {source_files}

**CONTEXT (first 1500 chars):**
{context}

Analyze this failure and respond with JSON:
{{
    "primary_archetype": "<one of the 6 archetypes>",
    "secondary_archetypes": ["<optional additional archetypes>"],
    "root_cause": "<1-2 sentence explanation of what went wrong>",
    "chunking_issue": "<specific chunking/metadata issue if applicable, or 'N/A'>",
    "recommendation": "<specific fix recommendation for pipeline engineer>"
}}
"""


def find_checkpoint_dir(checkpoint_id: str) -> Path:
    """Find checkpoint directory by ID (e.g., C020)."""
    matches = list(CHECKPOINTS_DIR.glob(f"{checkpoint_id}__*"))
    if not matches:
        raise ValueError(f"No checkpoint found matching: {checkpoint_id}")
    if len(matches) > 1:
        # Return most recent
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def load_checkpoint_results(checkpoint_dir: Path) -> dict:
    """Load results from checkpoint directory."""
    results_file = checkpoint_dir / "results.json"
    if not results_file.exists():
        # Try checkpoint.json
        results_file = checkpoint_dir / "checkpoint.json"
    
    if not results_file.exists():
        raise FileNotFoundError(f"No results file found in {checkpoint_dir}")
    
    with open(results_file) as f:
        data = json.load(f)
    
    # Handle both formats (results.json has "results" key, checkpoint.json is array)
    if isinstance(data, list):
        return {"results": data}
    return data


def extract_failures(results: list, max_score: int = 3) -> list:
    """Extract questions with overall_score <= max_score."""
    failures = []
    for r in results:
        judgment = r.get("judgment", {})
        overall = judgment.get("overall_score", 5)
        if overall <= max_score and "error" not in r:
            failures.append(r)
    return failures


def classify_failure(result: dict) -> dict:
    """Classify a single failure using LLM."""
    judgment = result.get("judgment", {})
    
    # Build prompt
    prompt = ARCHETYPE_PROMPT.format(
        question_id=result.get("question_id", "unknown"),
        question_type=result.get("question_type", "unknown"),
        difficulty=result.get("difficulty", "unknown"),
        question=result.get("question", "N/A"),
        ground_truth=result.get("ground_truth_answer", "N/A"),
        rag_answer=result.get("rag_answer", "N/A")[:2000],
        correctness=judgment.get("correctness", "?"),
        completeness=judgment.get("completeness", "?"),
        faithfulness=judgment.get("faithfulness", "?"),
        relevance=judgment.get("relevance", "?"),
        clarity=judgment.get("clarity", "?"),
        overall_score=judgment.get("overall_score", "?"),
        verdict=judgment.get("verdict", "?"),
        recall_hit=result.get("recall_hit", "unknown"),
        mrr=result.get("mrr", "unknown"),
        source_files=result.get("source_filenames", []),
        context=result.get("context", "N/A")[:1500],
    )
    
    try:
        response = generate_json(
            prompt,
            model="gemini-2.5-flash",
            temperature=0.0,
        )
        
        # Parse response - generate_json returns the parsed JSON directly
        analysis = response
        
        return {
            "question_id": result.get("question_id"),
            "question": result.get("question"),
            "ground_truth": result.get("ground_truth_answer"),
            "rag_answer": result.get("rag_answer", "")[:500],
            "question_type": result.get("question_type"),
            "difficulty": result.get("difficulty"),
            "source_filenames": result.get("source_filenames", []),
            "scores": {
                "correctness": judgment.get("correctness"),
                "completeness": judgment.get("completeness"),
                "faithfulness": judgment.get("faithfulness"),
                "relevance": judgment.get("relevance"),
                "clarity": judgment.get("clarity"),
                "overall_score": judgment.get("overall_score"),
                "verdict": judgment.get("verdict"),
            },
            "retrieval": {
                "recall_hit": result.get("recall_hit"),
                "mrr": result.get("mrr"),
            },
            "analysis": analysis,
        }
    except Exception as e:
        return {
            "question_id": result.get("question_id"),
            "error": str(e),
        }


def generate_summary(classified: list) -> dict:
    """Generate summary statistics from classified failures."""
    archetype_counts = defaultdict(int)
    by_difficulty = defaultdict(lambda: defaultdict(int))
    by_type = defaultdict(lambda: defaultdict(int))
    chunking_issues = []
    
    for item in classified:
        if "error" in item:
            continue
        
        analysis = item.get("analysis", {})
        primary = analysis.get("primary_archetype", "UNKNOWN")
        archetype_counts[primary] += 1
        
        diff = item.get("difficulty", "unknown")
        qtype = item.get("question_type", "unknown")
        by_difficulty[diff][primary] += 1
        by_type[qtype][primary] += 1
        
        chunking = analysis.get("chunking_issue", "N/A")
        if chunking and chunking != "N/A":
            chunking_issues.append({
                "question_id": item.get("question_id"),
                "issue": chunking,
                "source_files": item.get("source_filenames", []),
            })
    
    return {
        "total_failures": len(classified),
        "archetype_distribution": dict(archetype_counts),
        "by_difficulty": {k: dict(v) for k, v in by_difficulty.items()},
        "by_type": {k: dict(v) for k, v in by_type.items()},
        "chunking_issues": chunking_issues,
    }


def generate_engineering_package(classified: list, summary: dict, checkpoint_id: str, output_dir: Path):
    """Generate the deliverable package for pipeline engineer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Full analysis JSON
    analysis_file = output_dir / "failure_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump({
            "metadata": {
                "checkpoint": checkpoint_id,
                "generated": datetime.now().isoformat(),
                "total_failures": summary["total_failures"],
            },
            "summary": summary,
            "failures": classified,
        }, f, indent=2)
    
    # 2. Chunking issues CSV-style for easy review
    chunking_file = output_dir / "chunking_issues.json"
    with open(chunking_file, "w") as f:
        json.dump(summary["chunking_issues"], f, indent=2)
    
    # 3. Questions by archetype for targeted investigation
    by_archetype = defaultdict(list)
    for item in classified:
        if "error" not in item:
            archetype = item.get("analysis", {}).get("primary_archetype", "UNKNOWN")
            by_archetype[archetype].append({
                "question_id": item.get("question_id"),
                "question": item.get("question"),
                "ground_truth": item.get("ground_truth"),
                "source_files": item.get("source_filenames", []),
                "scores": item.get("scores"),
                "recommendation": item.get("analysis", {}).get("recommendation"),
            })
    
    archetype_file = output_dir / "failures_by_archetype.json"
    with open(archetype_file, "w") as f:
        json.dump(dict(by_archetype), f, indent=2)
    
    # 4. Markdown summary report
    report = generate_markdown_report(classified, summary, checkpoint_id)
    report_file = output_dir / "FAILURE_ANALYSIS_REPORT.md"
    with open(report_file, "w") as f:
        f.write(report)
    
    # 5. README for engineer
    readme = f"""# Failure Analysis Package - {checkpoint_id}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Contents

1. **FAILURE_ANALYSIS_REPORT.md** - Human-readable summary with recommendations
2. **failure_analysis.json** - Complete analysis data with all details
3. **failures_by_archetype.json** - Failures grouped by root cause type
4. **chunking_issues.json** - Specific chunking/metadata issues to investigate

## Quick Stats

- Total Failures Analyzed: {summary['total_failures']}
- Top Archetype: {max(summary['archetype_distribution'], key=summary['archetype_distribution'].get) if summary['archetype_distribution'] else 'N/A'}

## Archetype Distribution

"""
    for arch, count in sorted(summary['archetype_distribution'].items(), key=lambda x: -x[1]):
        pct = count / summary['total_failures'] * 100 if summary['total_failures'] > 0 else 0
        readme += f"- **{arch}**: {count} ({pct:.1f}%)\n"
    
    readme += """
## How to Use

1. Start with `FAILURE_ANALYSIS_REPORT.md` for overview
2. Check `chunking_issues.json` for specific files to investigate
3. Use `failures_by_archetype.json` to focus on specific failure types
4. Reference `failure_analysis.json` for complete details

## Priority Actions

Focus on **INCOMPLETE_CONTEXT** and **WRONG_DOCUMENT** failures first - 
these are typically addressable through chunking and metadata improvements.
"""
    
    readme_file = output_dir / "README.md"
    with open(readme_file, "w") as f:
        f.write(readme)
    
    return output_dir


def generate_markdown_report(classified: list, summary: dict, checkpoint_id: str) -> str:
    """Generate markdown report."""
    report = f"""# Failure Analysis Report - {checkpoint_id}

**Generated:** {datetime.now().strftime("%B %d, %Y")}  
**Total Failures Analyzed:** {summary['total_failures']}

---

## Executive Summary

"""
    
    # Archetype table
    report += "| Archetype | Count | % | Description |\n"
    report += "|-----------|-------|---|-------------|\n"
    
    descriptions = {
        "INCOMPLETE_CONTEXT": "Retrieved chunks missing full answer",
        "WRONG_DOCUMENT": "Relevant doc ranked poorly (low MRR)",
        "HALLUCINATION": "Generated plausible but incorrect info",
        "NUMERICAL_PRECISION": "Exact numbers wrong or missing",
        "COMPLEX_REASONING": "Multi-step reasoning failed",
        "NO_FAILURE": "Judge disagreement (actually correct)",
        "UNKNOWN": "Classification failed",
    }
    
    for arch, count in sorted(summary['archetype_distribution'].items(), key=lambda x: -x[1]):
        pct = count / summary['total_failures'] * 100 if summary['total_failures'] > 0 else 0
        desc = descriptions.get(arch, "Unknown")
        report += f"| **{arch}** | {count} | {pct:.1f}% | {desc} |\n"
    
    # Key insight
    retrieval_related = summary['archetype_distribution'].get('INCOMPLETE_CONTEXT', 0) + \
                       summary['archetype_distribution'].get('WRONG_DOCUMENT', 0)
    retrieval_pct = retrieval_related / summary['total_failures'] * 100 if summary['total_failures'] > 0 else 0
    
    report += f"""
### Key Insight

**{retrieval_pct:.0f}% of failures are retrieval-related** (INCOMPLETE_CONTEXT + WRONG_DOCUMENT). 
These are addressable through better chunking and metadata strategies.

---

## Breakdown by Difficulty

| Difficulty | """
    
    archetypes = list(summary['archetype_distribution'].keys())
    report += " | ".join(archetypes) + " |\n"
    report += "|------------|" + "|".join(["---"] * len(archetypes)) + "|\n"
    
    for diff in ["easy", "medium", "hard"]:
        if diff in summary['by_difficulty']:
            row = f"| **{diff.title()}** |"
            for arch in archetypes:
                count = summary['by_difficulty'][diff].get(arch, 0)
                row += f" {count} |"
            report += row + "\n"
    
    report += """
---

## Chunking Issues to Investigate

"""
    
    if summary['chunking_issues']:
        for issue in summary['chunking_issues'][:20]:
            report += f"- **{issue['question_id']}**: {issue['issue']}\n"
            if issue['source_files']:
                report += f"  - Files: {', '.join(issue['source_files'][:3])}\n"
    else:
        report += "*No specific chunking issues identified.*\n"
    
    report += """
---

## Detailed Failures

"""
    
    # Group by archetype and show top examples
    by_arch = defaultdict(list)
    for item in classified:
        if "error" not in item:
            arch = item.get("analysis", {}).get("primary_archetype", "UNKNOWN")
            by_arch[arch].append(item)
    
    for arch in sorted(by_arch.keys()):
        items = by_arch[arch]
        report += f"### {arch} ({len(items)} failures)\n\n"
        
        for item in items[:5]:  # Show top 5 per archetype
            report += f"**{item['question_id']}** ({item.get('difficulty', '?')}, {item.get('question_type', '?')})\n\n"
            question_text = item.get('question') or 'N/A'
            report += f"> {question_text[:200]}...\n\n"
            ground_truth_text = item.get('ground_truth') or 'N/A'
            report += f"- **Ground Truth:** {ground_truth_text[:150]}...\n"
            report += f"- **Scores:** corr={item['scores'].get('correctness')}, comp={item['scores'].get('completeness')}, faith={item['scores'].get('faithfulness')}\n"
            report += f"- **Root Cause:** {item.get('analysis', {}).get('root_cause', 'N/A')}\n"
            report += f"- **Recommendation:** {item.get('analysis', {}).get('recommendation', 'N/A')}\n\n"
        
        if len(items) > 5:
            report += f"*...and {len(items) - 5} more*\n\n"
        
        report += "---\n\n"
    
    report += """
## Recommendations

### High Priority
1. Review chunking strategy for documents with INCOMPLETE_CONTEXT failures
2. Investigate reranking for WRONG_DOCUMENT cases
3. Check metadata extraction for NUMERICAL_PRECISION issues

### Medium Priority
4. Enhance prompts for COMPLEX_REASONING questions
5. Review judge calibration for NO_FAILURE cases

---

*Report generated by automated failure analysis pipeline.*
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint failures")
    parser.add_argument("checkpoint_dir", nargs="?", help="Path to checkpoint directory")
    parser.add_argument("--checkpoint", "-c", help="Checkpoint ID (e.g., C020)")
    parser.add_argument("--output-dir", "-o", help="Output directory for analysis package")
    parser.add_argument("--max-score", type=int, default=3, help="Max overall_score to consider as failure (default: 3)")
    parser.add_argument("--workers", type=int, default=5, help="Parallel workers for classification")
    args = parser.parse_args()
    
    # Determine checkpoint directory
    if args.checkpoint:
        checkpoint_dir = find_checkpoint_dir(args.checkpoint)
        checkpoint_id = args.checkpoint
    elif args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_id = checkpoint_dir.name.split("__")[0]
    else:
        parser.error("Must specify either checkpoint_dir or --checkpoint")
    
    print(f"Analyzing checkpoint: {checkpoint_dir.name}")
    
    # Load results
    data = load_checkpoint_results(checkpoint_dir)
    results = data.get("results", [])
    print(f"Loaded {len(results)} results")
    
    # Extract failures
    failures = extract_failures(results, args.max_score)
    print(f"Found {len(failures)} failures (score <= {args.max_score})")
    
    if not failures:
        print("No failures to analyze!")
        return
    
    # Classify failures
    print(f"\nClassifying failures with {args.workers} workers...")
    classified = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(classify_failure, f): f for f in failures}
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            classified.append(result)
            
            if "error" in result:
                print(f"  [{i+1}/{len(failures)}] {result['question_id']}: ERROR - {result['error']}")
            else:
                arch = result.get("analysis", {}).get("primary_archetype", "?")
                print(f"  [{i+1}/{len(failures)}] {result['question_id']}: {arch}")
    
    # Generate summary
    summary = generate_summary(classified)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_dir / "failure_analysis"
    
    # Generate package
    print(f"\nGenerating engineering package...")
    output_dir = generate_engineering_package(classified, summary, checkpoint_id, output_dir)
    
    # Print summary
    print(f"\n{'='*60}")
    print("FAILURE ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total Failures: {summary['total_failures']}")
    print(f"\nArchetype Distribution:")
    for arch, count in sorted(summary['archetype_distribution'].items(), key=lambda x: -x[1]):
        pct = count / summary['total_failures'] * 100
        print(f"  {arch}: {count} ({pct:.1f}%)")
    
    print(f"\nOutput: {output_dir}")
    print(f"  - README.md")
    print(f"  - FAILURE_ANALYSIS_REPORT.md")
    print(f"  - failure_analysis.json")
    print(f"  - failures_by_archetype.json")
    print(f"  - chunking_issues.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
