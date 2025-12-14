#!/usr/bin/env python3
"""
Full evaluation pipeline:
1. Generate 200+ Q&A pairs from Knowledge Graph
2. Run LLM-as-Judge evaluation
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_step(script_name: str, description: str) -> bool:
    """Run a Python script and return success status"""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"{'='*60}")
    
    script_path = Path(__file__).parent / script_name
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(Path(__file__).parent),
    )
    
    return result.returncode == 0


def main():
    start_time = datetime.now()
    
    print("="*60)
    print("FULL EVALUATION PIPELINE")
    print(f"Started: {start_time.isoformat()}")
    print("="*60)
    
    # Step 1: Generate 200+ Q&A pairs
    success = run_step("generate_qa_200.py", "Generate 200+ Q&A pairs with multi-hop")
    if not success:
        print("\n❌ Q&A generation failed!")
        return 1
    
    # Step 2: Run LLM-as-Judge
    success = run_step("llm_as_judge.py", "Run LLM-as-Judge evaluation")
    if not success:
        print("\n❌ LLM-as-Judge evaluation failed!")
        return 1
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Started:  {start_time.isoformat()}")
    print(f"Finished: {end_time.isoformat()}")
    print(f"Duration: {duration}")
    print("\nOutput files:")
    print("  - output/qa_corpus_200.json (Q&A pairs)")
    print("  - output/llm_judge_results.json (Evaluation results)")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
