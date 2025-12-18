#!/usr/bin/env python3
"""
Gemini 3 Flash Preview Stress Test

Tests the gemini_client with high parallelism (50 workers) on 100 questions.
Uses the gold corpus for real questions but simulates the full pipeline
since sm-dev-01 has a merge conflict.

Usage:
    python scripts/eval/run_gemini3_test.py --questions 100 --workers 50
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from gemini_client import (
    generate_for_judge,
    generate_for_rag,
    get_model_info,
    health_check,
    _get_current_region,
    US_REGIONS,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Config
CORPUS_PATH = Path(__file__).parent.parent.parent / "clients" / "BFAI" / "qa" / "QA_BFAI_gold_v1-0__q458.json"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "reports" / "gemini3_test"

# Default to full corpus (458 questions)
DEFAULT_QUESTIONS = 458
DEFAULT_WORKERS = 50


def load_corpus(max_questions: int = DEFAULT_QUESTIONS):
    """Load questions from gold corpus."""
    if not CORPUS_PATH.exists():
        # Fallback to generating synthetic questions
        logger.warning(f"Corpus not found at {CORPUS_PATH}, using synthetic questions")
        return [
            {
                "question_id": f"Q{i:03d}",
                "question": f"What is the technical specification for component {i}?",
                "ground_truth_answer": f"The specification for component {i} is XYZ-{i*10}.",
                "question_type": "single_hop" if i % 2 == 0 else "multi_hop",
                "difficulty": ["easy", "medium", "hard"][i % 3],
            }
            for i in range(max_questions)
        ]
    
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    
    questions = data.get("questions", data)
    return questions[:max_questions]


def run_single_question(q: dict) -> dict:
    """Run a single question through generate + judge pipeline."""
    start = time.time()
    question_id = q.get("question_id", "unknown")
    question = q.get("question", "")
    ground_truth = q.get("ground_truth_answer", q.get("answer", ""))
    
    try:
        # Step 1: Generate RAG answer (simulated context)
        context = f"""
[1] Technical documentation for the equipment.
[2] The system operates within standard parameters.
[3] Refer to the manufacturer specifications for details.
"""
        
        gen_prompt = f"""You are a technical assistant for SCADA/Solar/Electrical equipment.
Answer the question based on the provided context. Be specific and accurate.

Context:
{context}

Question: {question}

Answer:"""
        
        # Import generate directly to get token counts
        from gemini_client import generate, GENERATOR_CONFIG
        
        gen_start = time.time()
        gen_result = generate(
            gen_prompt,
            max_output_tokens=GENERATOR_CONFIG.get("max_output_tokens"),
            temperature=GENERATOR_CONFIG.get("temperature"),
            thinking_level="LOW",
        )
        rag_answer = gen_result["text"]
        gen_time = time.time() - gen_start
        gen_usage = gen_result.get("usage", {})
        
        # Step 2: Judge the answer
        judge_prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth.

Question: {question}

Ground Truth: {ground_truth}

RAG Answer: {rag_answer}

Context: {context}

Score 1-5 for each (5=best):
1. correctness - factually correct vs ground truth?
2. completeness - covers key points?
3. faithfulness - faithful to context, no hallucinations?
4. relevance - relevant to question?
5. clarity - clear and well-structured?

Respond with JSON containing: correctness, completeness, faithfulness, relevance, clarity, overall_score (all 1-5), and verdict (pass|partial|fail)."""
        
        # Get judge result with token counts
        from gemini_client import generate, JUDGE_CONFIG
        import json as json_module
        
        judge_start = time.time()
        judge_result = generate(
            judge_prompt,
            response_mime_type="application/json",
            max_output_tokens=JUDGE_CONFIG.get("max_output_tokens"),
            temperature=JUDGE_CONFIG.get("temperature"),
            thinking_level="LOW",
        )
        judge_time = time.time() - judge_start
        judge_usage = judge_result.get("usage", {})
        
        # Parse judgment JSON
        judge_text = judge_result["text"]
        if judge_text:
            try:
                judgment = json_module.loads(judge_text)
                if isinstance(judgment, list) and len(judgment) > 0:
                    judgment = judgment[0]
            except:
                judgment = {"error": "parse_error", "verdict": "error"}
        else:
            judgment = {"error": "no_response", "verdict": "error"}
        
        total_time = time.time() - start
        
        return {
            "question_id": question_id,
            "question_type": q.get("question_type", "unknown"),
            "difficulty": q.get("difficulty", "unknown"),
            "success": True,
            "judgment": judgment,
            "rag_answer": rag_answer[:200] if rag_answer else None,
            "gen_time": gen_time,
            "judge_time": judge_time,
            "total_time": total_time,
            "tokens": {
                "gen_prompt": gen_usage.get("prompt_tokens", 0) if gen_usage else 0,
                "gen_response": gen_usage.get("response_tokens", 0) if gen_usage else 0,
                "gen_thinking": gen_usage.get("thinking_tokens", 0) if gen_usage else 0,
                "judge_prompt": judge_usage.get("prompt_tokens", 0) if judge_usage else 0,
                "judge_response": judge_usage.get("response_tokens", 0) if judge_usage else 0,
                "judge_thinking": judge_usage.get("thinking_tokens", 0) if judge_usage else 0,
            },
            "llm_metadata": {
                "gen": gen_result.get("llm_metadata"),
                "judge": judge_result.get("llm_metadata"),
            },
        }
        
    except Exception as e:
        logger.error(f"Error on {question_id}: {e}")
        return {
            "question_id": question_id,
            "success": False,
            "error": str(e),
            "total_time": time.time() - start,
        }


def run_parallel_test(questions: list, workers: int = 50):
    """Run parallel test with specified workers."""
    print(f"\n{'='*70}")
    print(f"GEMINI 3 FLASH PREVIEW STRESS TEST")
    print(f"{'='*70}")
    
    model_info = get_model_info()
    print(f"Model: {model_info['model_id']} ({model_info['status']})")
    print(f"Questions: {len(questions)}")
    print(f"Workers: {workers}")
    print(f"Regions: {US_REGIONS}")
    print(f"Current region: {_get_current_region()}")
    print(f"{'='*70}\n")
    
    # Health check
    print("Running health check...")
    health = health_check()
    if health["status"] != "healthy":
        print(f"ERROR: Health check failed: {health}")
        return None
    print(f"Health check: {health['status']}\n")
    
    # Run parallel
    results = []
    lock = Lock()
    processed = 0
    start_time = time.time()
    
    print(f"Starting parallel execution with {workers} workers...")
    print("-" * 70)
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single_question, q): q for q in questions}
        
        for future in as_completed(futures):
            q = futures[future]
            qid = q.get("question_id", "unknown")
            
            try:
                result = future.result(timeout=120)
                with lock:
                    results.append(result)
                    processed += 1
                    
                    if result["success"]:
                        verdict = result.get("judgment", {}).get("verdict", "?")
                        print(f"[{processed}/{len(questions)}] {qid}: {verdict} ({result['total_time']:.1f}s)")
                    else:
                        print(f"[{processed}/{len(questions)}] {qid}: ERROR - {result.get('error', 'unknown')[:50]}")
                        
            except Exception as e:
                with lock:
                    results.append({
                        "question_id": qid,
                        "success": False,
                        "error": str(e),
                    })
                    processed += 1
                    print(f"[{processed}/{len(questions)}] {qid}: TIMEOUT - {e}")
    
    total_time = time.time() - start_time
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Questions: {len(questions)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success rate: {len(successful)/len(questions)*100:.1f}%")
    
    if successful:
        avg_time = sum(r["total_time"] for r in successful) / len(successful)
        avg_gen = sum(r.get("gen_time", 0) for r in successful) / len(successful)
        avg_judge = sum(r.get("judge_time", 0) for r in successful) / len(successful)
        throughput = len(successful) / total_time * 60
        
        print(f"\nTiming:")
        print(f"  Avg per question: {avg_time:.2f}s")
        print(f"  Avg generation: {avg_gen:.2f}s")
        print(f"  Avg judge: {avg_judge:.2f}s")
        print(f"  Throughput: {throughput:.1f} questions/min")
        
        # Verdict distribution
        verdicts: dict[str, int] = {}
        for r in successful:
            judgment = r.get("judgment", {})
            # Handle case where judgment is a list instead of dict
            if isinstance(judgment, list):
                v = "parse_error"
            elif isinstance(judgment, dict):
                v = judgment.get("verdict", "unknown")
            else:
                v = "unknown"
            verdicts[v] = verdicts.get(v, 0) + 1
        
        print(f"\nVerdicts:")
        for v, count in sorted(verdicts.items()):
            print(f"  {v}: {count} ({count/len(successful)*100:.1f}%)")
        
        # Score averages
        scores = ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]
        print("\nAverage Scores:")
        for score in scores:
            values = []
            for r in successful:
                judgment = r.get("judgment", {})
                if isinstance(judgment, dict):
                    val = judgment.get(score)
                    if isinstance(val, (int, float)):
                        values.append(val)
            if values:
                print(f"  {score}: {sum(values)/len(values):.2f}")
        
        # Token counts
        print("\nToken Usage:")
        token_fields = ["gen_prompt", "gen_response", "gen_thinking", "judge_prompt", "judge_response", "judge_thinking"]
        for field in token_fields:
            values = [r.get("tokens", {}).get(field, 0) for r in successful if r.get("tokens")]
            if values:
                total = sum(values)
                avg = total / len(values)
                print(f"  {field}: {total:,} total, {avg:.0f} avg")
        
        # Total tokens
        total_all = sum(
            sum(r.get("tokens", {}).get(f, 0) for f in token_fields)
            for r in successful if r.get("tokens")
        )
        print(f"  TOTAL: {total_all:,} tokens")
    
    if failed:
        print(f"\nFailed questions:")
        for r in failed[:5]:
            print(f"  {r['question_id']}: {r.get('error', 'unknown')[:60]}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")
    
    print("=" * 70)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"gemini3_test_{timestamp}.json"
    
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model_info["model_id"],
            "questions": len(questions),
            "workers": workers,
            "total_time_seconds": total_time,
            "throughput_per_minute": len(successful) / total_time * 60 if successful else 0,
        },
        "summary": {
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(questions),
        },
        "results": results,
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description="Gemini 3 Flash Preview Stress Test")
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS, help="Number of questions to run (default: 458)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers (default: 50)")
    args = parser.parse_args()
    
    questions = load_corpus(args.questions)
    run_parallel_test(questions, args.workers)


if __name__ == "__main__":
    main()
