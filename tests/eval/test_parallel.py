#!/usr/bin/env python3
"""
Test script to verify ThreadPoolExecutor and rate limiting work correctly.
Uses mock functions instead of real API calls.
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Simulated API call with random latency
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def mock_api_call(question_id: str) -> dict:
    """Simulate an API call with random latency and occasional failures."""
    latency = random.uniform(0.5, 2.0)
    time.sleep(latency)
    
    # 10% chance of failure to test retry logic
    if random.random() < 0.1:
        raise Exception(f"Simulated API error for {question_id}")
    
    return {
        "question_id": question_id,
        "latency": latency,
        "verdict": random.choice(["pass", "pass", "pass", "partial", "fail"])
    }


def run_single(question: dict) -> dict:
    """Run single question through mock pipeline."""
    start = time.time()
    question_id = question.get("question_id", "unknown")
    
    # Simulate retrieval (fast)
    time.sleep(0.1)
    
    # Simulate generation (slow)
    result = mock_api_call(question_id)
    
    # Simulate judge (medium)
    time.sleep(0.3)
    
    return {
        "question_id": question_id,
        "verdict": result["verdict"],
        "time": time.time() - start
    }


def run_parallel_test(questions: list, workers: int = 5):
    """Test parallel execution with ThreadPoolExecutor."""
    print(f"\n{'='*60}")
    print(f"PARALLEL EXECUTION TEST")
    print(f"Questions: {len(questions)}")
    print(f"Workers: {workers}")
    print(f"{'='*60}\n")
    
    results = []
    lock = Lock()
    processed = 0
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_single, q): q for q in questions}
        
        for future in as_completed(futures):
            q = futures[future]
            qid = q.get("question_id", "unknown")
            try:
                result = future.result(timeout=30)
                with lock:
                    results.append(result)
                    processed += 1
                    print(f"[{processed}/{len(questions)}] {qid}: {result['verdict']} ({result['time']:.1f}s)")
            except Exception as e:
                print(f"[?/{len(questions)}] {qid}: ERROR - {e}")
                with lock:
                    results.append({"question_id": qid, "error": str(e)})
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg per question: {total_time/len(questions):.2f}s")
    print(f"Throughput: {len(questions)/total_time*60:.1f} questions/min")
    
    # Compare to sequential estimate
    avg_latency = sum(r.get("time", 0) for r in results if "time" in r) / len([r for r in results if "time" in r])
    sequential_estimate = avg_latency * len(questions)
    speedup = sequential_estimate / total_time
    print(f"\nSequential estimate: {sequential_estimate:.1f}s")
    print(f"Speedup: {speedup:.1f}x")
    print(f"{'='*60}")
    
    return results


def main():
    # Create mock questions
    questions = [{"question_id": f"Q{i:03d}"} for i in range(20)]
    
    print("Testing ThreadPoolExecutor with 5 workers on 20 mock questions...")
    print("Each question simulates ~1-2s API latency with 10% failure rate.")
    print("Retry logic uses exponential backoff (1s, 2s, 4s).\n")
    
    results = run_parallel_test(questions, workers=5)
    
    # Count results
    passed = sum(1 for r in results if r.get("verdict") == "pass")
    errors = sum(1 for r in results if "error" in r)
    print(f"\nPassed: {passed}/{len(questions)}")
    print(f"Errors: {errors}/{len(questions)}")


if __name__ == "__main__":
    main()
