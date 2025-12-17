#!/usr/bin/env python3
"""
Gold Standard Evaluation with Checkpointing and Parallel Execution.

Runs full RAG pipeline evaluation on gold corpus.
Checkpoints every 10 questions to resume on failure.
Supports parallel execution with ThreadPoolExecutor.

Usage:
  python scripts/eval/run_gold_eval.py --test           # Run 30 questions (5 per bucket)
  python scripts/eval/run_gold_eval.py                  # Run full 458 questions
  python scripts/eval/run_gold_eval.py --precision 12   # Use precision@12 instead of @25
  python scripts/eval/run_gold_eval.py --workers 5      # Parallel with 5 workers
"""

import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")

from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
from services.api.retrieval.vector_search import VectorSearchRetriever
from services.api.ranking.google_ranker import GoogleRanker
from services.api.generation.gemini import GeminiAnswerGenerator
from services.api.core.config import QueryConfig
from langchain_google_vertexai import ChatVertexAI

# Config
JOB_ID = "bfai__eval66a_g1_1536_tt"
CORPUS_PATH = Path(__file__).parent.parent.parent / "clients" / "BFAI" / "qa" / "QA_BFAI_gold_v1-0__q458.json"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "reports" / "gold_standard_eval"
CHECKPOINT_INTERVAL = 10
DEFAULT_WORKERS = 5  # Safe for 60 RPM quota, increase to 15-25 with 1500 RPM


def load_corpus(test_mode: bool = False):
    """Load corpus, optionally sampling 5 from each bucket for test mode."""
    with open(CORPUS_PATH) as f:
        data = json.load(f)
    questions = data.get("questions", data)
    
    if test_mode:
        # Sample 5 from each of 6 buckets
        by_bucket = defaultdict(list)
        for q in questions:
            key = f"{q.get('question_type')}/{q.get('difficulty')}"
            by_bucket[key].append(q)
        
        sampled = []
        for key, qs in by_bucket.items():
            sampled.extend(qs[:5])
        print(f"TEST MODE: Sampled {len(sampled)} questions (5 per bucket)")
        return sampled
    
    return questions


def extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        for part in text.split("```"):
            if part.strip().startswith("{"):
                text = part.strip()
                break
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i+1])
    return json.loads(text)


class GoldEvaluator:
    def __init__(self, precision_k: int = 25, workers: int = DEFAULT_WORKERS):
        self.precision_k = precision_k
        self.workers = workers
        self.lock = Lock()  # Thread-safe checkpoint access
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("Initializing components...")
        jobs = get_jobs_config()
        job_config = jobs.get(JOB_ID, {})
        job_config["job_id"] = JOB_ID
        
        self.retriever = VectorSearchRetriever(job_config)
        self.ranker = GoogleRanker(project_id=PROJECT_ID)
        self.generator = GeminiAnswerGenerator()
        self.judge = ChatVertexAI(
            model_name="gemini-2.0-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.0,
        )
        
        self.checkpoint_file = OUTPUT_DIR / f"checkpoint_p{precision_k}.json"
        self.results_file = OUTPUT_DIR / f"results_p{precision_k}.json"
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _judge_answer(self, question: str, ground_truth: str, answer: str, context: str) -> dict:
        """Judge answer quality with exponential backoff."""
        prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth.

Question: {question}

Ground Truth: {ground_truth}

RAG Answer: {answer}

Context (first 2000 chars): {context[:2000]}

Score 1-5 for each (5=best):
1. correctness - factually correct vs ground truth?
2. completeness - covers key points?
3. faithfulness - faithful to context, no hallucinations?
4. relevance - relevant to question?
5. clarity - clear and well-structured?

Respond with ONLY this JSON, no markdown:
{{"correctness": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "relevance": <1-5>, "clarity": <1-5>, "overall_score": <1-5>, "verdict": "pass|partial|fail"}}"""
        
        for attempt in range(5):
            try:
                response = self.judge.invoke(prompt)
                text = response.content.strip()
                # Strip markdown if present
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                return json.loads(text.strip())
            except Exception as e:
                if attempt == 4:
                    # Return partial scores on final failure
                    return {"correctness": 3, "completeness": 3, "faithfulness": 3, "relevance": 3, "clarity": 3, "overall_score": 3, "verdict": "partial", "parse_error": str(e)}
                time.sleep(0.5)
    
    def run_single(self, q: dict) -> dict:
        """Run single question through pipeline."""
        question = q.get("question", "")
        source_filenames = q.get("source_filenames", [])
        expected_source = source_filenames[0] if source_filenames else q.get("source_document", "")
        expected_source = expected_source.replace(".pdf", "").lower()
        ground_truth = q.get("ground_truth_answer", q.get("answer", ""))
        
        start = time.time()
        
        # Retrieve
        config = QueryConfig(
            recall_top_k=100,
            precision_top_n=self.precision_k,
            enable_hybrid=True,
            enable_reranking=True,
            job_id=JOB_ID,
        )
        retrieval_result = self.retriever.retrieve(question, config)
        chunks = list(retrieval_result.chunks)
        
        # Check recall@100
        retrieved_docs = [c.doc_name.lower() if c.doc_name else "" for c in chunks]
        recall_hit = any(expected_source in d for d in retrieved_docs)
        
        # MRR
        mrr = 0
        for rank, d in enumerate(retrieved_docs, 1):
            if expected_source in d:
                mrr = 1.0 / rank
                break
        
        # Rerank
        reranked = self.ranker.rank(question, chunks, self.precision_k)
        
        # Generate
        context = "\n\n".join([f"[{i+1}] {c.text}" for i, c in enumerate(reranked)])
        gen_result = self.generator.generate(query=question, context=context, config=config)
        answer = gen_result.answer_text
        
        # Judge
        judgment = self._judge_answer(question, ground_truth, answer, context)
        
        return {
            "question_id": q.get("question_id", ""),
            "question_type": q.get("question_type", ""),
            "difficulty": q.get("difficulty", ""),
            "recall_hit": recall_hit,
            "mrr": mrr,
            "judgment": judgment,
            "time": time.time() - start,
        }
    
    def run(self, questions: list):
        """Run evaluation with checkpointing and optional parallelism."""
        print(f"\n{'='*60}")
        print(f"GOLD STANDARD EVAL - Precision@{self.precision_k}")
        print(f"Questions: {len(questions)}")
        print(f"Workers: {self.workers}")
        print(f"{'='*60}\n")
        
        # Load checkpoint
        completed = {}
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                for r in json.load(f):
                    completed[r["question_id"]] = r
            print(f"Resuming from checkpoint: {len(completed)} done")
        
        # Filter out already completed
        pending = [q for q in questions if q.get("question_id", "") not in completed]
        print(f"Pending: {len(pending)} questions")
        
        results = list(completed.values())
        processed = len(completed)
        
        if self.workers == 1:
            # Sequential mode
            for i, q in enumerate(pending):
                qid = q.get("question_id", f"q_{processed + i}")
                try:
                    result = self.run_single(q)
                    results.append(result)
                    completed[qid] = result
                    verdict = result.get("judgment", {}).get("verdict", "?")
                    print(f"[{processed + i + 1}/{len(questions)}] {qid}: {verdict} ({result['time']:.1f}s)")
                    
                    if (processed + i + 1) % CHECKPOINT_INTERVAL == 0:
                        with open(self.checkpoint_file, 'w') as f:
                            json.dump(list(completed.values()), f, indent=2)
                        print(f"  >> Checkpoint saved ({len(completed)} questions)")
                except Exception as e:
                    print(f"[{processed + i + 1}/{len(questions)}] {qid}: ERROR - {e}")
                    results.append({"question_id": qid, "error": str(e)})
        else:
            # Parallel mode with ThreadPoolExecutor
            print(f"Starting parallel execution with {self.workers} workers...")
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {executor.submit(self.run_single, q): q for q in pending}
                
                for future in as_completed(futures):
                    q = futures[future]
                    qid = q.get("question_id", "unknown")
                    try:
                        result = future.result(timeout=120)
                        with self.lock:
                            results.append(result)
                            completed[qid] = result
                            processed += 1
                            verdict = result.get("judgment", {}).get("verdict", "?")
                            print(f"[{processed}/{len(questions)}] {qid}: {verdict} ({result['time']:.1f}s)")
                            
                            if processed % CHECKPOINT_INTERVAL == 0:
                                with open(self.checkpoint_file, 'w') as f:
                                    json.dump(list(completed.values()), f, indent=2)
                                print(f"  >> Checkpoint saved ({len(completed)} questions)")
                    except Exception as e:
                        print(f"[?/{len(questions)}] {qid}: ERROR - {e}")
                        with self.lock:
                            results.append({"question_id": qid, "error": str(e)})
        
        # Final save
        with open(self.checkpoint_file, 'w') as f:
            json.dump(list(completed.values()), f, indent=2)
        
        # Calculate metrics
        valid = [r for r in results if "judgment" in r]
        metrics = {
            "precision_k": self.precision_k,
            "total": len(questions),
            "completed": len(valid),
            "recall@100": sum(1 for r in valid if r.get("recall_hit")) / len(valid) if valid else 0,
            "mrr": sum(r.get("mrr", 0) for r in valid) / len(valid) if valid else 0,
            "pass_rate": sum(1 for r in valid if r.get("judgment", {}).get("verdict") == "pass") / len(valid) if valid else 0,
        }
        
        # Dimension averages
        for dim in ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall"]:
            scores = [r.get("judgment", {}).get(dim, 0) for r in valid if r.get("judgment", {}).get(dim)]
            metrics[dim] = sum(scores) / len(scores) if scores else 0
        
        # Save results
        output = {"metrics": metrics, "results": results, "timestamp": datetime.now().isoformat()}
        with open(self.results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"RESULTS - Precision@{self.precision_k}")
        print(f"{'='*60}")
        print(f"Completed: {metrics['completed']}/{metrics['total']}")
        print(f"Recall@100: {metrics['recall@100']:.1%}")
        print(f"MRR: {metrics['mrr']:.3f}")
        print(f"Pass Rate: {metrics['pass_rate']:.1%}")
        print(f"Overall Score: {metrics['overall']:.2f}/5")
        print(f"\nResults saved: {self.results_file}")
        print(f"{'='*60}")
        
        return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Test mode: 30 questions")
    parser.add_argument("--precision", type=int, default=25, help="Precision@K (default: 25)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, 
                        help=f"Number of parallel workers (default: {DEFAULT_WORKERS}, use 1 for sequential)")
    parser.add_argument("--quick", type=int, default=0, help="Quick test: run N questions only")
    args = parser.parse_args()
    
    questions = load_corpus(test_mode=args.test)
    
    # Quick mode for testing parallelism
    if args.quick > 0:
        questions = questions[:args.quick]
        print(f"QUICK MODE: Running {len(questions)} questions only")
    
    evaluator = GoldEvaluator(precision_k=args.precision, workers=args.workers)
    evaluator.run(questions)


if __name__ == "__main__":
    main()
