#!/usr/bin/env python3
"""
Precision Test Suite for RAG Evaluation.

Tests the RAG system at different precision levels (5, 10, 15, 20, 25)
using the real orchestrator infrastructure with:
- Recall: 100 candidates
- Semantic/Keyword: 50-50 blend
- Google Ranking API reranking

Generates a comparison report showing performance at each precision level.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional

from langchain_google_vertexai import ChatVertexAI

from config import config
from orchestrator_client import OrchestratorClient


class PrecisionTester:
    """Run LLM-as-judge evaluation at different precision levels."""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize judge LLM (Gemini 2.5 Flash, temp=0 for deterministic)
        self.judge_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LLM_LOCATION,
            temperature=0.0,
        )
        
        # Initialize RAG LLM for answer generation (Gemini 2.5 Flash, temp=0)
        self.rag_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LLM_LOCATION,
            temperature=0.0,
        )
        
        # Retry settings
        self.max_retries = 3
        self.checkpoint_file = self.output_dir / "precision_test_checkpoint.json"
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate RAG answer from context."""
        prompt = f"""You are a technical assistant for SCADA/Solar/Electrical equipment.
Answer the question based ONLY on the provided context. Be specific and accurate.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        response = self.rag_llm.invoke(prompt)
        return response.content
    
    def load_qa_corpus(self, filename: str = "qa_corpus_200.json") -> list[dict]:
        """Load the Q&A corpus."""
        qa_path = self.output_dir / filename
        if not qa_path.exists():
            raise FileNotFoundError(f"Q&A corpus not found: {qa_path}")
        
        with open(qa_path, 'r') as f:
            return json.load(f)
    
    def judge_answer(self, question: str, ground_truth: str, rag_answer: str, context: str) -> dict:
        """Use LLM to judge the RAG answer quality."""
        judge_prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the RAG system's answer against the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG System Answer: {rag_answer}

Retrieved Context (for reference):
{context[:2000]}

Evaluate on these criteria (score 1-5, where 5 is best):

1. **Correctness**: Is the RAG answer factually correct compared to ground truth?
2. **Completeness**: Does the RAG answer cover all key points from ground truth?
3. **Faithfulness**: Is the RAG answer faithful to the retrieved context (no hallucinations)?
4. **Relevance**: Is the RAG answer relevant to the question asked?
5. **Clarity**: Is the RAG answer clear and well-structured?

Respond with JSON only:
{{"correctness": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "relevance": <1-5>, "clarity": <1-5>, "overall_score": <1-5>, "verdict": "pass|partial|fail"}}
"""
        
        try:
            response = self.judge_llm.invoke(judge_prompt)
            response_text = response.content.strip()
            
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            return json.loads(response_text)
        except Exception as e:
            return {
                "correctness": 0, "completeness": 0, "faithfulness": 0,
                "relevance": 0, "clarity": 0, "overall_score": 0,
                "verdict": "error", "error": str(e)
            }
    
    def _save_checkpoint(self, data: dict):
        """Save checkpoint to resume if interrupted."""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  [Checkpoint saved: {len(data.get('completed_results', {}))} precision levels]")
    
    def _load_checkpoint(self) -> dict:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"completed_results": {}, "in_progress": None}
    
    def _process_single_question(
        self,
        client: OrchestratorClient,
        qa: dict,
        precision: int,
    ) -> dict:
        """Process a single question with retry logic via orchestrator (front door)."""
        question = qa.get("question", "")
        ground_truth = qa.get("answer", qa.get("ground_truth", ""))
        
        for attempt in range(self.max_retries):
            try:
                question_start = time.time()
                
                # FRONT DOOR: Call orchestrator API (full pipeline)
                # This uses: Hybrid Search (recall) -> Google Ranking API (precision) -> LLM Generation
                retrieval_start = time.time()
                result = client.query(question, top_k=precision)
                retrieval_time = round(time.time() - retrieval_start, 3)
                
                # Extract answer and context from orchestrator response
                rag_answer = result.get("answer", "")
                sources = result.get("sources", [])
                
                # Build context string from sources for judge
                context_parts = []
                for i, source in enumerate(sources):
                    name = source.get("name", "Unknown")
                    snippet = source.get("snippet", "")
                    context_parts.append(f"[{i+1}] {name}: {snippet}")
                context = "\n\n".join(context_parts)
                
                generation_time = result.get("timing", {}).get("generation", 0)
                
                # Judge the answer
                judge_start = time.time()
                judgment = self.judge_answer(question, ground_truth, rag_answer, context)
                judge_time = round(time.time() - judge_start, 3)
                
                total_time = round(time.time() - question_start, 3)
                
                return {
                    "question": question,
                    "question_type": qa.get("question_type", "unknown"),
                    "ground_truth": ground_truth,
                    "rag_answer": rag_answer,
                    "answer_length": len(rag_answer),
                    "ground_truth_length": len(ground_truth),
                    "sources_count": precision,
                    "judgment": judgment,
                    "attempts": attempt + 1,
                    "timing": {
                        "retrieval_seconds": retrieval_time,
                        "generation_seconds": generation_time,
                        "judge_seconds": judge_time,
                        "total_seconds": total_time,
                    },
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f" [Retry {attempt + 2}/{self.max_retries}]", end="")
                    time.sleep(2)  # Brief pause before retry
                else:
                    return {
                        "question": question,
                        "error": str(e),
                        "attempts": self.max_retries,
                    }
        return {"question": question, "error": "Max retries exceeded"}
    
    def run_precision_test(
        self,
        client: OrchestratorClient,
        precision: int,
        qa_corpus: list[dict],
        max_questions: Optional[int] = None,
        existing_results: Optional[list] = None,
        parallel_workers: int = 1,
    ) -> dict:
        """Run evaluation at a specific precision level with checkpoint support."""
        print(f"\n{'='*60}")
        print(f"PRECISION TEST: top_k={precision} (workers={parallel_workers})")
        print(f"{'='*60}")
        
        if max_questions:
            qa_corpus = qa_corpus[:max_questions]
        
        # Resume from existing results if provided
        results = existing_results or []
        start_idx = len(results)
        
        if start_idx > 0:
            print(f"  Resuming from question {start_idx + 1}/{len(qa_corpus)}")
        
        start_time = time.time()
        remaining_corpus = qa_corpus[start_idx:]
        
        if parallel_workers > 1:
            # Parallel execution
            results.extend(self._run_parallel(client, remaining_corpus, precision, parallel_workers, start_idx, len(qa_corpus)))
        else:
            # Sequential execution
            for i, qa in enumerate(remaining_corpus, start=start_idx):
                print(f"  [{i+1}/{len(qa_corpus)}] {qa.get('question', '')[:50]}...", end=" ")
                
                result = self._process_single_question(client, qa, precision)
                results.append(result)
                
                if "judgment" in result:
                    verdict = result["judgment"].get("verdict", "unknown")
                    score = result["judgment"].get("overall_score", 0)
                    print(f"Score: {score}/5, Verdict: {verdict}")
                else:
                    print(f"Error: {result.get('error', 'unknown')}")
                
                # Save intermediate results every 10 questions
                if (i + 1) % 10 == 0:
                    self._save_intermediate_results(precision, results)
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        metrics["precision_level"] = precision
        metrics["elapsed_seconds"] = round(elapsed, 2)
        metrics["questions_evaluated"] = len(qa_corpus)
        metrics["parallel_workers"] = parallel_workers
        
        return {
            "precision": precision,
            "metrics": metrics,
            "results": results,
        }
    
    def _run_parallel(
        self,
        client: OrchestratorClient,
        qa_corpus: list[dict],
        precision: int,
        workers: int,
        start_idx: int,
        total: int,
    ) -> list[dict]:
        """Run questions in parallel using ThreadPoolExecutor."""
        results = [None] * len(qa_corpus)
        completed = 0
        
        def process_with_index(args):
            idx, qa = args
            return idx, self._process_single_question(client, qa, precision)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_with_index, (i, qa)): i 
                for i, qa in enumerate(qa_corpus)
            }
            
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                completed += 1
                
                # Progress update
                global_idx = start_idx + idx + 1
                if "judgment" in result:
                    verdict = result["judgment"].get("verdict", "unknown")
                    score = result["judgment"].get("overall_score", 0)
                    print(f"  [{global_idx}/{total}] Score: {score}/5, Verdict: {verdict}")
                else:
                    print(f"  [{global_idx}/{total}] Error: {result.get('error', 'unknown')[:50]}")
                
                # Save checkpoint every 20 completed
                if completed % 20 == 0:
                    valid_results = [r for r in results if r is not None]
                    self._save_intermediate_results(precision, valid_results)
        
        return results
    
    def _save_intermediate_results(self, precision: int, results: list):
        """Save intermediate results for a precision level."""
        intermediate_file = self.output_dir / f"precision_{precision}_intermediate.json"
        with open(intermediate_file, 'w') as f:
            json.dump({
                "precision": precision,
                "results_count": len(results),
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2)
        print(f"  [Saved {len(results)} results to {intermediate_file.name}]")
    
    def _calculate_metrics(self, results: list[dict]) -> dict:
        """Calculate aggregate metrics from results."""
        valid_results = [r for r in results if "judgment" in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        metrics = {}
        for metric in ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]:
            values = [r["judgment"].get(metric, 0) for r in valid_results]
            metrics[metric] = round(sum(values) / len(values), 2) if values else 0
        
        # Verdict counts
        verdicts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
        for r in valid_results:
            v = r["judgment"].get("verdict", "error")
            if v in verdicts:
                verdicts[v] += 1
            else:
                verdicts["error"] += 1
        
        metrics["verdict_counts"] = verdicts
        metrics["pass_rate"] = round((verdicts["pass"] + verdicts["partial"]) / len(valid_results), 3) if valid_results else 0
        
        # Timing statistics
        results_with_timing = [r for r in valid_results if "timing" in r]
        if results_with_timing:
            retrieval_times = [r["timing"]["retrieval_seconds"] for r in results_with_timing]
            generation_times = [r["timing"]["generation_seconds"] for r in results_with_timing]
            judge_times = [r["timing"]["judge_seconds"] for r in results_with_timing]
            total_times = [r["timing"]["total_seconds"] for r in results_with_timing]
            
            metrics["timing"] = {
                "avg_retrieval_seconds": round(sum(retrieval_times) / len(retrieval_times), 3),
                "avg_generation_seconds": round(sum(generation_times) / len(generation_times), 3),
                "avg_judge_seconds": round(sum(judge_times) / len(judge_times), 3),
                "avg_total_seconds": round(sum(total_times) / len(total_times), 3),
                "total_time_seconds": round(sum(total_times), 2),
            }
        
        # Answer length statistics
        results_with_length = [r for r in valid_results if "answer_length" in r]
        if results_with_length:
            answer_lengths = [r["answer_length"] for r in results_with_length]
            gt_lengths = [r["ground_truth_length"] for r in results_with_length]
            
            metrics["answer_stats"] = {
                "avg_answer_length": round(sum(answer_lengths) / len(answer_lengths), 0),
                "min_answer_length": min(answer_lengths),
                "max_answer_length": max(answer_lengths),
                "avg_ground_truth_length": round(sum(gt_lengths) / len(gt_lengths), 0),
            }
        
        return metrics
    
    def run_all_precision_tests(
        self,
        precision_levels: Optional[list[int]] = None,
        max_questions: Optional[int] = None,
        parallel_workers: int = 1,
    ) -> dict:
        """Run evaluation at all precision levels and generate comparison report.
        
        Args:
            precision_levels: List of precision values to test (default: [5, 10, 15, 20, 25])
            max_questions: Limit number of questions (default: all)
            parallel_workers: Number of parallel workers (default: 1 = sequential)
                             Recommended: 4-8 for faster execution
        """
        precision_levels = precision_levels or config.PRECISION_LEVELS
        
        print("="*60)
        print("PRECISION TEST SUITE (via Orchestrator - Front Door)")
        print("="*60)
        print(f"Precision levels to test: {precision_levels}")
        print(f"Job ID: {config.ORCHESTRATOR_JOB_ID}")
        print(f"Orchestrator URL: {config.ORCHESTRATOR_API_URL}")
        print(f"Parallel workers: {parallel_workers}")
        
        # Create orchestrator client (FRONT DOOR)
        client = OrchestratorClient()
        
        # Check if orchestrator is running
        if not client.health_check():
            raise ConnectionError(
                "Orchestrator is not running!\n"
                "Start it with:\n"
                "  cd /Users/scottmacon/Documents/GitHub/sm-dev-01\n"
                "  source .venv/bin/activate\n"
                "  python -m uvicorn services.api.app:app --host 0.0.0.0 --port 8000"
            )
        print("✓ Orchestrator is running (front door)")
        
        # Load Q&A corpus
        qa_corpus = self.load_qa_corpus()
        print(f"✓ Loaded {len(qa_corpus)} Q&A pairs")
        
        # Run tests at each precision level
        all_results = {}
        for precision in precision_levels:
            test_result = self.run_precision_test(
                client, precision, qa_corpus, max_questions,
                parallel_workers=parallel_workers,
            )
            all_results[precision] = test_result
        
        # Generate comparison report
        report = self._generate_comparison_report(all_results, precision_levels)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"precision_test_results_{timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "recall": config.RECALL_TOP_K,
                    "semantic_weight": config.SEMANTIC_WEIGHT,
                    "enable_reranking": config.ENABLE_RERANKING,
                    "ranking_model": config.RANKING_MODEL,
                    "precision_levels": precision_levels,
                },
                "comparison": report,
                "detailed_results": {str(k): v for k, v in all_results.items()},
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Print comparison report
        self._print_comparison_report(report, precision_levels)
        
        return report
    
    def _generate_comparison_report(self, all_results: dict, precision_levels: list[int]) -> dict:
        """Generate comparison report across precision levels."""
        report = {
            "by_precision": {},
            "best_precision": {},
            "recommendation": "",
        }
        
        for precision in precision_levels:
            if precision in all_results:
                metrics = all_results[precision]["metrics"]
                timing = metrics.get("timing", {})
                answer_stats = metrics.get("answer_stats", {})
                
                report["by_precision"][precision] = {
                    "overall_score": metrics.get("overall_score", 0),
                    "pass_rate": metrics.get("pass_rate", 0),
                    "correctness": metrics.get("correctness", 0),
                    "completeness": metrics.get("completeness", 0),
                    "faithfulness": metrics.get("faithfulness", 0),
                    "relevance": metrics.get("relevance", 0),
                    "clarity": metrics.get("clarity", 0),
                    "avg_time_per_question": timing.get("avg_total_seconds", 0),
                    "total_time_seconds": timing.get("total_time_seconds", 0),
                    "avg_answer_length": answer_stats.get("avg_answer_length", 0),
                }
        
        # Find best precision for each metric
        for metric in ["overall_score", "pass_rate", "correctness", "completeness", "faithfulness"]:
            best_precision = max(
                precision_levels,
                key=lambda p: report["by_precision"].get(p, {}).get(metric, 0)
            )
            best_value = report["by_precision"].get(best_precision, {}).get(metric, 0)
            report["best_precision"][metric] = {"precision": best_precision, "value": best_value}
        
        # Generate recommendation
        best_overall = report["best_precision"]["overall_score"]["precision"]
        best_pass_rate = report["best_precision"]["pass_rate"]["precision"]
        
        if best_overall == best_pass_rate:
            report["recommendation"] = f"Recommended precision: {best_overall} (best for both overall score and pass rate)"
        else:
            report["recommendation"] = f"Trade-off: precision {best_overall} for best overall score, precision {best_pass_rate} for best pass rate"
        
        return report
    
    def _print_comparison_report(self, report: dict, precision_levels: list[int]):
        """Print the comparison report."""
        print("\n" + "="*120)
        print("PRECISION COMPARISON REPORT")
        print("="*120)
        
        # Header - Quality Metrics
        print(f"\n{'Precision':<10}", end="")
        for metric in ["Overall", "Pass%", "Correct", "Complete", "Faithful", "Relevant", "Clarity"]:
            print(f"{metric:<10}", end="")
        print(f"{'Avg Time':<10}{'Total Time':<12}{'Avg Len':<10}")
        print("-"*120)
        
        # Data rows
        for precision in precision_levels:
            data = report["by_precision"].get(precision, {})
            print(f"{precision:<10}", end="")
            print(f"{data.get('overall_score', 0):<10.2f}", end="")
            print(f"{data.get('pass_rate', 0)*100:<10.1f}", end="")
            print(f"{data.get('correctness', 0):<10.2f}", end="")
            print(f"{data.get('completeness', 0):<10.2f}", end="")
            print(f"{data.get('faithfulness', 0):<10.2f}", end="")
            print(f"{data.get('relevance', 0):<10.2f}", end="")
            print(f"{data.get('clarity', 0):<10.2f}", end="")
            print(f"{data.get('avg_time_per_question', 0):<10.2f}", end="")
            total_time = data.get('total_time_seconds', 0)
            print(f"{total_time/60:<12.1f}min", end="")
            print(f"{int(data.get('avg_answer_length', 0)):<10}")
        
        print("-"*120)
        
        # Best values
        print("\nBest Precision by Metric:")
        for metric, info in report["best_precision"].items():
            print(f"  {metric}: precision={info['precision']} (value={info['value']:.2f})")
        
        print(f"\n{report['recommendation']}")
        print("="*120)


def main():
    """Run the precision test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run precision tests for RAG evaluation")
    parser.add_argument("--workers", "-w", type=int, default=1,
                        help="Number of parallel workers (default: 1, recommended: 4-8)")
    parser.add_argument("--max-questions", "-n", type=int, default=None,
                        help="Limit number of questions (default: all)")
    parser.add_argument("--precision-levels", "-p", type=int, nargs="+", 
                        default=config.PRECISION_LEVELS,
                        help="Precision levels to test (default: 5 10 15 20 25)")
    args = parser.parse_args()
    
    tester = PrecisionTester()
    
    # Run all precision tests
    report = tester.run_all_precision_tests(
        precision_levels=args.precision_levels,
        max_questions=args.max_questions,
        parallel_workers=args.workers,
    )


if __name__ == "__main__":
    main()
