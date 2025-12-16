#!/usr/bin/env python3
"""
Embedding Comparison Test Suite.

Tests RAG retrieval quality across different embedding configurations:
1. text-embedding-005 (no task type)
2. gemini-embedding-001 (no task type)
3. gemini-embedding-001 with RETRIEVAL_QUERY task type

This validates the hypothesis that using proper task types improves retrieval quality.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from langchain_google_vertexai import ChatVertexAI

from config import config


# The 3 index configurations to test
EMBEDDING_CONFIGS = [
    {
        "job_id": "bfai__evalv3_text005",
        "name": "text-embedding-005",
        "description": "text-embedding-005, no task type",
        "embedding_model": "text-embedding-005",
        "task_type": None,
    },
    {
        "job_id": "bfai__eval66_gemini_no_tt",
        "name": "gemini-no-tasktype",
        "description": "gemini-embedding-001, 768 dim, NO task type",
        "embedding_model": "gemini-embedding-001",
        "task_type": None,
    },
    {
        "job_id": "bfai__eval66_gemini_tt",
        "name": "gemini-with-tasktype",
        "description": "gemini-embedding-001, 768 dim, WITH RETRIEVAL_QUERY task type",
        "embedding_model": "gemini-embedding-001",
        "task_type": "RETRIEVAL_QUERY",
    },
]


class EmbeddingComparisonTester:
    """Test RAG quality across different embedding configurations."""
    
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
        
        # Retry settings
        self.max_retries = 3
        
    def load_qa_corpus(self, filename: str = "qa_corpus_200.json") -> List[Dict]:
        """Load the Q&A corpus."""
        qa_path = self.output_dir / filename
        if not qa_path.exists():
            raise FileNotFoundError(f"Q&A corpus not found: {qa_path}")
        
        with open(qa_path, 'r') as f:
            return json.load(f)
    
    def judge_answer(
        self,
        question: str,
        ground_truth: str,
        rag_answer: str,
        context: str,
    ) -> Dict[str, Any]:
        """Judge the RAG answer against ground truth."""
        prompt = f"""You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system.
Evaluate the RAG answer against the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG System Answer: {rag_answer}

Retrieved Context (for reference):
{context[:2000]}

Rate the RAG answer on these criteria (1-5 scale):
1. Correctness: Does it match the ground truth factually?
2. Completeness: Does it cover all key points from ground truth?
3. Faithfulness: Is it grounded in the retrieved context?
4. Relevance: Does it directly answer the question?
5. Clarity: Is it well-written and clear?

Respond in JSON format:
{{
    "correctness": <1-5>,
    "completeness": <1-5>,
    "faithfulness": <1-5>,
    "relevance": <1-5>,
    "clarity": <1-5>,
    "overall_score": <1-5>,
    "verdict": "pass" | "partial" | "fail",
    "explanation": "<brief explanation>"
}}

Rules:
- "pass" = overall_score >= 4
- "partial" = overall_score == 3
- "fail" = overall_score <= 2
"""
        response = self.judge_llm.invoke(prompt)
        
        try:
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except Exception as e:
            return {
                "error": str(e),
                "raw_response": response.content[:500],
                "overall_score": 0,
                "verdict": "error",
            }
    
    def _call_orchestrator(self, job_id: str, query: str, top_k: int = 10) -> Dict:
        """Call the orchestrator API with a specific job_id."""
        import requests
        
        url = f"{config.ORCHESTRATOR_API_URL}/query"
        payload = {
            "query": query,
            "job_id": job_id,
            "top_k": top_k,
            "enable_hybrid": True,
            "enable_reranking": True,
        }
        
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    
    def _process_single_question(
        self,
        job_id: str,
        qa: Dict,
        precision: int = 10,
    ) -> Dict:
        """Process a single question with retry logic."""
        question = qa.get("question", "")
        ground_truth = qa.get("answer", qa.get("ground_truth", ""))
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Call orchestrator
                result = self._call_orchestrator(job_id, question, top_k=precision)
                retrieval_time = round(time.time() - start_time, 3)
                
                # Extract answer and context
                rag_answer = result.get("answer", "")
                sources = result.get("sources", [])
                
                # Build context string
                context_parts = []
                for i, source in enumerate(sources):
                    name = source.get("name", "Unknown")
                    snippet = source.get("snippet", "")
                    context_parts.append(f"[{i+1}] {name}: {snippet}")
                context = "\n\n".join(context_parts)
                
                # Judge the answer
                judge_start = time.time()
                judgment = self.judge_answer(question, ground_truth, rag_answer, context)
                judge_time = round(time.time() - judge_start, 3)
                
                total_time = round(time.time() - start_time, 3)
                
                return {
                    "question": question,
                    "ground_truth": ground_truth,
                    "rag_answer": rag_answer,
                    "answer_length": len(rag_answer),
                    "sources_count": len(sources),
                    "judgment": judgment,
                    "attempts": attempt + 1,
                    "timing": {
                        "retrieval_seconds": retrieval_time,
                        "judge_seconds": judge_time,
                        "total_seconds": total_time,
                    },
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f" [Retry {attempt + 2}/{self.max_retries}]", end="")
                    time.sleep(2)
                else:
                    return {
                        "question": question,
                        "error": str(e),
                        "attempts": self.max_retries,
                    }
        return {"question": question, "error": "Max retries exceeded"}
    
    def test_single_config(
        self,
        embedding_config: Dict,
        qa_corpus: List[Dict],
        max_questions: Optional[int] = None,
        parallel_workers: int = 1,
        precision: int = 10,
    ) -> Dict:
        """Test a single embedding configuration."""
        job_id = embedding_config["job_id"]
        name = embedding_config["name"]
        
        print(f"\n{'='*60}")
        print(f"TESTING: {name}")
        print(f"  Job ID: {job_id}")
        print(f"  Model: {embedding_config['embedding_model']}")
        print(f"  Task Type: {embedding_config['task_type']}")
        print(f"  Workers: {parallel_workers}")
        print(f"{'='*60}")
        
        if max_questions:
            qa_corpus = qa_corpus[:max_questions]
        
        results = []
        start_time = time.time()
        
        if parallel_workers > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(self._process_single_question, job_id, qa, precision): i
                    for i, qa in enumerate(qa_corpus)
                }
                
                for future in as_completed(futures):
                    idx = futures[future]
                    result = future.result()
                    results.append(result)
                    
                    if "judgment" in result:
                        score = result["judgment"].get("overall_score", 0)
                        verdict = result["judgment"].get("verdict", "unknown")
                        print(f"  [{len(results)}/{len(qa_corpus)}] Score: {score}/5, Verdict: {verdict}")
                    else:
                        print(f"  [{len(results)}/{len(qa_corpus)}] Error: {result.get('error', 'unknown')[:50]}")
        else:
            # Sequential execution
            for i, qa in enumerate(qa_corpus):
                print(f"  [{i+1}/{len(qa_corpus)}] {qa.get('question', '')[:50]}...", end=" ")
                result = self._process_single_question(job_id, qa, precision)
                results.append(result)
                
                if "judgment" in result:
                    score = result["judgment"].get("overall_score", 0)
                    verdict = result["judgment"].get("verdict", "unknown")
                    print(f"Score: {score}/5, Verdict: {verdict}")
                else:
                    print(f"Error: {result.get('error', 'unknown')[:50]}")
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        valid_results = [r for r in results if "judgment" in r]
        if valid_results:
            metrics = {
                "total_questions": len(qa_corpus),
                "successful": len(valid_results),
                "failed": len(qa_corpus) - len(valid_results),
                "overall_score": round(sum(r["judgment"]["overall_score"] for r in valid_results) / len(valid_results), 2),
                "pass_rate": round(sum(1 for r in valid_results if r["judgment"]["verdict"] == "pass") / len(valid_results), 2),
                "correctness": round(sum(r["judgment"].get("correctness", 0) for r in valid_results) / len(valid_results), 2),
                "completeness": round(sum(r["judgment"].get("completeness", 0) for r in valid_results) / len(valid_results), 2),
                "faithfulness": round(sum(r["judgment"].get("faithfulness", 0) for r in valid_results) / len(valid_results), 2),
                "relevance": round(sum(r["judgment"].get("relevance", 0) for r in valid_results) / len(valid_results), 2),
                "clarity": round(sum(r["judgment"].get("clarity", 0) for r in valid_results) / len(valid_results), 2),
                "avg_time_seconds": round(sum(r["timing"]["total_seconds"] for r in valid_results) / len(valid_results), 2),
                "total_time_seconds": round(elapsed, 2),
            }
        else:
            metrics = {"error": "No valid results"}
        
        return {
            "config": embedding_config,
            "metrics": metrics,
            "results": results,
        }
    
    def run_comparison(
        self,
        max_questions: Optional[int] = None,
        parallel_workers: int = 8,
        precision: int = 10,
        configs: Optional[List[Dict]] = None,
    ) -> Dict:
        """Run comparison across all embedding configurations."""
        configs = configs or EMBEDDING_CONFIGS
        
        print("="*70)
        print("EMBEDDING COMPARISON TEST")
        print("="*70)
        print(f"Configurations to test: {len(configs)}")
        print(f"Parallel workers: {parallel_workers}")
        print(f"Precision (top_k): {precision}")
        print(f"Orchestrator URL: {config.ORCHESTRATOR_API_URL}")
        
        # Check orchestrator health
        import requests
        try:
            resp = requests.get(f"{config.ORCHESTRATOR_API_URL}/health", timeout=5)
            if resp.status_code != 200:
                raise ConnectionError("Orchestrator health check failed")
            print("✓ Orchestrator is running")
        except Exception as e:
            raise ConnectionError(
                f"Orchestrator is not running at {config.ORCHESTRATOR_API_URL}\n"
                "Start it with:\n"
                "  cd /Users/scottmacon/Documents/GitHub/sm-dev-01\n"
                "  source .venv/bin/activate\n"
                "  python3 -m uvicorn services.api.app:app --host 0.0.0.0 --port 8000"
            )
        
        # Load Q&A corpus
        qa_corpus = self.load_qa_corpus()
        print(f"✓ Loaded {len(qa_corpus)} Q&A pairs")
        
        if max_questions:
            print(f"  (Testing with {max_questions} questions)")
        
        # Test each configuration
        all_results = {}
        for cfg in configs:
            result = self.test_single_config(
                cfg, qa_corpus, max_questions, parallel_workers, precision
            )
            all_results[cfg["name"]] = result
        
        # Generate comparison report
        report = self._generate_report(all_results)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"embedding_comparison_{timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "max_questions": max_questions,
                    "parallel_workers": parallel_workers,
                    "precision": precision,
                },
                "comparison": report,
                "detailed_results": all_results,
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        
        # Print comparison report
        self._print_report(report)
        
        return report
    
    def _generate_report(self, all_results: Dict) -> Dict:
        """Generate comparison report."""
        report = {"by_config": {}, "winner": None}
        
        best_score = 0
        best_config = None
        
        for name, data in all_results.items():
            metrics = data.get("metrics", {})
            report["by_config"][name] = {
                "description": data["config"]["description"],
                "embedding_model": data["config"]["embedding_model"],
                "task_type": data["config"]["task_type"],
                "overall_score": metrics.get("overall_score", 0),
                "pass_rate": metrics.get("pass_rate", 0),
                "correctness": metrics.get("correctness", 0),
                "completeness": metrics.get("completeness", 0),
                "faithfulness": metrics.get("faithfulness", 0),
                "relevance": metrics.get("relevance", 0),
                "clarity": metrics.get("clarity", 0),
                "avg_time_seconds": metrics.get("avg_time_seconds", 0),
            }
            
            score = metrics.get("overall_score", 0)
            if score > best_score:
                best_score = score
                best_config = name
        
        report["winner"] = {
            "name": best_config,
            "score": best_score,
        }
        
        return report
    
    def _print_report(self, report: Dict):
        """Print comparison report."""
        print("\n" + "="*90)
        print("EMBEDDING COMPARISON RESULTS")
        print("="*90)
        
        # Header
        print(f"{'Config':<25} {'Score':<8} {'Pass%':<8} {'Correct':<8} {'Complete':<8} {'Faithful':<8} {'Time':<8}")
        print("-"*90)
        
        for name, data in report["by_config"].items():
            print(f"{name:<25} ", end="")
            print(f"{data['overall_score']:<8.2f} ", end="")
            print(f"{data['pass_rate']*100:<8.0f} ", end="")
            print(f"{data['correctness']:<8.2f} ", end="")
            print(f"{data['completeness']:<8.2f} ", end="")
            print(f"{data['faithfulness']:<8.2f} ", end="")
            print(f"{data['avg_time_seconds']:<8.2f}")
        
        print("-"*90)
        
        winner = report.get("winner", {})
        print(f"\n🏆 WINNER: {winner.get('name')} (score: {winner.get('score', 0):.2f})")
        print("="*90)


def main():
    """Run the embedding comparison test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare embedding configurations for RAG")
    parser.add_argument("--workers", "-w", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--max-questions", "-n", type=int, default=None,
                        help="Limit number of questions (default: all)")
    parser.add_argument("--precision", "-p", type=int, default=10,
                        help="Precision level / top_k (default: 10)")
    args = parser.parse_args()
    
    tester = EmbeddingComparisonTester()
    
    report = tester.run_comparison(
        max_questions=args.max_questions,
        parallel_workers=args.workers,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
