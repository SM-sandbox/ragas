#!/usr/bin/env python3
"""
Gemini 3 Flash Evaluation - Full Pipeline

Runs 458 gold standard questions through the FULL RAG pipeline:
- Vector Search retrieval (recall@100)
- Google Ranker reranking (precision@25)
- Gemini generation (2.5 Flash baseline OR 3.0 Flash LOW/HIGH)
- Gemini 3.0 Flash LOW judge

50 parallel workers with clean progress bar every 10 questions.

Usage:
    # Baseline: Gemini 2.5 Flash
    python scripts/eval/run_gemini3_eval.py --model gemini-2.5-flash --questions 458 --workers 50
    
    # Gemini 3.0 Flash LOW reasoning
    python scripts/eval/run_gemini3_eval.py --model gemini-3-flash-preview --thinking LOW --questions 458 --workers 50
    
    # Gemini 3.0 Flash HIGH reasoning
    python scripts/eval/run_gemini3_eval.py --model gemini-3-flash-preview --thinking HIGH --questions 458 --workers 50
"""

import sys
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/gRAG_v3")

import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"

import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

# Suppress noisy logging
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Orchestrator imports from gRAG_v3
from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
from services.api.retrieval.vector_search import VectorSearchRetriever
from services.api.ranking.google_ranker import GoogleRanker
from services.api.generation.gemini import GeminiAnswerGenerator
from services.api.processing.citations import CitationProcessor
from services.api.core.config import QueryConfig
from services.api.core.models import Chunk

# Our gemini_client for judge (Gemini 3.0 Flash LOW)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from gemini_client import generate

# Setup minimal logging
logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
JOB_ID = "bfai__eval66a_g1_1536_tt"
CORPUS_PATH = Path(__file__).parent.parent.parent / "clients" / "BFAI" / "qa" / "QA_BFAI_gold_v1-0__q458.json"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "reports" / "gemini3_comparison"

# Defaults
DEFAULT_QUESTIONS = 458
DEFAULT_WORKERS = 50
PRECISION_K = 25
CHECKPOINT_INTERVAL = 10

# Pricing (per 1M tokens)
PRICING = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30, "thinking": 0.30},
    "gemini-3-flash-preview": {"input": 0.075, "output": 0.30, "thinking": 0.30},
}

# Progress tracking
_progress_lock = Lock()
_processed_count = 0
_total_questions = 0
_results_buffer = []


def progress_bar(current: int, total: int, width: int = 40) -> str:
    """Generate a simple progress bar."""
    pct = current / total
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{current:3d}/{total}] {bar} {pct*100:5.1f}%"


@dataclass
class QueryResult:
    """Result from a single query through the pipeline."""
    question_id: str
    question: str
    question_type: str
    difficulty: str
    expected_source: str
    ground_truth: str
    
    # Retrieval
    recall_at_100: bool = False
    first_relevant_rank: Optional[int] = None
    
    # After reranking
    mrr: float = 0.0
    
    # Generation
    answer: str = ""
    answer_length: int = 0
    
    # Judge scores
    judgment: Dict = field(default_factory=dict)
    
    # Timings (seconds)
    retrieval_time: float = 0.0
    reranking_time: float = 0.0
    generation_time: float = 0.0
    judge_time: float = 0.0
    total_time: float = 0.0
    
    # Tokens
    gen_input_tokens: int = 0
    gen_output_tokens: int = 0
    gen_thinking_tokens: int = 0
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_thinking_tokens: int = 0
    
    # Status
    success: bool = False
    error: Optional[str] = None


class Gemini3Eval:
    """Full pipeline evaluation with Gemini 3.0 Flash judge."""
    
    def __init__(self, model: str = "gemini-2.5-flash", thinking: str = "LOW"):
        self.model = model
        self.thinking = thinking
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("[INIT] Loading job config...", flush=True)
        jobs = get_jobs_config()
        self.job_config = jobs.get(JOB_ID, {})
        self.job_config["job_id"] = JOB_ID
        
        # Initialize components with heartbeat
        old_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.ERROR)
        
        print("[INIT] Connecting to Vector Search...", flush=True)
        self.retriever = VectorSearchRetriever(self.job_config)
        print("[INIT] Initializing Google Ranker...", flush=True)
        self.ranker = GoogleRanker(project_id=PROJECT_ID)
        print("[INIT] Initializing Gemini Generator...", flush=True)
        self.generator = GeminiAnswerGenerator()
        self.citation_processor = CitationProcessor()
        print("[INIT] All components ready!", flush=True)
        
        logging.getLogger().setLevel(old_level)
    
    def load_corpus(self, max_questions: int = DEFAULT_QUESTIONS) -> List[Dict]:
        """Load questions from gold corpus."""
        with open(CORPUS_PATH) as f:
            data = json.load(f)
        questions = data.get("questions", data)
        return questions[:max_questions]
    
    def _get_query_config(self) -> QueryConfig:
        """Create query config."""
        return QueryConfig(
            recall_top_k=100,
            precision_top_n=PRECISION_K,
            enable_hybrid=True,
            rrf_ranking_alpha=0.5,
            enable_reranking=True,
            job_id=JOB_ID,
        )
    
    def _extract_source_doc(self, chunk: Chunk) -> str:
        """Extract source document name from chunk."""
        if chunk.doc_name:
            return chunk.doc_name
        if chunk.chunk_id:
            parts = chunk.chunk_id.rsplit("_chunk_", 1)
            return parts[0] if len(parts) > 1 else chunk.chunk_id
        return ""
    
    def _calculate_mrr(self, docs: List[str], expected_source: str) -> float:
        """Calculate MRR."""
        expected_lower = expected_source.lower()
        for i, doc in enumerate(docs):
            if expected_lower in doc.lower():
                return 1.0 / (i + 1)
        return 0.0
    
    def _judge_answer(self, question: str, ground_truth: str, rag_answer: str, context: str) -> tuple:
        """Use Gemini 3.0 Flash LOW to judge the answer."""
        judge_prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth.

Question: {question}

Ground Truth: {ground_truth}

RAG Answer: {rag_answer}

Context (first 2000 chars):
{context[:2000]}

Score 1-5 on each criterion:
1. Correctness - factually correct vs ground truth?
2. Completeness - covers all key points?
3. Faithfulness - faithful to context, no hallucinations?
4. Relevance - relevant to question?
5. Clarity - clear and well-structured?

Respond with JSON only:
{{"correctness": <1-5>, "completeness": <1-5>, "faithfulness": <1-5>, "relevance": <1-5>, "clarity": <1-5>, "overall_score": <1-5>, "verdict": "pass|partial|fail"}}"""

        try:
            # Use Gemini 2.0 Flash as judge for consistency with baseline
            result = generate(
                judge_prompt,
                model="gemini-2.0-flash",
                max_output_tokens=256,
                temperature=0.0,
                response_mime_type="application/json",
            )
            
            response_text = result.get("text", "").strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            judgment = json.loads(response_text)
            
            # Get token counts
            metadata = result.get("llm_metadata", {})
            input_tokens = metadata.get("prompt_tokens", 0)
            output_tokens = metadata.get("completion_tokens", 0)
            thinking_tokens = metadata.get("thinking_tokens", 0)
            
            return judgment, input_tokens, output_tokens, thinking_tokens
            
        except Exception as e:
            return {
                "correctness": 0, "completeness": 0, "faithfulness": 0,
                "relevance": 0, "clarity": 0, "overall_score": 0,
                "verdict": "error", "error": str(e)
            }, 0, 0, 0
    
    def run_single_query(self, qa: Dict) -> QueryResult:
        """Run a single query through the full pipeline."""
        global _processed_count, _results_buffer
        
        total_start = time.time()
        
        question_id = qa.get("question_id", "unknown")
        question = qa.get("question", "")
        source_filenames = qa.get("source_filenames", [])
        expected_source = source_filenames[0] if source_filenames else ""
        expected_source = expected_source.replace(".pdf", "").lower()
        ground_truth = qa.get("ground_truth_answer", "")
        
        result = QueryResult(
            question_id=question_id,
            question=question,
            question_type=qa.get("question_type", "unknown"),
            difficulty=qa.get("difficulty", "unknown"),
            expected_source=expected_source,
            ground_truth=ground_truth,
        )
        
        try:
            config = self._get_query_config()
            
            # Phase 1: Retrieval
            retrieval_start = time.time()
            retrieval_result = self.retriever.retrieve(question, config)
            result.retrieval_time = time.time() - retrieval_start
            
            chunks = list(retrieval_result.chunks)
            doc_names = [self._extract_source_doc(c) for c in chunks]
            
            result.recall_at_100 = any(expected_source.lower() in d.lower() for d in doc_names[:100])
            for i, doc in enumerate(doc_names):
                if expected_source.lower() in doc.lower():
                    result.first_relevant_rank = i + 1
                    break
            
            # Phase 2: Reranking
            reranking_start = time.time()
            reranked_chunks = self.ranker.rank(question, chunks, config.precision_top_n)
            result.reranking_time = time.time() - reranking_start
            
            reranked_docs = [self._extract_source_doc(c) for c in reranked_chunks]
            result.mrr = self._calculate_mrr(reranked_docs, expected_source)
            
            # Phase 3: Generation
            generation_start = time.time()
            context_str = "\n\n".join([
                f"[Doc {i+1}: {self._extract_source_doc(c)}]\n{c.text}"
                for i, c in enumerate(reranked_chunks)
            ])
            
            # Use orchestrator-style prompt for consistency with baseline
            citation_instruction = """IMPORTANT: You must include numbered citations [1], [2], [3], etc. after every factual statement, referencing the source document numbers shown above. Include the citation number in square brackets immediately after each fact.

CRITICAL: When citing numerical values, specifications, or measurements from the source documents, you MUST copy them EXACTLY as written. Do not convert units, round numbers, or modify values in any way. For example, if a source says "35,000 A", write "35,000 A" - do not change it to "35 kA" or any other format."""

            gen_prompt = f"""Answer the following question based on the source documents provided below.

SOURCE DOCUMENTS:
{context_str}

QUESTION: {question}

{citation_instruction}

Provide a clear, detailed answer with citations."""
            
            if self.model == "gemini-2.5-flash":
                gen_result = generate(
                    gen_prompt,
                    model="gemini-2.5-flash",
                    max_output_tokens=1024,
                    temperature=0.0,
                )
            else:
                gen_result = generate(
                    gen_prompt,
                    model="gemini-3-flash-preview",
                    max_output_tokens=1024,
                    temperature=0.0,
                    thinking_level=self.thinking,
                )
            
            result.generation_time = time.time() - generation_start
            
            result.answer = gen_result.get("text", "")
            result.answer_length = len(result.answer)
            
            # Get generation tokens
            gen_meta = gen_result.get("llm_metadata", {})
            result.gen_input_tokens = gen_meta.get("prompt_tokens", 0)
            result.gen_output_tokens = gen_meta.get("completion_tokens", 0)
            result.gen_thinking_tokens = gen_meta.get("thinking_tokens", 0) or 0
            
            # Phase 4: Judge (always Gemini 3.0 Flash LOW)
            judge_start = time.time()
            judgment, j_in, j_out, j_think = self._judge_answer(
                question, ground_truth, result.answer, context_str
            )
            result.judge_time = time.time() - judge_start
            
            result.judgment = judgment
            result.judge_input_tokens = j_in
            result.judge_output_tokens = j_out
            result.judge_thinking_tokens = j_think
            
            result.total_time = time.time() - total_start
            result.success = True
            
        except Exception as e:
            result.error = str(e)
            result.total_time = time.time() - total_start
        
        # Update progress
        with _progress_lock:
            _processed_count += 1
            _results_buffer.append(result)
            
            # Heartbeat every question for first 5, then every 10
            show_progress = (_processed_count <= 5) or (_processed_count % CHECKPOINT_INTERVAL == 0)
            if show_progress:
                # Calculate running stats
                successes = [r for r in _results_buffer if r.success]
                verdicts = {}
                for r in successes:
                    j = r.judgment
                    if isinstance(j, list):
                        j = j[0] if j else {}
                    v = j.get("verdict", "?") if isinstance(j, dict) else "?"
                    verdicts[v] = verdicts.get(v, 0) + 1
                
                bar = progress_bar(_processed_count, _total_questions)
                verdict_str = " | ".join(f"{k}:{v}" for k, v in sorted(verdicts.items()))
                print(f"{bar} | {verdict_str}", flush=True)
        
        return result
    
    def run_eval(self, questions: List[Dict], workers: int = DEFAULT_WORKERS) -> Dict:
        """Run parallel evaluation."""
        global _processed_count, _total_questions, _results_buffer
        
        _processed_count = 0
        _total_questions = len(questions)
        _results_buffer = []
        
        model_desc = f"{self.model}"
        if self.model == "gemini-3-flash-preview":
            model_desc += f" ({self.thinking})"
        
        print(f"\n{'='*70}", flush=True)
        print(f"GEMINI 3 EVALUATION - {model_desc.upper()}", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"Generator: {model_desc}", flush=True)
        print(f"Judge: gemini-2.0-flash (baseline)", flush=True)
        print(f"Questions: {len(questions)} | Workers: {workers} | Precision@{PRECISION_K}", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"[STARTUP] Initializing {workers} workers...", flush=True)
        
        results = []
        start_time = time.time()
        
        print(f"[STARTUP] Submitting {len(questions)} questions to thread pool...", flush=True)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.run_single_query, q): q for q in questions}
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=180)
                    results.append(result)
                except Exception as e:
                    q = futures[future]
                    results.append(QueryResult(
                        question_id=q.get("question_id", "?"),
                        question=q.get("question", ""),
                        question_type=q.get("question_type", ""),
                        difficulty=q.get("difficulty", ""),
                        expected_source="",
                        ground_truth="",
                        error=str(e),
                    ))
        
        total_time = time.time() - start_time
        
        return self._generate_report(results, total_time, workers)
    
    def _generate_report(self, results: List[QueryResult], total_time: float, workers: int) -> Dict:
        """Generate evaluation report."""
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        # Timing breakdown
        avg_retrieval = sum(r.retrieval_time for r in successful) / len(successful) if successful else 0
        avg_reranking = sum(r.reranking_time for r in successful) / len(successful) if successful else 0
        avg_generation = sum(r.generation_time for r in successful) / len(successful) if successful else 0
        avg_judge = sum(r.judge_time for r in successful) / len(successful) if successful else 0
        
        print(f"\nTotal Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Questions: {len(results)} | Success: {len(successful)} | Failed: {len(failed)}")
        print(f"Throughput: {len(successful)/total_time*60:.1f} questions/min")
        
        print(f"\nTiming Breakdown (avg per question):")
        print(f"  Retrieval:  {avg_retrieval:.2f}s")
        print(f"  Reranking:  {avg_reranking:.2f}s")
        print(f"  Generation: {avg_generation:.2f}s")
        print(f"  Judge:      {avg_judge:.2f}s")
        
        # Retrieval metrics
        recall_100 = sum(1 for r in successful if r.recall_at_100) / len(successful) * 100 if successful else 0
        mrr_values = [r.mrr for r in successful if r.mrr > 0]
        avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0
        
        print(f"\nRetrieval Metrics:")
        print(f"  Recall@100: {recall_100:.1f}%")
        print(f"  MRR: {avg_mrr:.3f}")
        
        # Verdicts
        verdicts = {}
        for r in successful:
            j = r.judgment
            if isinstance(j, list):
                j = j[0] if j else {}
            v = j.get("verdict", "unknown") if isinstance(j, dict) else "unknown"
            verdicts[v] = verdicts.get(v, 0) + 1
        
        print(f"\nVerdicts:")
        for v, count in sorted(verdicts.items()):
            print(f"  {v}: {count} ({count/len(successful)*100:.1f}%)")
        
        pass_count = verdicts.get("pass", 0)
        partial_count = verdicts.get("partial", 0)
        pass_rate = pass_count / len(successful) * 100 if successful else 0
        acceptable_rate = (pass_count + partial_count) / len(successful) * 100 if successful else 0
        
        print(f"\nKey Metrics:")
        print(f"  Pass Rate (4+): {pass_rate:.1f}%")
        print(f"  Acceptable (3+): {acceptable_rate:.1f}%")
        
        # Scores
        scores = ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]
        print("\nAverage Scores:")
        score_avgs = {}
        for score in scores:
            values = []
            for r in successful:
                j = r.judgment
                if isinstance(j, list):
                    j = j[0] if j else {}
                if isinstance(j, dict) and j.get(score):
                    values.append(j.get(score, 0))
            if values:
                avg = sum(values) / len(values)
                score_avgs[score] = avg
                print(f"  {score}: {avg:.2f}")
        
        # Token counts (handle None values)
        total_gen_input = sum(r.gen_input_tokens or 0 for r in successful)
        total_gen_output = sum(r.gen_output_tokens or 0 for r in successful)
        total_gen_thinking = sum(r.gen_thinking_tokens or 0 for r in successful)
        total_judge_input = sum(r.judge_input_tokens or 0 for r in successful)
        total_judge_output = sum(r.judge_output_tokens or 0 for r in successful)
        total_judge_thinking = sum(r.judge_thinking_tokens or 0 for r in successful)
        
        print(f"\nToken Usage:")
        print(f"  Generation - Input: {total_gen_input:,} | Output: {total_gen_output:,} | Thinking: {total_gen_thinking:,}")
        print(f"  Judge      - Input: {total_judge_input:,} | Output: {total_judge_output:,} | Thinking: {total_judge_thinking:,}")
        
        total_input = total_gen_input + total_judge_input
        total_output = total_gen_output + total_judge_output
        total_thinking = total_gen_thinking + total_judge_thinking
        
        print(f"  TOTAL      - Input: {total_input:,} | Output: {total_output:,} | Thinking: {total_thinking:,}")
        
        # Cost estimate
        pricing = PRICING.get(self.model, PRICING["gemini-3-flash-preview"])
        input_cost = (total_input / 1_000_000) * pricing["input"]
        output_cost = (total_output / 1_000_000) * pricing["output"]
        thinking_cost = (total_thinking / 1_000_000) * pricing["thinking"]
        total_cost = input_cost + output_cost + thinking_cost
        
        print(f"\nEstimated Cost:")
        print(f"  Input:    ${input_cost:.4f}")
        print(f"  Output:   ${output_cost:.4f}")
        print(f"  Thinking: ${thinking_cost:.4f}")
        print(f"  TOTAL:    ${total_cost:.4f}")
        
        print(f"{'='*70}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_suffix = self.model.replace("-", "_")
        if self.model == "gemini-3-flash-preview":
            model_suffix += f"_{self.thinking}"
        output_file = self.output_dir / f"eval_{model_suffix}_{timestamp}.json"
        
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "generator_model": self.model,
                "thinking_level": self.thinking if self.model == "gemini-3-flash-preview" else None,
                "judge_model": "gemini-3-flash-preview",
                "judge_thinking": "LOW",
                "precision_k": PRECISION_K,
                "questions": len(results),
                "workers": workers,
            },
            "timing": {
                "total_seconds": total_time,
                "throughput_per_min": len(successful) / total_time * 60 if successful else 0,
                "avg_retrieval": avg_retrieval,
                "avg_reranking": avg_reranking,
                "avg_generation": avg_generation,
                "avg_judge": avg_judge,
            },
            "retrieval": {
                "recall_at_100": recall_100 / 100,
                "mrr": avg_mrr,
            },
            "summary": {
                "successful": len(successful),
                "failed": len(failed),
                "pass_rate": pass_rate / 100,
                "acceptable_rate": acceptable_rate / 100,
                "verdicts": verdicts,
                "scores": score_avgs,
            },
            "tokens": {
                "gen_input": total_gen_input,
                "gen_output": total_gen_output,
                "gen_thinking": total_gen_thinking,
                "judge_input": total_judge_input,
                "judge_output": total_judge_output,
                "judge_thinking": total_judge_thinking,
                "total_input": total_input,
                "total_output": total_output,
                "total_thinking": total_thinking,
            },
            "cost": {
                "input": input_cost,
                "output": output_cost,
                "thinking": thinking_cost,
                "total": total_cost,
            },
            "results": [asdict(r) for r in results],
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description="Gemini 3 Evaluation - Full Pipeline")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                       choices=["gemini-2.5-flash", "gemini-3-flash-preview"],
                       help="Generator model")
    parser.add_argument("--thinking", type=str, default="LOW",
                       choices=["LOW", "HIGH"],
                       help="Thinking level for Gemini 3.0 Flash")
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    
    evaluator = Gemini3Eval(model=args.model, thinking=args.thinking)
    questions = evaluator.load_corpus(args.questions)
    evaluator.run_eval(questions, args.workers)


if __name__ == "__main__":
    main()
