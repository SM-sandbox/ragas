#!/usr/bin/env python3
"""
Orchestrator Evaluation Test - Real Pipeline

Runs 458 gold standard questions through the FULL RAG pipeline:
- Vector Search retrieval (recall@100)
- Google Ranker reranking (precision@25)
- Gemini generation
- LLM Judge evaluation

Uses 50 parallel workers with checkpoints every 10 questions.

Usage:
    python scripts/eval/run_orchestrator_eval.py --questions 458 --workers 50
"""

import sys
# Add gRAG_v3 to path for orchestrator imports
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/gRAG_v3")

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

# Orchestrator imports from gRAG_v3
from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
from services.api.retrieval.vector_search import VectorSearchRetriever
from services.api.ranking.google_ranker import GoogleRanker
from services.api.generation.gemini import GeminiAnswerGenerator
from services.api.processing.citations import CitationProcessor
from services.api.core.config import QueryConfig
from services.api.core.models import Chunk

# LLM Judge - use Gemini 2.0 Flash for consistency with baseline
from langchain_google_vertexai import ChatVertexAI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
JOB_ID = "bfai__eval66a_g1_1536_tt"  # gemini-1536-RETRIEVAL_QUERY (best performer)
CORPUS_PATH = Path(__file__).parent.parent.parent / "clients" / "BFAI" / "qa" / "QA_BFAI_gold_v1-0__q458.json"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "reports" / "orchestrator_eval"

# Defaults
DEFAULT_QUESTIONS = 458
DEFAULT_WORKERS = 50
PRECISION_K = 25  # Baseline is precision@25
CHECKPOINT_INTERVAL = 10  # Checkpoint every 10 questions

# Progress tracking
_progress_lock = Lock()
_processed_count = 0
_total_questions = 0
_checkpoint_results = []


@dataclass
class QueryResult:
    """Result from a single query through the pipeline."""
    question_id: str
    question: str
    question_type: str
    difficulty: str
    expected_source: str
    ground_truth: str
    
    # Retrieval results
    retrieved_docs: List[str] = field(default_factory=list)
    recall_at_100: bool = False
    first_relevant_rank: Optional[int] = None
    
    # After reranking
    reranked_docs: List[str] = field(default_factory=list)
    mrr: float = 0.0
    
    # Generation
    answer: str = ""
    answer_length: int = 0
    sources_cited: int = 0
    
    # LLM Judge scores
    judgment: Dict = field(default_factory=dict)
    
    # Timings
    retrieval_time: float = 0.0
    reranking_time: float = 0.0
    generation_time: float = 0.0
    judge_time: float = 0.0
    total_time: float = 0.0
    
    # Status
    success: bool = False
    error: Optional[str] = None


class OrchestratorEval:
    """Orchestrator evaluation with parallel execution."""
    
    def __init__(self, job_id: str = JOB_ID):
        self.job_id = job_id
        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load job config
        jobs = get_jobs_config()
        self.job_config = jobs.get(job_id, {})
        self.job_config["job_id"] = job_id
        
        print(f"Initializing orchestrator components for job: {job_id}")
        
        # Initialize components (shared across threads)
        self.retriever = VectorSearchRetriever(self.job_config)
        self.ranker = GoogleRanker(project_id=PROJECT_ID)
        self.generator = GeminiAnswerGenerator()
        self.citation_processor = CitationProcessor()
        
        # LLM Judge - Gemini 2.0 Flash for consistency with baseline
        self.judge_llm = ChatVertexAI(
            model_name="gemini-2.0-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.0,
        )
        
        print(f"Judge model: gemini-2.0-flash (consistent with baseline)")
    
    def load_corpus(self, max_questions: int = DEFAULT_QUESTIONS) -> List[Dict]:
        """Load questions from gold corpus."""
        if not CORPUS_PATH.exists():
            raise FileNotFoundError(f"Corpus not found at {CORPUS_PATH}")
        
        with open(CORPUS_PATH) as f:
            data = json.load(f)
        
        questions = data.get("questions", data)
        return questions[:max_questions]
    
    def _get_query_config(self) -> QueryConfig:
        """Create query config for hybrid 50/50 search."""
        return QueryConfig(
            recall_top_k=100,
            precision_top_n=PRECISION_K,
            enable_hybrid=True,
            rrf_ranking_alpha=0.5,
            enable_reranking=True,
            job_id=self.job_id,
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
    
    def _judge_answer(self, question: str, ground_truth: str, rag_answer: str, context: str) -> Dict:
        """Use LLM to judge the RAG answer quality."""
        judge_prompt = f"""You are an expert evaluator for a RAG (Retrieval Augmented Generation) system.
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

Respond with JSON in this exact format:
{{
    "correctness": <1-5>,
    "completeness": <1-5>,
    "faithfulness": <1-5>,
    "relevance": <1-5>,
    "clarity": <1-5>,
    "overall_score": <1-5>,
    "verdict": "pass|partial|fail"
}}"""
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
                "correctness": 0,
                "completeness": 0,
                "faithfulness": 0,
                "relevance": 0,
                "clarity": 0,
                "overall_score": 0,
                "verdict": "error",
                "error": str(e)
            }
    
    def run_single_query(self, qa: Dict) -> QueryResult:
        """Run a single query through the full pipeline."""
        global _processed_count, _checkpoint_results
        
        total_start = time.time()
        
        question_id = qa.get("question_id", "unknown")
        question = qa.get("question", "")
        source_filenames = qa.get("source_filenames", [])
        expected_source = source_filenames[0] if source_filenames else qa.get("source_document", "")
        expected_source = expected_source.replace(".pdf", "").lower()
        ground_truth = qa.get("ground_truth_answer", qa.get("answer", ""))
        
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
            result.retrieved_docs = [self._extract_source_doc(c) for c in chunks]
            
            # Check recall@100
            result.recall_at_100 = any(
                expected_source.lower() in doc.lower() 
                for doc in result.retrieved_docs[:100]
            )
            
            # Find first relevant rank
            for i, doc in enumerate(result.retrieved_docs):
                if expected_source.lower() in doc.lower():
                    result.first_relevant_rank = i + 1
                    break
            
            # Phase 2: Reranking
            reranking_start = time.time()
            if chunks and config.enable_reranking:
                reranked_chunks = self.ranker.rank(question, chunks, config.precision_top_n)
            else:
                reranked_chunks = chunks[:config.precision_top_n]
            result.reranking_time = time.time() - reranking_start
            
            result.reranked_docs = [self._extract_source_doc(c) for c in reranked_chunks]
            result.mrr = self._calculate_mrr(result.reranked_docs, expected_source)
            
            # Phase 3: Generation
            generation_start = time.time()
            context_str = "\n\n".join([
                f"[Document {i+1}: {self._extract_source_doc(c)}]\n{c.text}"
                for i, c in enumerate(reranked_chunks)
            ])
            
            generation_result = self.generator.generate(
                query=question,
                context=context_str,
                config=config,
            )
            result.generation_time = time.time() - generation_start
            
            # Process citations
            processed_answer, sources = self.citation_processor.process(
                answer=generation_result.answer_text,
                chunks=reranked_chunks,
                max_sources=config.max_sources,
            )
            
            result.answer = processed_answer
            result.answer_length = len(processed_answer)
            result.sources_cited = len(sources)
            
            # Phase 4: LLM Judge
            judge_start = time.time()
            result.judgment = self._judge_answer(question, ground_truth, processed_answer, context_str)
            result.judge_time = time.time() - judge_start
            
            result.total_time = time.time() - total_start
            result.success = True
            
        except Exception as e:
            result.error = str(e)
            result.total_time = time.time() - total_start
            logger.error(f"Error on {question_id}: {e}")
        
        # Update progress
        with _progress_lock:
            _processed_count += 1
            _checkpoint_results.append(result)
            
            # Checkpoint every 10 questions
            if _processed_count % CHECKPOINT_INTERVAL == 0:
                verdict = result.judgment.get("verdict", "?") if result.success else "ERROR"
                print(f"\n{'='*60}")
                print(f"CHECKPOINT: {_processed_count}/{_total_questions} questions processed")
                print(f"Last: {question_id} -> {verdict} ({result.total_time:.1f}s)")
                
                # Quick stats
                successes = [r for r in _checkpoint_results if r.success]
                if successes:
                    verdicts = {}
                    for r in successes:
                        v = r.judgment.get("verdict", "unknown")
                        verdicts[v] = verdicts.get(v, 0) + 1
                    print(f"Verdicts so far: {verdicts}")
                print(f"{'='*60}\n")
        
        return result
    
    def run_parallel_eval(self, questions: List[Dict], workers: int = DEFAULT_WORKERS) -> Dict:
        """Run parallel evaluation with specified workers."""
        global _processed_count, _total_questions, _checkpoint_results
        
        _processed_count = 0
        _total_questions = len(questions)
        _checkpoint_results = []
        
        print(f"\n{'='*70}")
        print(f"ORCHESTRATOR EVALUATION - REAL PIPELINE")
        print(f"{'='*70}")
        print(f"Job: {self.job_id}")
        print(f"Questions: {len(questions)}")
        print(f"Workers: {workers}")
        print(f"Precision@K: {PRECISION_K}")
        print(f"Judge: gemini-2.0-flash")
        print(f"Checkpoints: every {CHECKPOINT_INTERVAL} questions")
        print(f"{'='*70}\n")
        
        results = []
        start_time = time.time()
        
        print(f"Starting parallel execution with {workers} workers...")
        print("-" * 70)
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self.run_single_query, q): q for q in questions}
            
            for future in as_completed(futures):
                q = futures[future]
                qid = q.get("question_id", "unknown")
                
                try:
                    result = future.result(timeout=180)  # 3 min timeout
                    results.append(result)
                    
                    if result.success:
                        verdict = result.judgment.get("verdict", "?")
                        print(f"[{len(results)}/{len(questions)}] {qid}: {verdict} ({result.total_time:.1f}s)")
                    else:
                        print(f"[{len(results)}/{len(questions)}] {qid}: ERROR - {result.error[:50] if result.error else 'unknown'}")
                        
                except Exception as e:
                    results.append(QueryResult(
                        question_id=qid,
                        question=q.get("question", ""),
                        question_type=q.get("question_type", "unknown"),
                        difficulty=q.get("difficulty", "unknown"),
                        expected_source="",
                        ground_truth="",
                        error=str(e),
                    ))
                    print(f"[{len(results)}/{len(questions)}] {qid}: TIMEOUT - {e}")
        
        total_time = time.time() - start_time
        
        return self._generate_report(results, total_time, workers)
    
    def _generate_report(self, results: List[QueryResult], total_time: float, workers: int) -> Dict:
        """Generate evaluation report."""
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\nTotal time: {total_time:.1f}s")
        print(f"Questions: {len(results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Success rate: {len(successful)/len(results)*100:.1f}%")
        
        if successful:
            throughput = len(successful) / total_time * 60
            avg_time = sum(r.total_time for r in successful) / len(successful)
            
            print(f"\nTiming:")
            print(f"  Avg per question: {avg_time:.2f}s")
            print(f"  Throughput: {throughput:.1f} questions/min")
            
            # Retrieval metrics
            recall_100 = sum(1 for r in successful if r.recall_at_100) / len(successful) * 100
            mrr_values = [r.mrr for r in successful if r.mrr > 0]
            avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0
            
            print(f"\nRetrieval:")
            print(f"  Recall@100: {recall_100:.1f}%")
            print(f"  MRR: {avg_mrr:.3f}")
            
            # Verdict distribution
            verdicts = {}
            for r in successful:
                v = r.judgment.get("verdict", "unknown")
                verdicts[v] = verdicts.get(v, 0) + 1
            
            print(f"\nVerdicts:")
            for v, count in sorted(verdicts.items()):
                print(f"  {v}: {count} ({count/len(successful)*100:.1f}%)")
            
            # Calculate pass rate
            pass_count = verdicts.get("pass", 0)
            partial_count = verdicts.get("partial", 0)
            fail_count = verdicts.get("fail", 0)
            
            pass_rate = pass_count / len(successful) * 100 if successful else 0
            acceptable_rate = (pass_count + partial_count) / len(successful) * 100 if successful else 0
            
            print(f"\nKey Metrics:")
            print(f"  Pass Rate (4+): {pass_rate:.1f}%")
            print(f"  Acceptable (3+): {acceptable_rate:.1f}%")
            
            # Score averages
            scores = ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]
            print("\nAverage Scores:")
            for score in scores:
                values = [r.judgment.get(score, 0) for r in successful if r.judgment.get(score)]
                if values:
                    print(f"  {score}: {sum(values)/len(values):.2f}")
        
        if failed:
            print(f"\nFailed questions:")
            for r in failed[:5]:
                print(f"  {r.question_id}: {r.error[:60] if r.error else 'unknown'}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
        
        print("=" * 70)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"orchestrator_eval_{timestamp}.json"
        
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "job_id": self.job_id,
                "judge_model": "gemini-2.0-flash",
                "precision_k": PRECISION_K,
                "questions": len(results),
                "workers": workers,
                "total_time_seconds": total_time,
                "throughput_per_minute": len(successful) / total_time * 60 if successful else 0,
            },
            "summary": {
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) if results else 0,
                "recall_at_100": sum(1 for r in successful if r.recall_at_100) / len(successful) if successful else 0,
                "pass_rate": verdicts.get("pass", 0) / len(successful) if successful else 0,
                "acceptable_rate": (verdicts.get("pass", 0) + verdicts.get("partial", 0)) / len(successful) if successful else 0,
            },
            "results": [asdict(r) for r in results],
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        return output_data


def main():
    parser = argparse.ArgumentParser(description="Orchestrator Evaluation - Real Pipeline")
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS, help="Number of questions (default: 458)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Parallel workers (default: 50)")
    args = parser.parse_args()
    
    evaluator = OrchestratorEval()
    questions = evaluator.load_corpus(args.questions)
    evaluator.run_parallel_eval(questions, args.workers)


if __name__ == "__main__":
    main()
