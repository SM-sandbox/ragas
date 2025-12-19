#!/usr/bin/env python3
"""
End-to-End Orchestrator Test

Tests the full RAG pipeline through the orchestrator:
- Retrieval (hybrid 50/50)
- Reranking (Google Ranking API)
- Answer Generation
- LLM-as-Judge evaluation

Runs 3 times for consistency analysis.
Tracks: MRR, Recall@K, Precision@10, answer length, all timings.
"""

import sys
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/gRAG_v3")

import json
import time
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

from tqdm import tqdm

# Orchestrator imports
from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
from services.api.retrieval.vector_search import VectorSearchRetriever
from services.api.ranking.google_ranker import GoogleRanker
from services.api.generation.gemini import GeminiAnswerGenerator
from services.api.processing.citations import CitationProcessor
from services.api.core.config import QueryConfig
from services.api.core.models import Chunk

# LLM Judge
from langchain_google_vertexai import ChatVertexAI

# Configuration
JOB_ID = "bfai__eval66a_g1_1536_tt"  # gemini-1536-RETRIEVAL_QUERY (best performer)
CORPUS_FILE = "clients_qa_gold/BFAI/qa/QA_BFAI_gold_v1-0__q458.json"  # Full 458 gold questions
K_VALUES = [5, 10, 15, 20, 25, 50, 100]
NUM_RUNS = 1  # Single run for gold standard eval
PRECISION_K = 25  # Can be changed to 12 for comparison

@dataclass
class QueryResult:
    """Result from a single query."""
    question_id: int
    question: str
    question_type: str
    expected_source: str
    ground_truth: str
    
    # Retrieval results
    retrieved_docs: List[str] = field(default_factory=list)
    first_relevant_rank: Optional[int] = None
    recall_at_k: Dict[int, bool] = field(default_factory=dict)
    
    # After reranking
    reranked_docs: List[str] = field(default_factory=list)
    precision_at_10: float = 0.0
    mrr_at_10: float = 0.0
    
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


class E2EOrchestatorTest:
    """End-to-end orchestrator test with LLM judging."""
    
    def __init__(self, job_id: str = JOB_ID):
        self.job_id = job_id
        self.output_dir = Path(__file__).parent.parent / "experiments" / f"{datetime.now().strftime('%Y-%m-%d')}_e2e_orchestrator"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load job config
        jobs = get_jobs_config()
        self.job_config = jobs.get(job_id, {})
        self.job_config["job_id"] = job_id
        
        # Initialize components
        print(f"Initializing orchestrator components for job: {job_id}")
        self.retriever = VectorSearchRetriever(self.job_config)
        self.ranker = GoogleRanker(project_id=PROJECT_ID)
        self.generator = GeminiAnswerGenerator()
        self.citation_processor = CitationProcessor()
        
        # LLM Judge
        self.judge_llm = ChatVertexAI(
            model_name="gemini-2.0-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.0,
        )
        
        # Load corpus
        self.corpus = self._load_corpus()
        print(f"Loaded {len(self.corpus)} questions")
    
    def _load_corpus(self) -> List[Dict]:
        """Load the Q&A corpus."""
        corpus_path = Path(__file__).parent.parent.parent / CORPUS_FILE
        with open(corpus_path) as f:
            data = json.load(f)
            # Handle both old format (list) and new format (dict with 'questions' key)
            if isinstance(data, dict) and "questions" in data:
                return data["questions"]
            return data
    
    def _get_query_config(self, top_k: int = 100) -> QueryConfig:
        """Create query config for hybrid 50/50 search."""
        return QueryConfig(
            recall_top_k=top_k,
            precision_top_n=PRECISION_K,  # After reranking (25 or 12)
            enable_hybrid=True,
            rrf_ranking_alpha=0.5,  # 50/50 dense/sparse
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
    
    def _calculate_recall(self, retrieved_docs: List[str], expected_source: str, k: int) -> bool:
        """Check if expected source is in top K results."""
        top_k_docs = retrieved_docs[:k]
        return any(expected_source.lower() in doc.lower() for doc in top_k_docs)
    
    def _calculate_mrr(self, retrieved_docs: List[str], expected_source: str, k: int) -> float:
        """Calculate MRR@K."""
        for i, doc in enumerate(retrieved_docs[:k]):
            if expected_source.lower() in doc.lower():
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_precision(self, reranked_docs: List[str], expected_source: str, k: int = 10) -> float:
        """Calculate Precision@K (binary - is expected doc in top K)."""
        top_k = reranked_docs[:k]
        relevant_count = sum(1 for doc in top_k if expected_source.lower() in doc.lower())
        return relevant_count / k if k > 0 else 0.0
    
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
    "verdict": "pass|partial|fail",
    "explanation": "Brief explanation of the evaluation"
}}
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
                "correctness": 0,
                "completeness": 0,
                "faithfulness": 0,
                "relevance": 0,
                "clarity": 0,
                "overall_score": 0,
                "verdict": "error",
                "explanation": f"Evaluation error: {str(e)}"
            }
    
    def run_single_query(self, qa: Dict) -> QueryResult:
        """Run a single query through the full pipeline."""
        total_start = time.time()
        
        question = qa.get("question", "")
        # Handle both old format (source_document) and new format (source_filenames list)
        source_filenames = qa.get("source_filenames", [])
        expected_source = source_filenames[0] if source_filenames else qa.get("source_document", "")
        expected_source = expected_source.replace(".pdf", "").lower()  # Normalize
        ground_truth = qa.get("ground_truth_answer", qa.get("answer", ""))
        
        result = QueryResult(
            question_id=qa.get("question_id", qa.get("id", 0)),
            question=question,
            question_type=qa.get("question_type", "unknown"),
            expected_source=expected_source,
            ground_truth=ground_truth,
        )
        
        config = self._get_query_config(top_k=100)
        
        # Phase 1: Retrieval
        retrieval_start = time.time()
        retrieval_result = self.retriever.retrieve(question, config)
        result.retrieval_time = time.time() - retrieval_start
        
        # Extract doc names
        chunks = list(retrieval_result.chunks)
        result.retrieved_docs = [self._extract_source_doc(c) for c in chunks]
        
        # Calculate recall at various K
        for k in K_VALUES:
            result.recall_at_k[k] = self._calculate_recall(result.retrieved_docs, expected_source, k)
        
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
        result.precision_at_10 = self._calculate_precision(result.reranked_docs, expected_source, 10)
        result.mrr_at_10 = self._calculate_mrr(result.reranked_docs, expected_source, 10)
        
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
        
        return result
    
    def run_full_test(self, run_number: int = 1) -> Dict:
        """Run full test on all questions."""
        print(f"\n{'='*60}")
        print(f"RUN {run_number} OF {NUM_RUNS}")
        print(f"{'='*60}")
        print(f"Job: {self.job_id}")
        print(f"Questions: {len(self.corpus)}")
        print(f"Config: Hybrid 50/50, Reranking enabled")
        print(f"{'='*60}\n")
        
        results = []
        
        for qa in tqdm(self.corpus, desc=f"Run {run_number}"):
            try:
                result = self.run_single_query(qa)
                results.append(result)
            except Exception as e:
                print(f"\nError on question {qa.get('id', '?')}: {e}")
                results.append(QueryResult(
                    question_id=qa.get("id", 0),
                    question=qa.get("question", ""),
                    question_type=qa.get("question_type", "unknown"),
                    expected_source=qa.get("source_document", ""),
                    ground_truth=qa.get("answer", ""),
                ))
        
        return self._calculate_run_metrics(results, run_number)
    
    def _calculate_run_metrics(self, results: List[QueryResult], run_number: int) -> Dict:
        """Calculate aggregate metrics for a run."""
        valid_results = [r for r in results if r.answer]
        
        # Recall@K
        recall_metrics = {}
        for k in K_VALUES:
            hits = sum(1 for r in results if r.recall_at_k.get(k, False))
            recall_metrics[f"recall@{k}"] = hits / len(results) if results else 0
        
        # MRR@10
        mrr_values = [r.mrr_at_10 for r in results]
        mrr_at_10 = statistics.mean(mrr_values) if mrr_values else 0
        
        # Precision@10
        precision_values = [r.precision_at_10 for r in results]
        precision_at_10 = statistics.mean(precision_values) if precision_values else 0
        
        # Answer length
        answer_lengths = [r.answer_length for r in valid_results]
        avg_answer_length = statistics.mean(answer_lengths) if answer_lengths else 0
        
        # LLM Judge metrics
        judge_metrics = {
            "correctness": [],
            "completeness": [],
            "faithfulness": [],
            "relevance": [],
            "clarity": [],
            "overall_score": [],
        }
        verdicts = {"pass": 0, "partial": 0, "fail": 0, "error": 0}
        
        for r in results:
            if r.judgment:
                for metric in judge_metrics:
                    judge_metrics[metric].append(r.judgment.get(metric, 0))
                verdict = r.judgment.get("verdict", "error")
                if verdict in verdicts:
                    verdicts[verdict] += 1
        
        judge_means = {k: statistics.mean(v) if v else 0 for k, v in judge_metrics.items()}
        pass_rate = (verdicts["pass"] + verdicts["partial"]) / len(results) if results else 0
        
        # Timings
        timing_metrics = {
            "retrieval": [r.retrieval_time for r in results],
            "reranking": [r.reranking_time for r in results],
            "generation": [r.generation_time for r in results],
            "judge": [r.judge_time for r in results],
            "total": [r.total_time for r in results],
        }
        
        timing_stats = {}
        for name, values in timing_metrics.items():
            if values:
                timing_stats[name] = {
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "total": sum(values),
                }
        
        return {
            "run_number": run_number,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "valid_answers": len(valid_results),
            "recall": recall_metrics,
            "mrr_at_10": mrr_at_10,
            "precision_at_10": precision_at_10,
            "avg_answer_length": avg_answer_length,
            "llm_judge": {
                "metrics": judge_means,
                "verdicts": verdicts,
                "pass_rate": pass_rate,
            },
            "timing": timing_stats,
            "detailed_results": [asdict(r) for r in results],
        }
    
    def run_consistency_test(self) -> Dict:
        """Run the test NUM_RUNS times for consistency analysis."""
        print("\n" + "="*70)
        print("E2E ORCHESTRATOR CONSISTENCY TEST")
        print("="*70)
        print(f"Job ID: {self.job_id}")
        print(f"Embedding: gemini-embedding-001, 1536 dim, RETRIEVAL_QUERY")
        print(f"Hybrid: 50/50 (dense/sparse)")
        print(f"Reranking: Google Ranking API")
        print(f"Questions: {len(self.corpus)}")
        print(f"Runs: {NUM_RUNS}")
        print("="*70)
        
        all_runs = []
        
        for run_num in range(1, NUM_RUNS + 1):
            run_result = self.run_full_test(run_num)
            all_runs.append(run_result)
            
            # Save intermediate results
            run_path = self.output_dir / f"run_{run_num}.json"
            with open(run_path, 'w') as f:
                json.dump(run_result, f, indent=2, default=str)
            print(f"✓ Run {run_num} saved to: {run_path}")
        
        # Calculate consistency metrics
        consistency = self._calculate_consistency(all_runs)
        
        # Generate final report
        final_report = {
            "test_config": {
                "job_id": self.job_id,
                "embedding_model": "gemini-embedding-001",
                "embedding_dimension": 1536,
                "task_type": "RETRIEVAL_QUERY",
                "hybrid_alpha": 0.5,
                "reranking": True,
                "num_runs": NUM_RUNS,
                "total_questions": len(self.corpus),
            },
            "runs": all_runs,
            "consistency": consistency,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save final report
        report_path = self.output_dir / "e2e_consistency_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(final_report)
        
        return final_report
    
    def _calculate_consistency(self, runs: List[Dict]) -> Dict:
        """Calculate consistency metrics across runs."""
        # Extract key metrics from each run
        recall_10_values = [r["recall"]["recall@10"] for r in runs]
        mrr_values = [r["mrr_at_10"] for r in runs]
        precision_values = [r["precision_at_10"] for r in runs]
        pass_rates = [r["llm_judge"]["pass_rate"] for r in runs]
        overall_scores = [r["llm_judge"]["metrics"]["overall_score"] for r in runs]
        
        def calc_stats(values):
            if len(values) < 2:
                return {"mean": values[0] if values else 0, "std": 0, "min": values[0] if values else 0, "max": values[0] if values else 0}
            return {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
                "min": min(values),
                "max": max(values),
            }
        
        return {
            "recall@10": calc_stats(recall_10_values),
            "mrr@10": calc_stats(mrr_values),
            "precision@10": calc_stats(precision_values),
            "pass_rate": calc_stats(pass_rates),
            "overall_score": calc_stats(overall_scores),
        }
    
    def _generate_markdown_report(self, report: Dict):
        """Generate markdown report."""
        md_path = Path(__file__).parent.parent / "reports" / "E2E_Orchestrator_Test_Report.md"
        
        config = report["test_config"]
        consistency = report["consistency"]
        runs = report["runs"]
        
        # Get first run for detailed metrics
        run1 = runs[0] if runs else {}
        
        md_content = f"""# E2E Orchestrator Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Runs:** {config['num_runs']}

---

## 1. Test Configuration

| Parameter | Value |
|-----------|-------|
| Job ID | `{config['job_id']}` |
| Embedding Model | {config['embedding_model']} |
| Dimensions | {config['embedding_dimension']} |
| Task Type | {config['task_type']} |
| Hybrid Alpha | {config['hybrid_alpha']} (50/50 dense/sparse) |
| Reranking | Google Ranking API |
| Questions | {config['total_questions']} |

---

## 2. Consistency Analysis (Across {config['num_runs']} Runs)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
| Recall@10 | {consistency['recall@10']['mean']*100:.1f}% | {consistency['recall@10']['std']*100:.2f}% | {consistency['recall@10']['min']*100:.1f}% | {consistency['recall@10']['max']*100:.1f}% |
| MRR@10 | {consistency['mrr@10']['mean']:.3f} | {consistency['mrr@10']['std']:.4f} | {consistency['mrr@10']['min']:.3f} | {consistency['mrr@10']['max']:.3f} |
| Precision@10 | {consistency['precision@10']['mean']*100:.1f}% | {consistency['precision@10']['std']*100:.2f}% | {consistency['precision@10']['min']*100:.1f}% | {consistency['precision@10']['max']*100:.1f}% |
| Pass Rate | {consistency['pass_rate']['mean']*100:.1f}% | {consistency['pass_rate']['std']*100:.2f}% | {consistency['pass_rate']['min']*100:.1f}% | {consistency['pass_rate']['max']*100:.1f}% |
| Overall Score | {consistency['overall_score']['mean']:.2f}/5 | {consistency['overall_score']['std']:.3f} | {consistency['overall_score']['min']:.2f} | {consistency['overall_score']['max']:.2f} |

---

## 3. Retrieval Metrics (Run 1)

### 3.1 Recall@K

| K | Recall |
|---|--------|
"""
        for k in K_VALUES:
            recall_val = run1.get("recall", {}).get(f"recall@{k}", 0)
            md_content += f"| {k} | {recall_val*100:.1f}% |\n"
        
        md_content += f"""
### 3.2 MRR & Precision

| Metric | Value |
|--------|-------|
| MRR@10 | {run1.get('mrr_at_10', 0):.3f} |
| Precision@10 | {run1.get('precision_at_10', 0)*100:.1f}% |

---

## 4. LLM-as-Judge Results (Run 1)

### 4.1 Quality Scores (1-5 scale)

| Metric | Score |
|--------|-------|
"""
        judge_metrics = run1.get("llm_judge", {}).get("metrics", {})
        for metric in ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]:
            md_content += f"| {metric.capitalize()} | {judge_metrics.get(metric, 0):.2f} |\n"
        
        verdicts = run1.get("llm_judge", {}).get("verdicts", {})
        pass_rate = run1.get("llm_judge", {}).get("pass_rate", 0)
        
        md_content += f"""
### 4.2 Verdict Distribution

| Verdict | Count |
|---------|-------|
| Pass | {verdicts.get('pass', 0)} |
| Partial | {verdicts.get('partial', 0)} |
| Fail | {verdicts.get('fail', 0)} |
| Error | {verdicts.get('error', 0)} |

**Pass Rate:** {pass_rate*100:.1f}%

---

## 5. Answer Statistics

| Metric | Value |
|--------|-------|
| Avg Answer Length | {run1.get('avg_answer_length', 0):.0f} chars |

---

## 6. Timing Breakdown (Run 1)

| Phase | Avg | Min | Max | Total |
|-------|-----|-----|-----|-------|
"""
        timing = run1.get("timing", {})
        for phase in ["retrieval", "reranking", "generation", "judge", "total"]:
            t = timing.get(phase, {})
            md_content += f"| {phase.capitalize()} | {t.get('mean', 0):.3f}s | {t.get('min', 0):.3f}s | {t.get('max', 0):.3f}s | {t.get('total', 0):.1f}s |\n"
        
        md_content += f"""
---

## 7. Comparison with Previous Results

### Previous LLM Judge Results (2025-12-13)

| Metric | Previous | Current | Δ |
|--------|----------|---------|---|
| Overall Score | 4.16/5 | {judge_metrics.get('overall_score', 0):.2f}/5 | {judge_metrics.get('overall_score', 0) - 4.16:+.2f} |
| Pass Rate | 82.6% | {pass_rate*100:.1f}% | {(pass_rate - 0.826)*100:+.1f}% |
| Correctness | 4.13/5 | {judge_metrics.get('correctness', 0):.2f}/5 | {judge_metrics.get('correctness', 0) - 4.13:+.2f} |
| Faithfulness | 4.95/5 | {judge_metrics.get('faithfulness', 0):.2f}/5 | {judge_metrics.get('faithfulness', 0) - 4.95:+.2f} |

---

## 8. Key Findings

1. **Consistency**: Results are {'highly consistent' if consistency['recall@10']['std'] < 0.01 else 'moderately consistent'} across {config['num_runs']} runs (Recall@10 std: {consistency['recall@10']['std']*100:.2f}%)

2. **Retrieval Quality**: Recall@10 = {consistency['recall@10']['mean']*100:.1f}%, MRR@10 = {consistency['mrr@10']['mean']:.3f}

3. **Answer Quality**: Overall LLM Judge score = {consistency['overall_score']['mean']:.2f}/5, Pass rate = {consistency['pass_rate']['mean']*100:.1f}%

4. **Performance**: Average total time per query = {timing.get('total', {}).get('mean', 0):.2f}s
   - Retrieval: {timing.get('retrieval', {}).get('mean', 0):.3f}s
   - Reranking: {timing.get('reranking', {}).get('mean', 0):.3f}s
   - Generation: {timing.get('generation', {}).get('mean', 0):.3f}s
   - LLM Judge: {timing.get('judge', {}).get('mean', 0):.3f}s
"""
        
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        print(f"\n✓ Markdown report saved to: {md_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="E2E Orchestrator Test")
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs")
    parser.add_argument("--max-questions", type=int, default=None, help="Max questions (for testing)")
    args = parser.parse_args()
    
    num_runs = args.runs
    
    tester = E2EOrchestatorTest()
    
    if args.max_questions:
        tester.corpus = tester.corpus[:args.max_questions]
    
    report = tester.run_consistency_test()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
