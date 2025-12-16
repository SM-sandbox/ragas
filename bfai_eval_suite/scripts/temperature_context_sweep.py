#!/usr/bin/env python3
"""
Temperature and Context Size Sweep Experiments

PROJECT 1: Temperature Sweep (0.0, 0.1, 0.2, 0.3)
PROJECT 2: Context Size Sweep (5, 10, 15, 20, 25, 50, 100)

Features:
- Cached retrieval/reranking (run once, reuse for all experiments)
- 5x retry with exponential backoff
- JSONL checkpoints (resume from where left off)
- 15 workers for parallel processing
- Progress bar with ETA
- .md report generated after each experiment
"""

import sys
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")

import json
import time
import random
import statistics
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from tqdm import tqdm

# Orchestrator imports
from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
from services.api.retrieval.vector_search import VectorSearchRetriever
from services.api.ranking.google_ranker import GoogleRanker
from services.api.generation.gemini import GeminiAnswerGenerator
from services.api.core.config import QueryConfig
from services.api.core.models import Chunk

# LLM Judge
from langchain_google_vertexai import ChatVertexAI

# Configuration
JOB_ID = "bfai__eval66a_g1_1536_tt"
CORPUS_FILE = "corpus/qa_corpus_200.json"
MAX_WORKERS = 15
MAX_RETRIES = 5
BASE_DELAY = 1.0

# Experiment settings
TEMPERATURES = [0.0, 0.1, 0.2, 0.3]
CONTEXT_SIZES = [5, 10, 15, 20, 25, 50, 100]
RECALL_TOP_K = 100  # Retrieve 100, cache all


def retry_with_backoff(func, max_retries=MAX_RETRIES, base_delay=BASE_DELAY):
    """Execute function with retry and exponential backoff."""
    last_exception: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            if "429" in str(e) or "resource exhausted" in error_str or "quota" in error_str:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"  Rate limited, retry {attempt+1}/{max_retries} in {delay:.1f}s...")
                time.sleep(delay)
            else:
                # Non-retryable error after first attempt
                if attempt > 0:
                    raise
                delay = base_delay + random.uniform(0, 0.5)
                time.sleep(delay)
    raise last_exception


class TemperatureContextSweep:
    """Run temperature and context size experiments."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.experiment_dir = self.base_dir / "experiments" / f"{datetime.now().strftime('%Y-%m-%d')}_temp_context_sweep"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir = self.base_dir / "reports"
        
        # Load job config
        jobs = get_jobs_config()
        self.job_config = jobs.get(JOB_ID, {})
        self.job_config["job_id"] = JOB_ID
        
        # Initialize orchestrator components
        print(f"Initializing orchestrator components for job: {JOB_ID}")
        self.retriever = VectorSearchRetriever(self.job_config)
        self.ranker = GoogleRanker(project_id=PROJECT_ID)
        
        # LLM Judge (always temp 0.0 for consistency)
        self.judge_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.0,
        )
        
        # Load corpus
        self.corpus = self._load_corpus()
        print(f"Loaded {len(self.corpus)} questions")
        
        # Cache for retrieval results
        self.retrieval_cache_file = self.experiment_dir / "retrieval_cache.json"
        self.retrieval_cache = {}
    
    def _load_corpus(self) -> List[Dict]:
        """Load the Q&A corpus."""
        corpus_path = self.base_dir / CORPUS_FILE
        with open(corpus_path) as f:
            return json.load(f)
    
    def _get_query_config(self, top_k: int = RECALL_TOP_K, precision_n: int = 10) -> QueryConfig:
        """Create query config."""
        return QueryConfig(
            recall_top_k=top_k,
            precision_top_n=precision_n,
            enable_hybrid=True,
            rrf_ranking_alpha=0.5,
            enable_reranking=False,  # We do reranking separately
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
    
    def _chunk_to_dict(self, chunk: Chunk) -> Dict:
        """Convert chunk to serializable dict."""
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "doc_id": chunk.doc_id,
            "doc_name": chunk.doc_name,
            "page": chunk.page,
            "score": chunk.score,
            "uri": chunk.uri,
        }
    
    def _dict_to_chunk(self, d: Dict) -> Chunk:
        """Convert dict back to Chunk."""
        return Chunk(
            chunk_id=d["chunk_id"],
            text=d["text"],
            doc_id=d["doc_id"],
            doc_name=d["doc_name"],
            page=d.get("page"),
            score=d.get("score", 0.0),
            uri=d.get("uri"),
        )
    
    # =========================================================================
    # PHASE 1: Retrieval + Reranking (cached)
    # =========================================================================
    
    def run_retrieval_phase(self):
        """Run retrieval and reranking for all questions, cache results."""
        print("\n" + "=" * 70)
        print("PHASE 1: RETRIEVAL + RERANKING (CACHED)")
        print("=" * 70)
        
        # Check if cache exists
        if self.retrieval_cache_file.exists():
            print(f"Loading cached retrieval results from {self.retrieval_cache_file}")
            with open(self.retrieval_cache_file) as f:
                self.retrieval_cache = json.load(f)
            print(f"Loaded {len(self.retrieval_cache)} cached results")
            return
        
        config = self._get_query_config(top_k=RECALL_TOP_K)
        
        results = {}
        timing = {"retrieval": [], "reranking": []}
        
        for item in tqdm(self.corpus, desc="Retrieval + Reranking"):
            q_id = str(item.get("id", item.get("question_id", len(results))))
            question = item["question"]
            
            # Retrieval
            start = time.time()
            def do_retrieve():
                return self.retriever.retrieve(question, config)
            
            try:
                result = retry_with_backoff(do_retrieve)
                chunks = list(result.chunks)
            except Exception as e:
                print(f"\nRetrieval failed for Q{q_id}: {e}")
                continue
            
            retrieval_time = time.time() - start
            timing["retrieval"].append(retrieval_time)
            
            # Reranking (get all 100 reranked)
            start = time.time()
            def do_rerank():
                return self.ranker.rank(question, chunks, top_n=RECALL_TOP_K)
            
            try:
                reranked = retry_with_backoff(do_rerank)
            except Exception as e:
                print(f"\nReranking failed for Q{q_id}: {e}")
                reranked = chunks[:RECALL_TOP_K]
            
            reranking_time = time.time() - start
            timing["reranking"].append(reranking_time)
            
            # Store in cache
            results[q_id] = {
                "question": question,
                "question_type": item.get("question_type", "unknown"),
                "ground_truth": item.get("ground_truth", item.get("answer", "")),
                "expected_source": item.get("source_document", item.get("doc_name", "")),
                "retrieved_docs": [self._chunk_to_dict(c) for c in chunks],
                "reranked_docs": [self._chunk_to_dict(c) for c in reranked],
                "retrieval_time": retrieval_time,
                "reranking_time": reranking_time,
            }
        
        # Save cache
        self.retrieval_cache = results
        with open(self.retrieval_cache_file, "w") as f:
            json.dump(results, f)
        
        # Print timing summary
        print(f"\n✓ Cached {len(results)} retrieval results")
        print(f"  Avg retrieval time: {statistics.mean(timing['retrieval']):.3f}s")
        print(f"  Avg reranking time: {statistics.mean(timing['reranking']):.3f}s")
    
    # =========================================================================
    # PHASE 2: Generation + Judging
    # =========================================================================
    
    def _build_context(self, chunks: List[Dict], top_n: int) -> str:
        """Build context string from top N chunks."""
        context_parts = []
        for i, c in enumerate(chunks[:top_n]):
            text = c["text"][:2000]  # Limit per chunk
            context_parts.append(f"[Document {i+1}: {c['doc_name']}]\n{text}")
        return "\n\n".join(context_parts)
    
    def _judge_answer(self, question: str, ground_truth: str, answer: str) -> Dict:
        """Use LLM to judge answer quality."""
        prompt = f"""You are an expert evaluator. Compare the RAG system's answer against the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG System Answer: {answer}

Rate the RAG answer on these dimensions (1-5 scale):
1. Correctness: Is the information factually accurate?
2. Completeness: Does it cover all key points from ground truth?
3. Faithfulness: Is it grounded in retrieved context (no hallucination)?
4. Relevance: Does it directly answer the question?
5. Clarity: Is it well-written and easy to understand?

Provide your response in this exact JSON format:
{{
    "correctness": <1-5>,
    "completeness": <1-5>,
    "faithfulness": <1-5>,
    "relevance": <1-5>,
    "clarity": <1-5>,
    "overall_score": <1-5>,
    "verdict": "<pass|partial|fail>",
    "explanation": "<brief explanation>"
}}

Only output the JSON, nothing else."""

        try:
            response = self.judge_llm.invoke(prompt)
            content = response.content.strip()
            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content)
        except Exception as e:
            return {
                "correctness": 0, "completeness": 0, "faithfulness": 0,
                "relevance": 0, "clarity": 0, "overall_score": 0,
                "verdict": "error", "explanation": str(e)
            }
    
    def _process_question(self, q_id: str, cached: Dict, temperature: float, context_size: int, generator: GeminiAnswerGenerator) -> Dict:
        """Process a single question: generate + judge."""
        question = cached["question"]
        ground_truth = cached["ground_truth"]
        reranked = cached["reranked_docs"]
        
        # Build context
        context = self._build_context(reranked, context_size)
        
        # Generate answer
        config = self._get_query_config(precision_n=context_size)
        
        start = time.time()
        def do_generate():
            return generator.generate(question, context, config)
        
        try:
            gen_result = retry_with_backoff(do_generate)
            answer = gen_result.answer_text
        except Exception as e:
            answer = f"[Generation Error: {e}]"
        generation_time = time.time() - start
        
        # Judge answer
        start = time.time()
        def do_judge():
            return self._judge_answer(question, ground_truth, answer)
        
        try:
            judgment = retry_with_backoff(do_judge)
        except Exception as e:
            judgment = {"verdict": "error", "explanation": str(e), "overall_score": 0}
        judge_time = time.time() - start
        
        # Calculate metrics
        expected_source = cached.get("expected_source", "")
        retrieved_docs = [c["doc_name"] for c in cached["retrieved_docs"]]
        reranked_docs = [c["doc_name"] for c in reranked[:context_size]]
        
        # Recall@100
        recall_100 = any(expected_source.lower() in doc.lower() for doc in retrieved_docs) if expected_source else False
        
        # Precision@N (how many of top N are from expected source)
        if expected_source:
            precision_n = sum(1 for doc in reranked_docs if expected_source.lower() in doc.lower()) / len(reranked_docs) if reranked_docs else 0
        else:
            precision_n = 0
        
        return {
            "question_id": q_id,
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "answer_length": len(answer),
            "judgment": judgment,
            "recall_100": recall_100,
            "precision_n": precision_n,
            "generation_time": generation_time,
            "judge_time": judge_time,
            "retrieval_time": cached["retrieval_time"],
            "reranking_time": cached["reranking_time"],
        }
    
    def run_experiment(self, temperature: float, context_size: int, experiment_name: str) -> Dict:
        """Run a single experiment configuration."""
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"Temperature: {temperature}, Context Size: {context_size}")
        print(f"{'=' * 70}")
        
        # Checkpoint file
        checkpoint_file = self.experiment_dir / f"{experiment_name}_checkpoint.jsonl"
        results_file = self.experiment_dir / f"{experiment_name}_results.json"
        
        # Load existing checkpoint
        completed = {}
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                for line in f:
                    r = json.loads(line)
                    completed[r["question_id"]] = r
            print(f"Resuming from checkpoint: {len(completed)} completed")
        
        # Create generator with specified temperature
        generator = GeminiAnswerGenerator(
            default_model="gemini-2.5-flash",
            default_temperature=temperature,
        )
        
        # Process remaining questions
        remaining = [(q_id, cached) for q_id, cached in self.retrieval_cache.items() if q_id not in completed]
        
        if not remaining:
            print("All questions already processed!")
            results = list(completed.values())
        else:
            results = list(completed.values())
            
            with open(checkpoint_file, "a") as f:
                for q_id, cached in tqdm(remaining, desc=experiment_name):
                    result = self._process_question(q_id, cached, temperature, context_size, generator)
                    results.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
        
        # Aggregate metrics
        metrics = self._aggregate_metrics(results, temperature, context_size)
        
        # Save full results
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Generate report
        self._generate_report(metrics, experiment_name)
        
        return metrics
    
    def _aggregate_metrics(self, results: List[Dict], temperature: float, context_size: int) -> Dict:
        """Aggregate metrics from results."""
        valid = [r for r in results if r["judgment"].get("verdict") != "error"]
        
        # Recall@100
        recall_100 = sum(1 for r in results if r.get("recall_100", False)) / len(results) if results else 0
        
        # Precision@N
        precision_n = statistics.mean([r.get("precision_n", 0) for r in results]) if results else 0
        
        # LLM Judge metrics
        verdicts = [r["judgment"].get("verdict", "error") for r in results]
        pass_rate = verdicts.count("pass") / len(verdicts) if verdicts else 0
        
        judge_scores = {
            "correctness": [],
            "completeness": [],
            "faithfulness": [],
            "relevance": [],
            "clarity": [],
            "overall_score": [],
        }
        for r in valid:
            j = r["judgment"]
            for key in judge_scores:
                if key in j and isinstance(j[key], (int, float)) and j[key] > 0:
                    judge_scores[key].append(j[key])
        
        avg_scores = {k: statistics.mean(v) if v else 0 for k, v in judge_scores.items()}
        
        # Timing
        timing = {
            "retrieval": {
                "mean": statistics.mean([r["retrieval_time"] for r in results]),
                "total": sum(r["retrieval_time"] for r in results),
            },
            "reranking": {
                "mean": statistics.mean([r["reranking_time"] for r in results]),
                "total": sum(r["reranking_time"] for r in results),
            },
            "generation": {
                "mean": statistics.mean([r["generation_time"] for r in results]),
                "total": sum(r["generation_time"] for r in results),
            },
            "judge": {
                "mean": statistics.mean([r["judge_time"] for r in results]),
                "total": sum(r["judge_time"] for r in results),
            },
        }
        timing["total"] = {
            "mean": sum(t["mean"] for t in timing.values()),
            "total": sum(t["total"] for t in timing.values()),
        }
        
        return {
            "experiment": {
                "temperature": temperature,
                "context_size": context_size,
                "timestamp": datetime.now().isoformat(),
            },
            "questions": {
                "total": len(results),
                "valid": len(valid),
            },
            "retrieval": {
                "recall_100": recall_100,
                "precision_n": precision_n,
            },
            "llm_judge": {
                "pass_rate": pass_rate,
                "verdict_counts": {
                    "pass": verdicts.count("pass"),
                    "partial": verdicts.count("partial"),
                    "fail": verdicts.count("fail"),
                    "error": verdicts.count("error"),
                },
                "metrics": avg_scores,
            },
            "timing": timing,
            "detailed_results": results,
        }
    
    def _generate_report(self, metrics: Dict, experiment_name: str):
        """Generate markdown report for experiment."""
        exp = metrics["experiment"]
        ret = metrics["retrieval"]
        judge = metrics["llm_judge"]
        timing = metrics["timing"]
        
        report = f"""# {experiment_name.replace('_', ' ').title()} Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Configuration

| Setting | Value |
|---------|-------|
| Temperature | {exp['temperature']} |
| Context Size | {exp['context_size']} chunks |
| Questions | {metrics['questions']['total']} |
| Embedding | gemini-1536-RETRIEVAL_QUERY |
| Generation Model | gemini-2.5-flash |
| Judge Model | gemini-2.5-flash (temp 0.0) |

---

## Retrieval Metrics

| Metric | Value |
|--------|-------|
| Recall@100 | {ret['recall_100']*100:.1f}% |
| Precision@{exp['context_size']} | {ret['precision_n']*100:.1f}% |

---

## LLM Judge Results

### Pass Rate

| Verdict | Count | Percentage |
|---------|-------|------------|
| Pass | {judge['verdict_counts']['pass']} | {judge['verdict_counts']['pass']/metrics['questions']['total']*100:.1f}% |
| Partial | {judge['verdict_counts']['partial']} | {judge['verdict_counts']['partial']/metrics['questions']['total']*100:.1f}% |
| Fail | {judge['verdict_counts']['fail']} | {judge['verdict_counts']['fail']/metrics['questions']['total']*100:.1f}% |
| Error | {judge['verdict_counts']['error']} | {judge['verdict_counts']['error']/metrics['questions']['total']*100:.1f}% |

### Quality Scores (1-5)

| Dimension | Score |
|-----------|-------|
| Overall Score | {judge['metrics']['overall_score']:.2f} |
| Correctness | {judge['metrics']['correctness']:.2f} |
| Completeness | {judge['metrics']['completeness']:.2f} |
| Faithfulness | {judge['metrics']['faithfulness']:.2f} |
| Relevance | {judge['metrics']['relevance']:.2f} |
| Clarity | {judge['metrics']['clarity']:.2f} |

---

## Timing

| Phase | Avg per Query | Total |
|-------|---------------|-------|
| Retrieval | {timing['retrieval']['mean']:.3f}s | {timing['retrieval']['total']:.1f}s |
| Reranking | {timing['reranking']['mean']:.3f}s | {timing['reranking']['total']:.1f}s |
| Generation | {timing['generation']['mean']:.3f}s | {timing['generation']['total']:.1f}s |
| LLM Judge | {timing['judge']['mean']:.3f}s | {timing['judge']['total']:.1f}s |
| **Total** | {timing['total']['mean']:.3f}s | {timing['total']['total']:.1f}s |
"""
        
        report_file = self.reports_dir / f"{experiment_name}_Report.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"✓ Report saved: {report_file}")
    
    def generate_summary_report(self, all_results: Dict[str, Dict]):
        """Generate combined summary report."""
        report = f"""# Temperature & Context Size Sweep Summary

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Experiment Overview

- **Questions:** 224
- **Embedding:** gemini-1536-RETRIEVAL_QUERY
- **Generation Model:** gemini-2.5-flash
- **Judge Model:** gemini-2.5-flash (temp 0.0)
- **Retrieval:** Top 100 (hybrid 50/50), reranked

---

## Temperature Sweep Results (Context Size = 10)

| Temp | Pass Rate | Overall Score | Correctness | Completeness | Faithfulness | Gen Time |
|------|-----------|---------------|-------------|--------------|--------------|----------|
"""
        
        # Temperature results
        for temp in TEMPERATURES:
            key = f"temp_{temp}"
            if key in all_results:
                m = all_results[key]
                j = m["llm_judge"]
                t = m["timing"]
                report += f"| {temp} | {j['pass_rate']*100:.1f}% | {j['metrics']['overall_score']:.2f} | {j['metrics']['correctness']:.2f} | {j['metrics']['completeness']:.2f} | {j['metrics']['faithfulness']:.2f} | {t['generation']['mean']:.2f}s |\n"
        
        # Find best temperature
        best_temp = 0.0
        best_score = 0
        for temp in TEMPERATURES:
            key = f"temp_{temp}"
            if key in all_results:
                score = all_results[key]["llm_judge"]["metrics"]["overall_score"]
                if score > best_score:
                    best_score = score
                    best_temp = temp
        
        report += f"""
**Best Temperature: {best_temp}** (Overall Score: {best_score:.2f})

---

## Context Size Sweep Results (Temperature = {best_temp})

| Context | Pass Rate | Overall Score | Correctness | Completeness | Faithfulness | Gen Time |
|---------|-----------|---------------|-------------|--------------|--------------|----------|
"""
        
        # Context size results
        for size in CONTEXT_SIZES:
            key = f"context_{size}"
            if key in all_results:
                m = all_results[key]
                j = m["llm_judge"]
                t = m["timing"]
                report += f"| {size} | {j['pass_rate']*100:.1f}% | {j['metrics']['overall_score']:.2f} | {j['metrics']['correctness']:.2f} | {j['metrics']['completeness']:.2f} | {j['metrics']['faithfulness']:.2f} | {t['generation']['mean']:.2f}s |\n"
        
        # Find optimal context size
        best_context = 10
        best_context_score = 0
        for size in CONTEXT_SIZES:
            key = f"context_{size}"
            if key in all_results:
                score = all_results[key]["llm_judge"]["metrics"]["overall_score"]
                if score > best_context_score:
                    best_context_score = score
                    best_context = size
        
        report += f"""
**Optimal Context Size: {best_context}** (Overall Score: {best_context_score:.2f})

---

## Key Findings

1. **Best Temperature:** {best_temp}
2. **Optimal Context Size:** {best_context} chunks
3. **Diminishing Returns:** Check if scores decrease after optimal context size

---

## Timing Summary

| Experiment | Retrieval | Reranking | Generation | Judge | Total |
|------------|-----------|-----------|------------|-------|-------|
"""
        
        for name, m in all_results.items():
            t = m["timing"]
            report += f"| {name} | {t['retrieval']['total']:.0f}s | {t['reranking']['total']:.0f}s | {t['generation']['total']:.0f}s | {t['judge']['total']:.0f}s | {t['total']['total']:.0f}s |\n"
        
        report_file = self.reports_dir / "Temperature_Context_Sweep_Summary.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"\n✓ Summary report saved: {report_file}")
    
    def run_all(self):
        """Run all experiments."""
        print("\n" + "=" * 70)
        print("TEMPERATURE & CONTEXT SIZE SWEEP EXPERIMENTS")
        print("=" * 70)
        print(f"Temperatures: {TEMPERATURES}")
        print(f"Context Sizes: {CONTEXT_SIZES}")
        print(f"Questions: {len(self.corpus)}")
        print("=" * 70)
        
        # Phase 1: Retrieval + Reranking (cached)
        self.run_retrieval_phase()
        
        all_results = {}
        
        # Phase 2a: Temperature Sweep (context = 10)
        print("\n" + "=" * 70)
        print("PROJECT 1: TEMPERATURE SWEEP")
        print("=" * 70)
        
        for temp in TEMPERATURES:
            name = f"temp_{temp}"
            metrics = self.run_experiment(temp, context_size=10, experiment_name=name)
            all_results[name] = metrics
        
        # Find best temperature
        best_temp = 0.0
        best_score = 0
        for temp in TEMPERATURES:
            key = f"temp_{temp}"
            if key in all_results:
                score = all_results[key]["llm_judge"]["metrics"]["overall_score"]
                if score > best_score:
                    best_score = score
                    best_temp = temp
        
        print(f"\n✓ Best temperature: {best_temp} (score: {best_score:.2f})")
        
        # Phase 2b: Context Size Sweep (using best temperature)
        print("\n" + "=" * 70)
        print(f"PROJECT 2: CONTEXT SIZE SWEEP (temp={best_temp})")
        print("=" * 70)
        
        for size in CONTEXT_SIZES:
            name = f"context_{size}"
            metrics = self.run_experiment(best_temp, context_size=size, experiment_name=name)
            all_results[name] = metrics
        
        # Generate summary report
        self.generate_summary_report(all_results)
        
        print("\n" + "=" * 70)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 70)


if __name__ == "__main__":
    sweep = TemperatureContextSweep()
    sweep.run_all()
