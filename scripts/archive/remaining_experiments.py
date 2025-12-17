#!/usr/bin/env python3
"""
Remaining Experiments: context_50, context_100, and reasoning model sweep.

This script:
1. Resumes context_50 from checkpoint (18/224 done)
2. Runs context_100
3. Runs reasoning experiments with Gemini 2.5 Pro and 2.0 Flash Thinking
"""

import sys
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")

import json
import time
import random
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from vertexai.generative_models import GenerativeModel, GenerationConfig

# Orchestrator imports
from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
from services.api.retrieval.vector_search import VectorSearchRetriever
from services.api.ranking.google_ranker import GoogleRanker
from services.api.core.config import QueryConfig

# LLM Judge
from langchain_google_vertexai import ChatVertexAI

# Configuration
JOB_ID = "bfai__eval66a_g1_1536_tt"
CORPUS_FILE = "corpus/qa_corpus_200.json"
MAX_RETRIES = 5
BASE_DELAY = 1.0


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
                if attempt > 0:
                    raise
                delay = base_delay + random.uniform(0, 0.5)
                time.sleep(delay)
    raise last_exception


class RemainingExperiments:
    """Run remaining context and reasoning experiments."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.experiment_dir = self.base_dir / "experiments" / "2025-12-15_temp_context_sweep"
        self.reports_dir = self.base_dir / "reports"
        
        # Load job config
        jobs = get_jobs_config()
        self.job_config = jobs.get(JOB_ID, {})
        self.job_config["job_id"] = JOB_ID
        
        # Initialize components
        print(f"Initializing for job: {JOB_ID}")
        self.retriever = VectorSearchRetriever(self.job_config)
        self.ranker = GoogleRanker(project_id=PROJECT_ID)
        
        # LLM Judge (always temp 0.0)
        self.judge_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.0,
        )
        
        # Load corpus
        corpus_path = self.base_dir / CORPUS_FILE
        with open(corpus_path) as f:
            self.corpus = json.load(f)
        print(f"Loaded {len(self.corpus)} questions")
        
        # Load retrieval cache
        cache_file = self.experiment_dir / "retrieval_cache.json"
        with open(cache_file) as f:
            self.retrieval_cache = json.load(f)
        print(f"Loaded retrieval cache: {len(self.retrieval_cache)} questions")
    
    def _build_context(self, chunks: List[Dict], top_n: int) -> str:
        """Build context string from top N chunks."""
        context_parts = []
        for i, c in enumerate(chunks[:top_n]):
            text = c["text"][:2000]
            context_parts.append(f"[Document {i+1}: {c['doc_name']}]\n{text}")
        return "\n\n".join(context_parts)
    
    def _generate_answer(self, query: str, context: str, model: str, temperature: float, thinking_budget: Optional[int] = None) -> tuple:
        """Generate answer with optional reasoning/thinking budget.
        
        Returns:
            tuple: (answer, gen_time, token_info)
            token_info is a dict with prompt_tokens, completion_tokens, thinking_tokens
        """
        start = time.time()
        
        prompt = f"""Based on the following context, answer the question. Cite sources using [Document N] format.

Context:
{context}

Question: {query}

Answer:"""
        
        gen_model = GenerativeModel(model_name=model)
        
        # Build generation config
        config_dict = {
            "temperature": temperature,
            "max_output_tokens": 8192,
        }
        
        # Add thinking config if specified (for reasoning models)
        if thinking_budget is not None:
            try:
                from vertexai.generative_models import ThinkingConfig
                config_dict["thinking_config"] = ThinkingConfig(thinking_budget=thinking_budget)
            except ImportError:
                print(f"  Warning: ThinkingConfig not available, skipping thinking_budget")
        
        config = GenerationConfig(**config_dict)
        
        def do_generate():
            return gen_model.generate_content(prompt, generation_config=config)
        
        token_info = {"prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0}
        
        try:
            response = retry_with_backoff(do_generate)
            answer = response.text
            
            # Extract token counts from usage_metadata
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                usage = response.usage_metadata
                token_info["prompt_tokens"] = getattr(usage, 'prompt_token_count', 0) or 0
                token_info["completion_tokens"] = getattr(usage, 'candidates_token_count', 0) or 0
                # Check for thinking tokens (may be in thoughts_token_count for reasoning models)
                token_info["thinking_tokens"] = getattr(usage, 'thoughts_token_count', 0) or 0
        except Exception as e:
            answer = f"[Generation Error: {e}]"
        
        gen_time = time.time() - start
        return answer, gen_time, token_info
    
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
            def do_judge():
                return self.judge_llm.invoke(prompt)
            response = retry_with_backoff(do_judge)
            content = response.content.strip()
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
    
    def run_experiment(self, experiment_name: str, context_size: int, model: str = "gemini-2.5-flash", 
                       temperature: float = 0.0, thinking_budget: Optional[int] = None):
        """Run a single experiment."""
        print(f"\n{'=' * 70}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"Model: {model}, Temp: {temperature}, Context: {context_size}, Thinking: {thinking_budget}")
        print(f"{'=' * 70}")
        
        checkpoint_file = self.experiment_dir / f"{experiment_name}_checkpoint.jsonl"
        results_file = self.experiment_dir / f"{experiment_name}_results.json"
        
        # Load existing checkpoint
        completed = {}
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                for line in f:
                    r = json.loads(line)
                    completed[r["question_id"]] = r
            print(f"Resuming from checkpoint: {len(completed)}/224 completed")
        
        # Process remaining questions
        remaining = [(q_id, cached) for q_id, cached in self.retrieval_cache.items() if q_id not in completed]
        
        if not remaining:
            print("All questions already processed!")
            results = list(completed.values())
        else:
            results = list(completed.values())
            
            with open(checkpoint_file, "a") as f:
                for q_id, cached in tqdm(remaining, desc=experiment_name):
                    question = cached["question"]
                    ground_truth = cached["ground_truth"]
                    reranked = cached["reranked_docs"]
                    
                    # Build context
                    context = self._build_context(reranked, context_size)
                    
                    # Generate
                    answer, gen_time, token_info = self._generate_answer(question, context, model, temperature, thinking_budget)
                    
                    # Judge
                    start = time.time()
                    judgment = retry_with_backoff(lambda: self._judge_answer(question, ground_truth, answer))
                    judge_time = time.time() - start
                    
                    result = {
                        "question_id": q_id,
                        "question": question,
                        "ground_truth": ground_truth,
                        "answer": answer,
                        "answer_length": len(answer),
                        "judgment": judgment,
                        "generation_time": gen_time,
                        "judge_time": judge_time,
                        "retrieval_time": cached["retrieval_time"],
                        "reranking_time": cached["reranking_time"],
                        "prompt_tokens": token_info["prompt_tokens"],
                        "completion_tokens": token_info["completion_tokens"],
                        "thinking_tokens": token_info["thinking_tokens"],
                    }
                    results.append(result)
                    f.write(json.dumps(result) + "\n")
                    f.flush()
        
        # Aggregate and save
        metrics = self._aggregate_metrics(results, experiment_name, context_size, model, temperature, thinking_budget)
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        self._generate_report(metrics, experiment_name)
        return metrics
    
    def _aggregate_metrics(self, results: List[Dict], name: str, context_size: int, 
                           model: str, temperature: float, thinking_budget: Optional[int]) -> Dict:
        """Aggregate metrics from results."""
        valid = [r for r in results if r["judgment"].get("verdict") != "error"]
        
        verdicts = [r["judgment"].get("verdict", "error") for r in results]
        pass_rate = verdicts.count("pass") / len(verdicts) if verdicts else 0
        
        judge_scores = {
            "correctness": [], "completeness": [], "faithfulness": [],
            "relevance": [], "clarity": [], "overall_score": [],
        }
        for r in valid:
            j = r["judgment"]
            for key in judge_scores:
                if key in j and isinstance(j[key], (int, float)) and j[key] > 0:
                    judge_scores[key].append(j[key])
        
        avg_scores = {k: statistics.mean(v) if v else 0 for k, v in judge_scores.items()}
        
        timing = {
            "retrieval": {"mean": statistics.mean([r["retrieval_time"] for r in results])},
            "reranking": {"mean": statistics.mean([r["reranking_time"] for r in results])},
            "generation": {"mean": statistics.mean([r["generation_time"] for r in results])},
            "judge": {"mean": statistics.mean([r["judge_time"] for r in results])},
        }
        
        return {
            "experiment": {
                "name": name,
                "model": model,
                "temperature": temperature,
                "context_size": context_size,
                "thinking_budget": thinking_budget,
                "timestamp": datetime.now().isoformat(),
            },
            "questions": {"total": len(results), "valid": len(valid)},
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
        """Generate markdown report."""
        exp = metrics["experiment"]
        judge = metrics["llm_judge"]
        timing = metrics["timing"]
        
        thinking_str = f", Thinking Budget: {exp['thinking_budget']}" if exp['thinking_budget'] else ""
        
        report = f"""# {experiment_name.replace('_', ' ').title()} Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Configuration

| Setting | Value |
|---------|-------|
| Model | {exp['model']} |
| Temperature | {exp['temperature']} |
| Context Size | {exp['context_size']} chunks |
| Thinking Budget | {exp['thinking_budget'] or 'N/A'} |
| Questions | {metrics['questions']['total']} |

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

| Phase | Avg per Query |
|-------|---------------|
| Retrieval | {timing['retrieval']['mean']:.3f}s |
| Reranking | {timing['reranking']['mean']:.3f}s |
| Generation | {timing['generation']['mean']:.3f}s |
| LLM Judge | {timing['judge']['mean']:.3f}s |
"""
        
        report_file = self.reports_dir / f"{experiment_name}_Report.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"✓ Report saved: {report_file}")
    
    def run_all(self):
        """Run context_50 and context_100 only (reasoning experiments later)."""
        print("\n" + "=" * 70)
        print("REMAINING CONTEXT EXPERIMENTS")
        print("=" * 70)
        
        # 1. Context 50 (resume from checkpoint - 18/224 done)
        self.run_experiment("context_50", context_size=50)
        
        # 2. Context 100
        self.run_experiment("context_100", context_size=100)
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n" + "=" * 70)
        print("CONTEXT EXPERIMENTS COMPLETE")
        print("=" * 70)
    
    def generate_summary_report(self):
        """Generate combined summary report for all experiments."""
        print("\nGenerating summary report...")
        
        # Load all results
        all_results = {}
        for f in self.experiment_dir.glob("*_results.json"):
            name = f.stem.replace("_results", "")
            with open(f) as fp:
                all_results[name] = json.load(fp)
        
        # Build report
        report = f"""# Temperature & Context Size Sweep - Complete Results

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## Experiment Overview

- **Questions:** 224
- **Embedding:** gemini-1536-RETRIEVAL_QUERY  
- **Generation Model:** gemini-2.5-flash
- **Judge Model:** gemini-2.5-flash (temp 0.0)
- **Retrieval:** Top 100 (hybrid 50/50)

---

## Temperature Sweep Results (Context Size = 10)

| Temp | Pass Rate | Overall | Correctness | Completeness | Faithfulness | Gen Time |
|------|-----------|---------|-------------|--------------|--------------|----------|
"""
        
        for temp in [0.0, 0.1, 0.2, 0.3]:
            key = f"temp_{temp}"
            if key in all_results:
                m = all_results[key]
                j = m["llm_judge"]
                t = m["timing"]
                report += f"| {temp} | {j['pass_rate']*100:.1f}% | {j['metrics']['overall_score']:.2f} | {j['metrics']['correctness']:.2f} | {j['metrics']['completeness']:.2f} | {j['metrics']['faithfulness']:.2f} | {t['generation']['mean']:.2f}s |\n"
        
        report += """
**Finding:** Temperature has minimal impact (0.0-0.3 all produce ~same results). Use 0.0 for determinism.

---

## Context Size Sweep Results (Temperature = 0.0)

| Context | Pass Rate | Overall | Correctness | Completeness | Faithfulness | Gen Time |
|---------|-----------|---------|-------------|--------------|--------------|----------|
"""
        
        for size in [5, 10, 15, 20, 25, 50, 100]:
            key = f"context_{size}"
            if key in all_results:
                m = all_results[key]
                j = m["llm_judge"]
                t = m["timing"]
                report += f"| {size} | {j['pass_rate']*100:.1f}% | {j['metrics']['overall_score']:.2f} | {j['metrics']['correctness']:.2f} | {j['metrics']['completeness']:.2f} | {j['metrics']['faithfulness']:.2f} | {t['generation']['mean']:.2f}s |\n"
        
        # Find best context size
        best_size = 5
        best_score = 0
        for size in [5, 10, 15, 20, 25, 50, 100]:
            key = f"context_{size}"
            if key in all_results:
                score = all_results[key]["llm_judge"]["metrics"]["overall_score"]
                if score > best_score:
                    best_score = score
                    best_size = size
        
        report += f"""
**Best Context Size:** {best_size} chunks (Overall Score: {best_score:.2f})

---

## Key Findings

### 1. Temperature Has No Meaningful Impact
All temperatures (0.0-0.3) produced virtually identical results. Stick with **temp 0.0** for deterministic outputs.

### 2. More Context = Better Answers
Quality improves consistently as context size increases:
- 5 chunks → 25 chunks: Pass rate improves significantly
- Completeness benefits most from more context

### 3. Optimal Configuration
- **Temperature:** 0.0
- **Context Size:** {best_size} chunks (pending 50/100 results)

---

## Timing Summary

| Experiment | Gen Time (avg) |
|------------|----------------|
"""
        
        for name in sorted(all_results.keys()):
            m = all_results[name]
            t = m["timing"]["generation"]["mean"]
            report += f"| {name} | {t:.2f}s |\n"
        
        report_file = self.reports_dir / "Temperature_Context_Sweep_Summary.md"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"✓ Summary report saved: {report_file}")


if __name__ == "__main__":
    exp = RemainingExperiments()
    exp.run_all()
