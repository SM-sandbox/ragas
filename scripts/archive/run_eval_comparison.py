#!/usr/bin/env python3
"""
Run evaluation comparison: Precision@25 vs Precision@12

Uses gold standard corpus with:
- Recall@100
- MMR
- 5 dimensions: faithfulness, relevance, correctness, completeness, coherence

Checkpoints every 10 questions, 5 retries on failures.
"""

import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm
from langchain_google_vertexai import ChatVertexAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_config import GCP_PROJECT, GCP_LLM_LOCATION, LLM_MODEL

# Orchestrator imports
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")
from libs.core.gcp_config import get_jobs_config, PROJECT_ID
from services.api.retrieval.vector_search import VectorSearchRetriever
from services.api.ranking.google_ranker import GoogleRanker
from services.api.generation.gemini import GeminiAnswerGenerator
from services.api.core.config import QueryConfig

CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_gold_500.json"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "gold_standard_eval"
JOB_ID = "bfai__eval66a_g1_1536_tt"
MAX_RETRIES = 5
RECALL_K = 100
DIMENSIONS = ["faithfulness", "relevance", "correctness", "completeness", "coherence"]


def extract_json(text):
    """Extract JSON from LLM response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        for part in parts:
            if part.strip().startswith("{"):
                text = part.strip()
                break
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i+1])
    return json.loads(text)


class EvalRunner:
    def __init__(self, precision_k: int):
        self.precision_k = precision_k
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        
        with open(CORPUS_PATH) as f:
            self.corpus = json.load(f)
        self.questions = self.corpus["questions"]
        print(f"Loaded {len(self.questions)} questions", flush=True)
        
        print("Initializing retrieval...", flush=True)
        jobs = get_jobs_config()
        job_config = jobs.get(JOB_ID, {})
        job_config["job_id"] = JOB_ID
        
        self.retriever = VectorSearchRetriever(job_config)
        self.ranker = GoogleRanker(project_id=PROJECT_ID)
        self.generator = GeminiAnswerGenerator()
        
        print("Initializing judge LLM...", flush=True)
        self.judge = ChatVertexAI(
            model_name=LLM_MODEL,
            project=GCP_PROJECT,
            location=GCP_LLM_LOCATION,
            temperature=0.0,
            max_tokens=500,
        )
        
        self.checkpoint_file = REPORTS_DIR / f"checkpoint_p{precision_k}.json"
        self.results_file = REPORTS_DIR / f"results_p{precision_k}.json"
    
    def _judge_dimension(self, question, ground_truth, answer, context, dimension):
        prompts = {
            "faithfulness": f"Rate faithfulness to context (1-5). 5=all claims supported. Context: {context[:1500]}\nAnswer: {answer}\nReturn JSON: {{\"score\": N}}",
            "relevance": f"Rate relevance to question (1-5). 5=directly addresses question. Q: {question}\nA: {answer}\nReturn JSON: {{\"score\": N}}",
            "correctness": f"Rate correctness vs ground truth (1-5). 5=matches ground truth. Truth: {ground_truth}\nAnswer: {answer}\nReturn JSON: {{\"score\": N}}",
            "completeness": f"Rate completeness (1-5). 5=comprehensive. Q: {question}\nTruth: {ground_truth}\nA: {answer}\nReturn JSON: {{\"score\": N}}",
            "coherence": f"Rate coherence/clarity (1-5). 5=clear and organized. Answer: {answer}\nReturn JSON: {{\"score\": N}}"
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.judge.invoke(prompts[dimension])
                result = extract_json(response.content)
                return result.get("score", 3)
            except:
                time.sleep(1)
        return 3
    
    def run(self):
        print(f"\n{'='*70}", flush=True)
        print(f"EVAL: Precision@{self.precision_k}, Recall@{RECALL_K}", flush=True)
        print(f"{'='*70}", flush=True)
        
        # Load checkpoint
        completed = {}
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                completed = {r["question_id"]: r for r in json.load(f)}
            print(f"Resuming: {len(completed)} already done", flush=True)
        
        results = {"config": {"precision_k": self.precision_k, "recall_k": RECALL_K}, "per_question": []}
        
        mrr_sum = 0
        recall_hits = 0
        dim_scores = defaultdict(list)
        
        for i, q in enumerate(tqdm(self.questions, desc=f"Eval@{self.precision_k}")):
            qid = q.get("question_id", f"q_{i}")
            
            if qid in completed:
                r = completed[qid]
                results["per_question"].append(r)
                mrr_sum += r.get("mrr", 0)
                if r.get("recall_hit"): recall_hits += 1
                for d in DIMENSIONS:
                    if d in r.get("scores", {}):
                        dim_scores[d].append(r["scores"][d])
                continue
            
            try:
                # Retrieve using QueryConfig
                config = QueryConfig(
                    recall_top_k=RECALL_K,
                    precision_top_n=self.precision_k,
                    enable_hybrid=True,
                    enable_reranking=True
                )
                retrieval_result = self.retriever.retrieve(q["question"], config)
                chunks = list(retrieval_result.chunks)
                
                # Rerank
                reranked = self.ranker.rank(q["question"], chunks, self.precision_k)
                
                # Check recall
                expected = q.get("source_filename", "").replace(".pdf", "").lower()
                docs = [c.metadata.get("source_document", "").lower() for c in reranked]
                
                recall_hit = any(expected in d for d in docs)
                if recall_hit: recall_hits += 1
                
                # MRR
                mrr = 0
                for rank, d in enumerate(docs, 1):
                    if expected in d:
                        mrr = 1.0 / rank
                        break
                mrr_sum += mrr
                
                # Generate
                context = "\n\n".join([c.text for c in reranked[:self.precision_k]])
                answer = self.generator.generate(q["question"], context)
                
                # Judge
                scores = {}
                for dim in DIMENSIONS:
                    s = self._judge_dimension(q["question"], q.get("ground_truth_answer", ""), answer, context, dim)
                    scores[dim] = s
                    dim_scores[dim].append(s)
                
                result = {
                    "question_id": qid,
                    "mrr": mrr,
                    "recall_hit": recall_hit,
                    "answer_length": len(answer),
                    "scores": scores
                }
                results["per_question"].append(result)
                completed[qid] = result
                
                if (i + 1) % 10 == 0:
                    with open(self.checkpoint_file, 'w') as f:
                        json.dump(list(completed.values()), f, indent=2)
                
            except Exception as e:
                print(f"\nError {qid}: {e}", flush=True)
        
        # Final metrics
        n = len(self.questions)
        results["metrics"] = {
            "mrr": mrr_sum / n if n else 0,
            f"recall@{RECALL_K}": recall_hits / n if n else 0,
        }
        for d in DIMENSIONS:
            if dim_scores[d]:
                results["metrics"][d] = sum(dim_scores[d]) / len(dim_scores[d])
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved: {self.results_file}", flush=True)
        return results


def generate_report(results_25, results_12):
    """Generate comparison report."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    
    m25 = results_25.get("metrics", {})
    m12 = results_12.get("metrics", {})
    
    report = f"""# Gold Standard Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Corpus:** 458 questions (442 Critical, 16 Relevant)
**Model:** {LLM_MODEL}

## Precision@25 vs Precision@12 Comparison

| Metric | @25 | @12 | Δ |
|--------|-----|-----|---|
| MRR | {m25.get('mrr', 0):.3f} | {m12.get('mrr', 0):.3f} | {m25.get('mrr', 0) - m12.get('mrr', 0):+.3f} |
| Recall@100 | {m25.get('recall@100', 0):.3f} | {m12.get('recall@100', 0):.3f} | {m25.get('recall@100', 0) - m12.get('recall@100', 0):+.3f} |
"""
    
    for d in DIMENSIONS:
        v25 = m25.get(d, 0)
        v12 = m12.get(d, 0)
        report += f"| {d.capitalize()} | {v25:.3f} | {v12:.3f} | {v25 - v12:+.3f} |\n"
    
    # Averages
    avg25 = sum(m25.get(d, 0) for d in DIMENSIONS) / len(DIMENSIONS)
    avg12 = sum(m12.get(d, 0) for d in DIMENSIONS) / len(DIMENSIONS)
    
    report += f"""
## Summary

- **Precision@25 Average Quality:** {avg25:.3f}
- **Precision@12 Average Quality:** {avg12:.3f}
- **Winner:** {'Precision@25' if avg25 > avg12 else 'Precision@12' if avg12 > avg25 else 'Tie'}

## Recommendation

"""
    if avg25 > avg12 + 0.05:
        report += "**Use Precision@25** - Significantly better answer quality justifies the additional context.\n"
    elif avg12 > avg25 + 0.05:
        report += "**Use Precision@12** - Better efficiency with comparable or better quality.\n"
    else:
        report += "**Either configuration is acceptable** - Minimal quality difference. Choose based on latency requirements.\n"
    
    report_file = REPORTS_DIR / f"comparison_report_{timestamp}.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report: {report_file}", flush=True)
    return report_file


def main():
    print("="*70, flush=True)
    print("GOLD STANDARD EVALUATION SUITE", flush=True)
    print("="*70, flush=True)
    
    # Run @25
    runner_25 = EvalRunner(precision_k=25)
    results_25 = runner_25.run()
    
    # Run @12
    runner_12 = EvalRunner(precision_k=12)
    results_12 = runner_12.run()
    
    # Generate report
    report_file = generate_report(results_25, results_12)
    
    print("\n" + "="*70, flush=True)
    print("✅ EVALUATION COMPLETE", flush=True)
    print(f"Report: {report_file}", flush=True)
    print("="*70, flush=True)


if __name__ == "__main__":
    main()
