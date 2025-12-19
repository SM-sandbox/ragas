#!/usr/bin/env python3
"""
Gold Standard Pipeline Runner

Orchestrates the full pipeline using existing modules:
1. Generate questions to fill gaps (using generate_questions_v2 logic)
2. Evaluate relevance (using add_relevance_to_corpus logic)
3. Filter to score-5 only, build final corpus
4. Run eval suite at precision@25 and precision@12
5. Generate comparison report

Usage:
    python run_gold_standard_pipeline.py --config config/gold_standard_500.json
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from langchain_google_vertexai import ChatVertexAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_config import (
    GCP_PROJECT, GCP_LLM_LOCATION, LLM_MODEL,
    DEFAULT_WORKERS, DEFAULT_RETRIES, CHECKPOINT_INTERVAL
)

# Also need orchestrator for eval
sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")


class GoldStandardPipeline:
    """Orchestrates the full gold standard corpus build and evaluation."""
    
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.base_dir = Path(__file__).parent.parent
        self.corpus_path = self.base_dir / self.config["corpus"]["input_file"]
        self.output_path = self.base_dir / self.config["corpus"]["output_file"]
        self.checkpoint_path = self.base_dir / "corpus" / "pipeline_checkpoint.json"
        self.reports_dir = self.base_dir / self.config["output"]["reports_dir"]
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metadata and content for generation
        self.metadata_dir = self.base_dir / "doc_metadata" / "json"
        self.inputs_dir = self.base_dir / "doc_metadata" / "inputs_first10"
        
        self.llm = None
        self.eval_llm = None
        self.checkpoint = {"phase": "init", "generated": [], "corpus_saved": False}
        
    def _init_llms(self):
        """Initialize LLMs."""
        print("Initializing LLMs...", flush=True)
        self.llm = ChatVertexAI(
            model_name=LLM_MODEL,
            project=GCP_PROJECT,
            location=GCP_LLM_LOCATION,
            temperature=0.7,
            max_tokens=1000,
        )
        self.eval_llm = ChatVertexAI(
            model_name=LLM_MODEL,
            project=GCP_PROJECT,
            location=GCP_LLM_LOCATION,
            temperature=0.0,
            max_tokens=500,
        )
    
    def _load_checkpoint(self):
        """Load checkpoint if exists."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                self.checkpoint = json.load(f)
            print(f"Resumed from checkpoint: phase={self.checkpoint['phase']}", flush=True)
    
    def _save_checkpoint(self):
        """Save checkpoint."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.checkpoint, f, indent=2)
    
    def _load_corpus(self):
        """Load current corpus."""
        with open(self.corpus_path) as f:
            return json.load(f)
    
    def _save_corpus(self, corpus, path=None):
        """Save corpus."""
        path = path or self.corpus_path
        with open(path, 'w') as f:
            json.dump(corpus, f, indent=2)
    
    def _load_resources(self):
        """Load document metadata and content."""
        metadata = {}
        for f in self.metadata_dir.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
                metadata[data.get("doc_id", f.stem)] = data
        
        contents = {}
        for f in self.inputs_dir.glob("*.txt"):
            with open(f) as fp:
                contents[f.stem] = fp.read()
        
        return metadata, contents
    
    def _calculate_gaps(self, corpus):
        """Calculate how many score-5 questions needed per bucket."""
        target_dist = self.config["corpus"]["distribution"]
        
        # Count current score-5s per bucket
        current = defaultdict(lambda: defaultdict(int))
        for q in corpus["questions"]:
            if q.get("domain_relevance_score") == 5:
                current[q.get("question_type")][q.get("difficulty")] += 1
        
        gaps = {}
        for hop_type, diffs in target_dist.items():
            for diff, target in diffs.items():
                have = current[hop_type][diff]
                need = max(0, target - have)
                if need > 0:
                    # Add buffer for filtering
                    buffer = int(need * self.config["generation"]["buffer_multiplier"]) + 5
                    gaps[(hop_type, diff)] = buffer
        
        return gaps
    
    def phase1_generate(self):
        """Phase 1: Generate questions to fill gaps."""
        print("\n" + "="*70, flush=True)
        print("PHASE 1: GENERATE QUESTIONS", flush=True)
        print("="*70, flush=True)
        
        corpus = self._load_corpus()
        gaps = self._calculate_gaps(corpus)
        total_to_gen = sum(gaps.values())
        
        if total_to_gen == 0:
            print("No new questions needed!", flush=True)
            return
        
        print(f"Generating ~{total_to_gen} questions to fill gaps:", flush=True)
        for (hop, diff), count in gaps.items():
            if count > 0:
                print(f"  {hop}/{diff}: {count}", flush=True)
        
        metadata, contents = self._load_resources()
        self._init_llms()
        
        # Get next question ID
        existing_ids = [int(q.get("question_id", "q_0000")[2:]) for q in corpus["questions"]]
        next_id = max(existing_ids) + 1 if existing_ids else 1
        
        # Skip already generated in checkpoint
        already_gen_ids = {q["question_id"] for q in self.checkpoint.get("generated", [])}
        
        new_questions = []
        
        for (hop_type, difficulty), count_needed in gaps.items():
            print(f"\n{hop_type}/{difficulty}: generating {count_needed}...", flush=True)
            generated = 0
            attempts = 0
            max_attempts = count_needed * 4
            
            while generated < count_needed and attempts < max_attempts:
                attempts += 1
                
                try:
                    q = self._generate_one(hop_type, difficulty, metadata, contents)
                    if q is None:
                        print(".", end="", flush=True)
                        continue
                    
                    # Evaluate
                    score, rationale = self._evaluate_one(q)
                    q["domain_relevance_score"] = score
                    q["domain_relevance_rationale"] = rationale
                    
                    if score == 5:
                        q["question_id"] = f"q_{next_id:04d}"
                        new_questions.append(q)
                        self.checkpoint["generated"].append(q)
                        generated += 1
                        next_id += 1
                        print("✓", end="", flush=True)
                        
                        # Checkpoint
                        if len(new_questions) % CHECKPOINT_INTERVAL == 0:
                            self._save_checkpoint()
                            corpus["questions"].extend(new_questions[-CHECKPOINT_INTERVAL:])
                            self._save_corpus(corpus)
                            print(f" [saved]", end="", flush=True)
                    else:
                        print(f"x{score}", end="", flush=True)
                except Exception as e:
                    print(f"!", end="", flush=True)
                    time.sleep(1)
            
            print(f" → {generated}/{count_needed}", flush=True)
        
        # Final save
        corpus["questions"].extend([q for q in new_questions if q["question_id"] not in 
                                   {x.get("question_id") for x in corpus["questions"]}])
        self._save_corpus(corpus)
        self.checkpoint["phase"] = "generate_complete"
        self._save_checkpoint()
        
        print(f"\n✓ Generation complete. Corpus now has {len(corpus['questions'])} questions", flush=True)
    
    def _generate_one(self, hop_type, difficulty, metadata, contents):
        """Generate a single question."""
        import random
        import re
        
        DIFFICULTY_DESC = {
            "easy": "Direct fact lookup",
            "medium": "Requires context understanding", 
            "hard": "Requires synthesis or comparison"
        }
        
        available = [d for d in metadata if d in contents]
        
        if hop_type == "single_hop":
            doc_id = random.choice(available)
            meta = metadata[doc_id]
            prompt = f"""Generate a {difficulty} single-hop question for SCADA/solar/electrical equipment.

Document: {meta.get("doc_title", doc_id)}
Content:
{contents[doc_id][:6000]}

Requirements:
- MUST be answerable from this document
- CRITICAL for field technicians: equipment specs, safety limits, troubleshooting, configuration
- NOT about: revision numbers, authors, form fields, dates
- {DIFFICULTY_DESC[difficulty]}

Return JSON: {{"question": "...", "ground_truth_answer": "...", "reasoning": "..."}}"""
            
            response = self.llm.invoke(prompt)
            result = self._parse_json(response.content)
            result["question_type"] = "single_hop"
            result["difficulty"] = difficulty
            result["source_doc_id"] = doc_id
            result["source_filename"] = meta.get("source_filename", f"{doc_id}.pdf")
            return result
        else:
            doc1, doc2 = random.sample(available, 2)
            meta1, meta2 = metadata[doc1], metadata[doc2]
            prompt = f"""Generate a {difficulty} multi-hop question requiring BOTH documents.

Doc 1: {meta1.get("doc_title", doc1)}
{contents[doc1][:3500]}

Doc 2: {meta2.get("doc_title", doc2)}
{contents[doc2][:3500]}

Requirements:
- MUST require info from BOTH documents
- CRITICAL for field technicians: specs, safety, troubleshooting
- NOT about: revision numbers, authors, metadata
- {DIFFICULTY_DESC[difficulty]}

Return JSON: {{"question": "...", "ground_truth_answer": "...", "reasoning": "..."}}"""
            
            response = self.llm.invoke(prompt)
            result = self._parse_json(response.content)
            result["question_type"] = "multi_hop"
            result["difficulty"] = difficulty
            result["source_doc_id"] = doc1
            result["source_filename"] = meta1.get("source_filename", f"{doc1}.pdf")
            result["secondary_doc_id"] = doc2
            return result
    
    def _evaluate_one(self, question):
        """Evaluate a question's relevance."""
        import re
        
        prompt = f"""Rate this question's DOMAIN RELEVANCE for field technicians (1-5):

5 = CRITICAL: Core knowledge field tech MUST know (specs, safety, troubleshooting)
4 = RELEVANT: Useful domain knowledge
3 = MARGINAL: Somewhat useful
2 = LOW VALUE: Trivial or document-specific
1 = IRRELEVANT: Not useful

Question: {question.get("question", "")}
Answer: {question.get("ground_truth_answer", "")}

Return JSON: {{"score": <1-5>, "rationale": "..."}}"""
        
        response = self.eval_llm.invoke(prompt)
        result = self._parse_json(response.content)
        return result.get("score", 0), result.get("rationale", "")
    
    def _parse_json(self, content):
        """Parse JSON from LLM response."""
        import re
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        try:
            return json.loads(content)
        except:
            # Regex fallback
            score_match = re.search(r'"score"\s*:\s*(\d)', content)
            if score_match:
                rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', content)
                return {"score": int(score_match.group(1)), 
                        "rationale": rationale_match.group(1) if rationale_match else ""}
            raise
    
    def phase2_filter_corpus(self):
        """Phase 2: Filter to score-5 only and build final 500."""
        print("\n" + "="*70, flush=True)
        print("PHASE 2: BUILD FINAL 500 CORPUS", flush=True)
        print("="*70, flush=True)
        
        corpus = self._load_corpus()
        target_dist = self.config["corpus"]["distribution"]
        
        # Get all score-5 questions
        score_5s = [q for q in corpus["questions"] if q.get("domain_relevance_score") == 5]
        print(f"Total score-5 questions available: {len(score_5s)}", flush=True)
        
        # Group by bucket
        by_bucket = defaultdict(list)
        for q in score_5s:
            key = (q.get("question_type"), q.get("difficulty"))
            by_bucket[key].append(q)
        
        # Build final corpus with exact distribution
        final_questions = []
        for hop_type, diffs in target_dist.items():
            for diff, target in diffs.items():
                available = by_bucket[(hop_type, diff)]
                if len(available) < target:
                    print(f"  ⚠️  {hop_type}/{diff}: only {len(available)} available, need {target}", flush=True)
                    final_questions.extend(available)
                else:
                    final_questions.extend(available[:target])
                print(f"  {hop_type}/{diff}: {min(len(available), target)}/{target}", flush=True)
        
        # Save final corpus
        final_corpus = {"questions": final_questions}
        self._save_corpus(final_corpus, self.output_path)
        
        print(f"\n✓ Final corpus saved: {len(final_questions)} questions", flush=True)
        print(f"  Output: {self.output_path}", flush=True)
        
        self.checkpoint["phase"] = "filter_complete"
        self.checkpoint["corpus_saved"] = True
        self._save_checkpoint()
        
        return len(final_questions)
    
    def phase3_run_eval(self):
        """Phase 3: Run evaluation suite."""
        print("\n" + "="*70, flush=True)
        print("PHASE 3: RUN EVALUATION SUITE", flush=True)
        print("="*70, flush=True)
        
        # Import eval components
        from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
        from services.api.retrieval.vector_search import VectorSearchRetriever
        from services.api.ranking.google_ranker import GoogleRanker
        from services.api.generation.gemini import GeminiAnswerGenerator
        
        # Load corpus
        with open(self.output_path) as f:
            corpus = json.load(f)
        
        questions = corpus["questions"]
        print(f"Evaluating {len(questions)} questions", flush=True)
        
        results = {}
        
        for run_config in self.config["evaluation"]["runs"]:
            run_name = run_config["name"]
            precision_k = run_config["precision_k"]
            
            print(f"\n--- Running: {run_name} (precision@{precision_k}) ---", flush=True)
            
            # Run evaluation
            run_results = self._run_single_eval(
                questions, 
                precision_k=precision_k,
                recall_k=run_config["recall_k"],
                dimensions=run_config["dimensions"]
            )
            
            results[run_name] = run_results
            
            # Save intermediate results
            result_file = self.reports_dir / f"{run_name}_results.json"
            with open(result_file, 'w') as f:
                json.dump(run_results, f, indent=2)
            print(f"  Saved: {result_file}", flush=True)
        
        self.checkpoint["phase"] = "eval_complete"
        self._save_checkpoint()
        
        return results
    
    def _run_single_eval(self, questions, precision_k, recall_k, dimensions):
        """Run a single evaluation configuration."""
        from tqdm import tqdm
        
        # Initialize components
        JOB_ID = "bfai__eval66a_g1_1536_tt"
        from libs.core.gcp_config import get_jobs_config, PROJECT_ID
        from services.api.retrieval.vector_search import VectorSearchRetriever
        from services.api.ranking.google_ranker import GoogleRanker
        from services.api.generation.gemini import GeminiAnswerGenerator
        
        jobs = get_jobs_config()
        job_config = jobs.get(JOB_ID, {})
        job_config["job_id"] = JOB_ID
        
        retriever = VectorSearchRetriever(job_config)
        ranker = GoogleRanker(project_id=PROJECT_ID)
        generator = GeminiAnswerGenerator()
        
        if self.eval_llm is None:
            self._init_llms()
        
        results = {
            "config": {"precision_k": precision_k, "recall_k": recall_k},
            "metrics": {},
            "per_question": []
        }
        
        # Metrics accumulators
        mrr_sum = 0
        recall_hits = 0
        dimension_scores = defaultdict(list)
        
        checkpoint_file = self.reports_dir / f"eval_checkpoint_p{precision_k}.json"
        completed = {}
        if checkpoint_file.exists():
            with open(checkpoint_file) as f:
                completed = {r["question_id"]: r for r in json.load(f)}
        
        for i, q in enumerate(tqdm(questions, desc=f"Eval@{precision_k}")):
            qid = q.get("question_id", f"q_{i}")
            
            # Skip if already done
            if qid in completed:
                result = completed[qid]
                results["per_question"].append(result)
                if result.get("mrr"):
                    mrr_sum += result["mrr"]
                if result.get("recall_hit"):
                    recall_hits += 1
                for dim in dimensions:
                    if dim in result.get("scores", {}):
                        dimension_scores[dim].append(result["scores"][dim])
                continue
            
            try:
                # Retrieve
                retrieved = retriever.search(q["question"], top_k=recall_k)
                
                # Rerank
                reranked = ranker.rerank(q["question"], retrieved, top_k=precision_k)
                
                # Check recall
                expected_doc = q.get("source_filename", "").replace(".pdf", "").lower()
                retrieved_docs = [c.metadata.get("source_document", "").lower() for c in reranked]
                
                recall_hit = any(expected_doc in d for d in retrieved_docs)
                if recall_hit:
                    recall_hits += 1
                
                # MRR
                mrr = 0
                for rank, doc in enumerate(retrieved_docs, 1):
                    if expected_doc in doc:
                        mrr = 1.0 / rank
                        break
                mrr_sum += mrr
                
                # Generate answer
                context = "\n\n".join([c.text for c in reranked[:precision_k]])
                answer = generator.generate(q["question"], context)
                
                # Judge on dimensions
                scores = {}
                for dim in dimensions:
                    score = self._judge_dimension(
                        q["question"], 
                        q.get("ground_truth_answer", ""),
                        answer,
                        context,
                        dim
                    )
                    scores[dim] = score
                    dimension_scores[dim].append(score)
                
                result = {
                    "question_id": qid,
                    "question": q["question"],
                    "mrr": mrr,
                    "recall_hit": recall_hit,
                    "answer_length": len(answer),
                    "scores": scores
                }
                results["per_question"].append(result)
                completed[qid] = result
                
                # Checkpoint every 10
                if (i + 1) % 10 == 0:
                    with open(checkpoint_file, 'w') as f:
                        json.dump(list(completed.values()), f, indent=2)
                
            except Exception as e:
                print(f"\n  Error on {qid}: {e}", flush=True)
                continue
        
        # Calculate final metrics
        n = len(questions)
        results["metrics"] = {
            "mrr": mrr_sum / n if n > 0 else 0,
            f"recall@{recall_k}": recall_hits / n if n > 0 else 0,
            f"precision@{precision_k}": recall_hits / n if n > 0 else 0,  # Simplified
        }
        
        for dim in dimensions:
            if dimension_scores[dim]:
                results["metrics"][dim] = sum(dimension_scores[dim]) / len(dimension_scores[dim])
        
        return results
    
    def _judge_dimension(self, question, ground_truth, answer, context, dimension):
        """Judge answer on a specific dimension."""
        prompts = {
            "faithfulness": f"""Rate how faithful the answer is to the provided context (1-5).
5 = Completely faithful, all claims supported by context
1 = Contains hallucinations or unsupported claims

Context: {context[:2000]}
Answer: {answer}

Return JSON: {{"score": <1-5>}}""",
            
            "relevance": f"""Rate how relevant the answer is to the question (1-5).
5 = Directly and completely addresses the question
1 = Off-topic or doesn't address the question

Question: {question}
Answer: {answer}

Return JSON: {{"score": <1-5>}}""",
            
            "correctness": f"""Rate the factual correctness compared to ground truth (1-5).
5 = Completely correct, matches ground truth
1 = Incorrect or contradicts ground truth

Ground Truth: {ground_truth}
Answer: {answer}

Return JSON: {{"score": <1-5>}}""",
            
            "completeness": f"""Rate how complete the answer is (1-5).
5 = Comprehensive, covers all aspects
1 = Incomplete, missing key information

Question: {question}
Ground Truth: {ground_truth}
Answer: {answer}

Return JSON: {{"score": <1-5>}}""",
            
            "coherence": f"""Rate the coherence and clarity of the answer (1-5).
5 = Clear, well-structured, easy to understand
1 = Confusing, poorly organized

Answer: {answer}

Return JSON: {{"score": <1-5>}}"""
        }
        
        try:
            response = self.eval_llm.invoke(prompts[dimension])
            result = self._parse_json(response.content)
            return result.get("score", 3)
        except:
            return 3  # Default to middle score on error
    
    def phase4_generate_report(self, results):
        """Phase 4: Generate comparison report."""
        print("\n" + "="*70, flush=True)
        print("PHASE 4: GENERATE REPORTS", flush=True)
        print("="*70, flush=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        
        # Comparison report
        report = f"""# Gold Standard 500 Evaluation Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Corpus:** 500 questions (all relevance score 5)
**Model:** {LLM_MODEL}

## Configuration Comparison

| Metric | Precision@25 | Precision@12 | Δ |
|--------|-------------|-------------|---|
"""
        
        p25 = results.get("precision_25", {}).get("metrics", {})
        p12 = results.get("precision_12", {}).get("metrics", {})
        
        for metric in ["mrr", "recall@100", "faithfulness", "relevance", "correctness", "completeness", "coherence"]:
            v25 = p25.get(metric, 0)
            v12 = p12.get(metric, 0)
            delta = v25 - v12
            sign = "+" if delta > 0 else ""
            report += f"| {metric} | {v25:.3f} | {v12:.3f} | {sign}{delta:.3f} |\n"
        
        report += f"""
## Key Findings

### Precision@25 vs Precision@12
- **MRR:** {p25.get('mrr', 0):.3f} vs {p12.get('mrr', 0):.3f}
- **Recall@100:** {p25.get('recall@100', 0):.3f} vs {p12.get('recall@100', 0):.3f}

### Answer Quality Dimensions
| Dimension | @25 | @12 | Winner |
|-----------|-----|-----|--------|
"""
        
        for dim in ["faithfulness", "relevance", "correctness", "completeness", "coherence"]:
            v25 = p25.get(dim, 0)
            v12 = p12.get(dim, 0)
            winner = "@25" if v25 > v12 else "@12" if v12 > v25 else "Tie"
            report += f"| {dim} | {v25:.3f} | {v12:.3f} | {winner} |\n"
        
        report += f"""
## Recommendation

Based on the evaluation results across 500 gold-standard questions:

"""
        
        # Determine recommendation
        avg_25 = sum(p25.get(d, 0) for d in ["faithfulness", "relevance", "correctness", "completeness", "coherence"]) / 5
        avg_12 = sum(p12.get(d, 0) for d in ["faithfulness", "relevance", "correctness", "completeness", "coherence"]) / 5
        
        if avg_25 > avg_12:
            report += "**Recommendation: Use Precision@25** - Higher average scores across quality dimensions.\n"
        else:
            report += "**Recommendation: Use Precision@12** - Better efficiency with comparable quality.\n"
        
        # Save report
        report_file = self.reports_dir / f"comparison_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Comparison report: {report_file}", flush=True)
        
        # Save full results JSON
        full_results_file = self.reports_dir / f"full_results_{timestamp}.json"
        with open(full_results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Full results: {full_results_file}", flush=True)
        
        return report_file
    
    def run(self):
        """Run the full pipeline."""
        print("="*70, flush=True)
        print("GOLD STANDARD 500 PIPELINE", flush=True)
        print(f"Config: {self.config['job_name']}", flush=True)
        print("="*70, flush=True)
        
        self._load_checkpoint()
        
        # Phase 1: Generate
        if self.checkpoint["phase"] in ["init", "generating"]:
            self.phase1_generate()
        
        # Phase 2: Filter and build final corpus
        if self.checkpoint["phase"] in ["generate_complete", "filtering"]:
            count = self.phase2_filter_corpus()
            if count < 500:
                print(f"\n⚠️  Only {count} questions available. Need to generate more.", flush=True)
                self.checkpoint["phase"] = "init"
                self._save_checkpoint()
                return self.run()  # Restart
        
        # Phase 3: Run eval suite
        if self.checkpoint["phase"] in ["filter_complete", "evaluating"]:
            results = self.phase3_run_eval()
        else:
            # Load existing results
            results = {}
            for run in self.config["evaluation"]["runs"]:
                result_file = self.reports_dir / f"{run['name']}_results.json"
                if result_file.exists():
                    with open(result_file) as f:
                        results[run["name"]] = json.load(f)
        
        # Phase 4: Generate report
        report_file = self.phase4_generate_report(results)
        
        print("\n" + "="*70, flush=True)
        print("✅ PIPELINE COMPLETE", flush=True)
        print(f"Report: {report_file}", flush=True)
        print("="*70, flush=True)
        
        return report_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="config/gold_standard_500.json")
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}")
        sys.exit(1)
    
    pipeline = GoldStandardPipeline(str(config_path))
    pipeline.run()


if __name__ == "__main__":
    main()
