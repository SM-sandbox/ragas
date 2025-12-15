#!/usr/bin/env python3
"""
Question Relevance Evaluator

Evaluates each Q&A question for domain relevance - is this a question a real 
SCADA/solar technician would actually ask, or is it trivial/irrelevant noise?

Relevance Scale (1-5):
- 5: Critical - Core technical knowledge a field tech MUST know
- 4: Relevant - Useful domain knowledge for the job  
- 3: Marginal - Somewhat useful but not essential
- 2: Low Value - Trivial or overly specific to document
- 1: Irrelevant - Not useful for domain work (addresses, watermarks, etc.)

Usage:
    python question_relevance_evaluator.py [--workers 15] [--max-questions N]
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from langchain_google_vertexai import ChatVertexAI

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_config import (
    GCP_PROJECT, GCP_LLM_LOCATION, LLM_MODEL,
    DEFAULT_WORKERS, DEFAULT_RETRIES, CHECKPOINT_INTERVAL,
    PROGRESS_BAR_CONFIG
)

# Paths
CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_200.json"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments"


RELEVANCE_PROMPT = """You are evaluating questions for a SCADA/solar/electrical equipment knowledge base.

The domain is: Industrial SCADA systems, solar inverters, energy monitoring equipment, electrical switchboards, and related technical documentation.

For each question, rate its DOMAIN RELEVANCE on a 1-5 scale:

5 - CRITICAL: Core technical knowledge a field technician MUST know
   Examples: Equipment specs, safety limits, configuration procedures, troubleshooting steps
   
4 - RELEVANT: Useful domain knowledge for the job
   Examples: Feature descriptions, compatibility info, operational procedures
   
3 - MARGINAL: Somewhat useful but not essential
   Examples: General product info, file formats, minor details
   
2 - LOW VALUE: Trivial or overly document-specific
   Examples: Document revision numbers, page references, formatting details
   
1 - IRRELEVANT: Not useful for domain work
   Examples: Company addresses, who downloaded a PDF, watermark text, legal boilerplate

Question to evaluate:
{question}

Ground truth answer (for context):
{answer}

Source document: {source_document}

Respond with JSON only:
{{
    "relevance_score": <1-5>,
    "relevance_label": "<critical|relevant|marginal|low_value|irrelevant>",
    "reasoning": "<brief explanation>",
    "flags": [<list of issues like "document_metadata", "company_info", "watermark", "legal_boilerplate", "non_technical">]
}}
"""


class QuestionRelevanceEvaluator:
    """Evaluates Q&A questions for domain relevance."""
    
    def __init__(self):
        self.llm = ChatVertexAI(
            model_name=LLM_MODEL,
            project=GCP_PROJECT,
            location=GCP_LLM_LOCATION,
            temperature=0.0,
            max_tokens=500,
        )
        self.qa_corpus = self._load_corpus()
        self.results = []
        self.buckets = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        self.checkpoint_path = None
        self.save_interval = CHECKPOINT_INTERVAL
        
    def _load_corpus(self) -> List[Dict]:
        """Load the Q&A corpus."""
        with open(CORPUS_PATH) as f:
            return json.load(f)
    
    def _load_checkpoint(self, checkpoint_path: Path) -> set:
        """Load completed question IDs from checkpoint."""
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                data = json.load(f)
                self.results = data.get("results", [])
                # Rebuild buckets from results
                for r in self.results:
                    if r.get("success") and r.get("relevance_score"):
                        score = r["relevance_score"]
                        self.buckets[score] = self.buckets.get(score, 0) + 1
                return {r["question_id"] for r in self.results}
        return set()
    
    def _save_checkpoint(self, output_path: Path):
        """Save current progress."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_evaluated": len(self.results),
            "buckets": self.buckets,
            "results": self.results
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def evaluate_question(self, qa_item: Dict, retries: int = DEFAULT_RETRIES) -> Dict:
        """Evaluate a single question for relevance."""
        question_id = qa_item.get("id", hash(qa_item["question"]))
        start_time = time.time()
        
        prompt = RELEVANCE_PROMPT.format(
            question=qa_item["question"],
            answer=qa_item["answer"],
            source_document=qa_item.get("source_document", "Unknown")
        )
        
        for attempt in range(retries):
            try:
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                
                # Parse JSON from response
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                result = json.loads(content)
                elapsed = time.time() - start_time
                
                return {
                    "question_id": question_id,
                    "question": qa_item["question"],
                    "source_document": qa_item.get("source_document", "Unknown"),
                    "relevance_score": result["relevance_score"],
                    "relevance_label": result["relevance_label"],
                    "reasoning": result["reasoning"],
                    "flags": result.get("flags", []),
                    "success": True,
                    "time_seconds": round(elapsed, 2),
                    "answer_chars": len(qa_item["answer"])
                }
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {
                        "question_id": question_id,
                        "question": qa_item["question"],
                        "source_document": qa_item.get("source_document", "Unknown"),
                        "relevance_score": None,
                        "relevance_label": None,
                        "reasoning": None,
                        "flags": [],
                        "success": False,
                        "error": str(e),
                        "time_seconds": round(time.time() - start_time, 2),
                        "answer_chars": len(qa_item["answer"])
                    }
    
    def run_evaluation(
        self,
        max_questions: Optional[int] = None,
        parallel_workers: int = DEFAULT_WORKERS
    ) -> Dict:
        """Run the full relevance evaluation."""
        
        # Setup output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = OUTPUT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_question_relevance"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        data_dir = experiment_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        output_file = data_dir / f"relevance_evaluation_{timestamp}.json"
        checkpoint_file = data_dir / "checkpoint_relevance.json"
        self.checkpoint_path = checkpoint_file
        
        # Load checkpoint if exists
        completed_ids = self._load_checkpoint(checkpoint_file)
        
        # Filter corpus
        corpus = self.qa_corpus[:max_questions] if max_questions else self.qa_corpus
        pending = [q for q in corpus if q.get("id", hash(q["question"])) not in completed_ids]
        
        print("=" * 70)
        print("QUESTION RELEVANCE EVALUATION")
        print("=" * 70)
        print(f"LLM Model: {LLM_MODEL}")
        print(f"Workers: {parallel_workers}")
        print(f"Total Questions: {len(corpus)}")
        print(f"Already Completed: {len(completed_ids)}")
        print(f"Pending: {len(pending)}")
        print("=" * 70)
        
        if not pending:
            print("All questions already evaluated!")
        else:
            # Progress bar with live metrics
            pbar = tqdm(
                total=len(pending),
                desc="Evaluating",
                **PROGRESS_BAR_CONFIG
            )
            
            def update_metrics():
                total = sum(self.buckets.values())
                if total > 0:
                    avg = sum(k * v for k, v in self.buckets.items()) / total
                    bucket_str = "|".join(f"{k}:{v}" for k, v in sorted(self.buckets.items(), reverse=True))
                    pbar.set_postfix_str(f"avg={avg:.2f} [{bucket_str}]")
            
            # Parallel execution
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(self.evaluate_question, q): q for q in pending}
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    self.results.append(result)
                    
                    if result["success"] and result["relevance_score"]:
                        score = result["relevance_score"]
                        self.buckets[score] = self.buckets.get(score, 0) + 1
                    
                    pbar.update(1)
                    update_metrics()
                    
                    # Checkpoint
                    if (i + 1) % self.save_interval == 0:
                        self._save_checkpoint(checkpoint_file)
            
            pbar.close()
        
        # Calculate final metrics
        successful = [r for r in self.results if r["success"]]
        total_time = sum(r["time_seconds"] for r in successful)
        
        metrics = {
            "total_questions": len(self.results),
            "successful": len(successful),
            "failed": len(self.results) - len(successful),
            "distribution": self.buckets,
            "avg_relevance": sum(r["relevance_score"] for r in successful) / len(successful) if successful else 0,
            "high_quality_count": self.buckets.get(5, 0) + self.buckets.get(4, 0),
            "low_quality_count": self.buckets.get(2, 0) + self.buckets.get(1, 0),
            "total_time_seconds": round(total_time, 2),
            "avg_time_per_question": round(total_time / len(successful), 2) if successful else 0,
        }
        
        # Questions to potentially remove
        low_quality = [r for r in successful if r["relevance_score"] <= 2]
        
        # Save final output
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "llm_model": LLM_MODEL,
                "workers": parallel_workers,
                "retries": DEFAULT_RETRIES,
            },
            "metrics": metrics,
            "low_quality_questions": low_quality,
            "results": self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("QUESTION RELEVANCE RESULTS")
        print("=" * 70)
        print(f"Total Evaluated: {metrics['total_questions']}")
        print(f"Average Relevance: {metrics['avg_relevance']:.2f}")
        print(f"\nDistribution:")
        for score in [5, 4, 3, 2, 1]:
            count = self.buckets.get(score, 0)
            pct = count / metrics['total_questions'] * 100 if metrics['total_questions'] > 0 else 0
            label = {5: "Critical", 4: "Relevant", 3: "Marginal", 2: "Low Value", 1: "Irrelevant"}[score]
            bar = "█" * int(pct / 2)
            print(f"  {score} ({label:10}): {count:3} ({pct:5.1f}%) {bar}")
        print(f"\nHigh Quality (4-5): {metrics['high_quality_count']} questions")
        print(f"Low Quality (1-2):  {metrics['low_quality_count']} questions (candidates for removal)")
        print(f"\nTotal Time: {metrics['total_time_seconds']:.1f}s")
        print(f"Avg Time/Question: {metrics['avg_time_per_question']:.2f}s")
        print("=" * 70)
        
        # Show low quality questions
        if low_quality:
            print(f"\n⚠️  LOW QUALITY QUESTIONS ({len(low_quality)}):")
            print("-" * 70)
            for q in low_quality[:10]:  # Show first 10
                print(f"  [{q['relevance_score']}] {q['question'][:80]}...")
                print(f"      Reason: {q['reasoning']}")
                if q['flags']:
                    print(f"      Flags: {', '.join(q['flags'])}")
            if len(low_quality) > 10:
                print(f"  ... and {len(low_quality) - 10} more")
        
        return metrics


def main():
    """Run the question relevance evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Question Relevance Evaluator")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS, 
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--max-questions", "-n", type=int, default=None, 
                        help="Limit questions to evaluate")
    args = parser.parse_args()
    
    evaluator = QuestionRelevanceEvaluator()
    evaluator.run_evaluation(
        max_questions=args.max_questions,
        parallel_workers=args.workers
    )


if __name__ == "__main__":
    main()
