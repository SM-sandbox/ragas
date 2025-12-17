#!/usr/bin/env python3
"""
Add Domain Relevance Scores to Q&A Corpus

Evaluates each question for domain relevance and appends the score + rationale
directly to each question in the corpus JSON.

Relevance Scale (1-5):
- 5: Critical - Core technical knowledge a field tech MUST know
- 4: Relevant - Useful domain knowledge for the job  
- 3: Marginal - Somewhat useful but not essential
- 2: Low Value - Trivial or overly specific to document
- 1: Irrelevant - Not useful for domain work (addresses, watermarks, etc.)

Usage:
    python add_relevance_to_corpus.py [--workers 15] [--input corpus/qa_corpus_v2.json]
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

# Default paths
DEFAULT_CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_v2.json"
CHECKPOINT_DIR = Path(__file__).parent.parent / "corpus"


RELEVANCE_PROMPT = """You are evaluating questions for a SCADA/solar/electrical equipment knowledge base.

The domain is: Industrial SCADA systems, solar inverters, energy monitoring equipment, electrical switchboards, battery energy storage systems (BESS), and related technical documentation.

For each question, rate its DOMAIN RELEVANCE on a 1-5 scale. Think like a field technician or engineer - would they actually need to know this?

5 - CRITICAL: Core technical knowledge a field technician MUST know
   Examples: Equipment specs, safety limits, configuration procedures, troubleshooting steps, wiring diagrams
   
4 - RELEVANT: Useful domain knowledge for the job
   Examples: Feature descriptions, compatibility info, operational procedures, maintenance schedules
   
3 - MARGINAL: Somewhat useful but not essential
   Examples: General product info, file formats, minor operational details
   
2 - LOW VALUE: Trivial or overly document-specific
   Examples: Document revision numbers, page references, formatting details, form field names
   
1 - IRRELEVANT: Not useful for domain work
   Examples: Company addresses, who downloaded a PDF, watermark text, legal boilerplate, document subtitles

Question to evaluate:
{question}

Ground truth answer (for context):
{answer}

Source document: {source_document}

Respond with JSON only:
{{
    "score": <1-5>,
    "rationale": "<1-2 sentence explanation of why this score, from perspective of a field tech>"
}}
"""


class CorpusRelevanceAnnotator:
    """Adds relevance scores to Q&A corpus questions."""
    
    def __init__(self, corpus_path: Path):
        self.llm = ChatVertexAI(
            model_name=LLM_MODEL,
            project=GCP_PROJECT,
            location=GCP_LLM_LOCATION,
            temperature=0.0,
            max_tokens=1000,
        )
        self.corpus_path = corpus_path
        self.corpus_data = self._load_corpus()
        self.checkpoint_path = CHECKPOINT_DIR / "relevance_checkpoint.json"
        self.buckets = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
        
    def _load_corpus(self) -> Dict:
        """Load the Q&A corpus."""
        with open(self.corpus_path) as f:
            return json.load(f)
    
    def _load_checkpoint(self) -> Dict[str, Dict]:
        """Load checkpoint with completed evaluations."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                data = json.load(f)
                # Rebuild buckets - ensure score is int
                for qid, result in data.items():
                    score = result.get("score")
                    if score is not None:
                        score = int(score)  # Ensure int
                        self.buckets[score] = self.buckets.get(score, 0) + 1
                return data
        return {}
    
    def _save_checkpoint(self, completed: Dict[str, Dict]):
        """Save checkpoint."""
        with open(self.checkpoint_path, 'w') as f:
            json.dump(completed, f, indent=2)
    
    def evaluate_question(self, question: Dict, retries: int = DEFAULT_RETRIES) -> Dict:
        """Evaluate a single question for relevance."""
        question_id = question.get("question_id", "unknown")
        
        prompt = RELEVANCE_PROMPT.format(
            question=question["question"],
            answer=question.get("ground_truth_answer", ""),
            source_document=question.get("source_filename", question.get("source_doc_id", "Unknown"))
        )
        
        for attempt in range(retries):
            try:
                response = self.llm.invoke(prompt)
                content = response.content.strip()
                
                # Parse JSON from response - try multiple approaches
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                # Try direct parse first
                try:
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract score and rationale with regex
                    import re
                    score_match = re.search(r'"score"\s*:\s*(\d)', content)
                    # Try multiple patterns for rationale
                    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', content)
                    if not rationale_match:
                        # Try to capture everything after "rationale": " until the end or closing brace
                        rationale_match = re.search(r'"rationale"\s*:\s*"(.+?)(?:"\s*}|$)', content, re.DOTALL)
                    
                    if score_match:
                        score = int(score_match.group(1))
                        if rationale_match:
                            rationale = rationale_match.group(1).replace('\n', ' ').strip()
                            # Clean up any trailing incomplete parts
                            if rationale.endswith('\\'):
                                rationale = rationale[:-1]
                        else:
                            rationale = "Technical domain question requiring field knowledge."
                        result = {"score": score, "rationale": rationale}
                    else:
                        raise ValueError(f"Could not parse score from: {content[:200]}")
                
                return {
                    "question_id": question_id,
                    "score": result["score"],
                    "rationale": result["rationale"],
                    "success": True
                }
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "question_id": question_id,
                        "score": None,
                        "rationale": f"Evaluation failed: {str(e)}",
                        "success": False
                    }
    
    def run(self, parallel_workers: int = DEFAULT_WORKERS) -> None:
        """Run relevance evaluation and update corpus."""
        
        questions = self.corpus_data.get("questions", [])
        
        # Load checkpoint
        completed = self._load_checkpoint()
        
        # Find pending questions
        pending = []
        pending_indices = []
        for i, q in enumerate(questions):
            qid = q.get("question_id", f"q_{i}")
            if qid not in completed:
                pending.append(q)
                pending_indices.append(i)
        
        print("=" * 70)
        print("DOMAIN RELEVANCE ANNOTATION")
        print("=" * 70)
        print(f"LLM Model: {LLM_MODEL}")
        print(f"Workers: {parallel_workers}")
        print(f"Total Questions: {len(questions)}")
        print(f"Already Completed: {len(completed)}")
        print(f"Pending: {len(pending)}")
        print("=" * 70)
        
        if pending:
            pbar = tqdm(total=len(pending), desc="Evaluating", **PROGRESS_BAR_CONFIG)
            
            def update_metrics():
                total = sum(self.buckets.values())
                if total > 0:
                    avg = sum(k * v for k, v in self.buckets.items()) / total
                    bucket_str = "|".join(f"{k}:{v}" for k, v in sorted(self.buckets.items(), reverse=True))
                    pbar.set_postfix_str(f"avg={avg:.2f} [{bucket_str}]")
            
            # Parallel execution
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {executor.submit(self.evaluate_question, q): (i, q) for i, q in zip(pending_indices, pending)}
                
                for count, future in enumerate(as_completed(futures)):
                    idx, q = futures[future]
                    result = future.result()
                    qid = q.get("question_id", f"q_{idx}")
                    
                    completed[qid] = {
                        "score": result["score"],
                        "rationale": result["rationale"]
                    }
                    
                    if result["success"] and result["score"]:
                        self.buckets[result["score"]] = self.buckets.get(result["score"], 0) + 1
                    
                    pbar.update(1)
                    update_metrics()
                    
                    # Checkpoint every N
                    if (count + 1) % CHECKPOINT_INTERVAL == 0:
                        self._save_checkpoint(completed)
            
            pbar.close()
            self._save_checkpoint(completed)
        
        # Now update the corpus with relevance scores
        print("\nUpdating corpus with relevance scores...")
        
        for i, q in enumerate(questions):
            qid = q.get("question_id", f"q_{i}")
            if qid in completed:
                q["domain_relevance_score"] = completed[qid]["score"]
                q["domain_relevance_rationale"] = completed[qid]["rationale"]
        
        # Save updated corpus
        output_path = self.corpus_path  # Overwrite original
        with open(output_path, 'w') as f:
            json.dump(self.corpus_data, f, indent=2)
        
        print(f"✓ Updated corpus saved to: {output_path}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("RELEVANCE DISTRIBUTION")
        print("=" * 70)
        total = sum(self.buckets.values())
        for score in [5, 4, 3, 2, 1]:
            count = self.buckets.get(score, 0)
            pct = count / total * 100 if total > 0 else 0
            label = {5: "Critical", 4: "Relevant", 3: "Marginal", 2: "Low Value", 1: "Irrelevant"}[score]
            bar = "█" * int(pct / 2)
            print(f"  {score} ({label:10}): {count:3} ({pct:5.1f}%) {bar}")
        
        avg = sum(k * v for k, v in self.buckets.items()) / total if total > 0 else 0
        print(f"\nAverage Relevance Score: {avg:.2f}")
        print(f"High Quality (4-5): {self.buckets.get(5, 0) + self.buckets.get(4, 0)} questions")
        print(f"Low Quality (1-2):  {self.buckets.get(2, 0) + self.buckets.get(1, 0)} questions")
        print("=" * 70)
        
        # Show some low quality examples
        low_quality = [(q["question"], q.get("domain_relevance_rationale", "")) 
                       for q in questions 
                       if q.get("domain_relevance_score") is not None and q.get("domain_relevance_score") <= 2]
        
        if low_quality:
            print(f"\n⚠️  LOW QUALITY EXAMPLES (score 1-2):")
            for q, rationale in low_quality[:5]:
                print(f"  Q: {q[:70]}...")
                print(f"     → {rationale}")
                print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Add relevance scores to Q&A corpus")
    parser.add_argument("--input", "-i", type=str, default=str(DEFAULT_CORPUS_PATH),
                        help="Input corpus JSON file")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()
    
    corpus_path = Path(args.input)
    if not corpus_path.exists():
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)
    
    annotator = CorpusRelevanceAnnotator(corpus_path)
    annotator.run(parallel_workers=args.workers)


if __name__ == "__main__":
    main()
