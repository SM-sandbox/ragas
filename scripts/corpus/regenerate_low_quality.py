#!/usr/bin/env python3
"""
Regenerate Low-Quality Questions

This script:
1. Identifies questions with relevance scores 1-3
2. Removes them from the corpus
3. Regenerates replacements matching the same distribution (hop type + difficulty)
4. Evaluates the new questions
5. Repeats until all questions are 4-5

Usage:
    python regenerate_low_quality.py [--max-iterations 5]
"""

import json
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from langchain_google_vertexai import ChatVertexAI

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_config import (
    GCP_PROJECT, GCP_LLM_LOCATION, LLM_MODEL,
    DEFAULT_WORKERS, DEFAULT_RETRIES, PROGRESS_BAR_CONFIG
)

# Paths
CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_v2.json"
METADATA_DIR = Path(__file__).parent.parent / "doc_metadata" / "json"
INPUTS_DIR = Path(__file__).parent.parent / "doc_metadata" / "inputs_first10"


# Prompts
SINGLE_HOP_PROMPT = """You are generating evaluation questions for a SCADA/solar/electrical equipment RAG system.

Document: {doc_title}
Source file: {source_filename}

Document content:
{content}

Generate ONE {difficulty} single-hop question that:
1. Can be answered directly from this document
2. Is HIGHLY RELEVANT to a field technician's work (equipment specs, safety limits, configuration, troubleshooting)
3. Has a clear, factual answer
4. Is NOT about document metadata (revision numbers, authors, dates, form fields)

Difficulty: {difficulty}
- easy: Direct fact lookup (e.g., "What is the maximum voltage rating?")
- medium: Requires understanding context (e.g., "What safety precaution is required before maintenance?")
- hard: Requires synthesis or comparison (e.g., "How does the fault detection differ between modes?")

Respond with JSON only:
{{
    "question": "Your question here",
    "ground_truth_answer": "The correct answer",
    "reasoning": "Why this is a good technical question for field techs"
}}
"""

MULTI_HOP_PROMPT = """You are generating evaluation questions for a SCADA/solar/electrical equipment RAG system.

Document 1: {doc1_title}
Content 1:
{content1}

Document 2: {doc2_title}
Content 2:
{content2}

Generate ONE {difficulty} multi-hop question that:
1. REQUIRES information from BOTH documents to answer
2. Is HIGHLY RELEVANT to a field technician's work (equipment specs, safety, configuration, troubleshooting)
3. Tests ability to synthesize information across sources
4. Is NOT about document metadata

Difficulty: {difficulty}
- easy: Compare a simple fact between two documents
- medium: Synthesize related procedures or specs
- hard: Complex comparison requiring deep understanding of both

Respond with JSON only:
{{
    "question": "Your multi-hop question",
    "ground_truth_answer": "Answer combining both sources",
    "reasoning": "Why this requires both documents"
}}
"""

RELEVANCE_PROMPT = """You are evaluating questions for a SCADA/solar/electrical equipment knowledge base.

Rate this question's DOMAIN RELEVANCE on a 1-5 scale. Think like a field technician.

5 - CRITICAL: Core technical knowledge a field tech MUST know (specs, safety, troubleshooting)
4 - RELEVANT: Useful domain knowledge (features, procedures, compatibility)
3 - MARGINAL: Somewhat useful but not essential
2 - LOW VALUE: Trivial or document-specific (revision numbers, form fields)
1 - IRRELEVANT: Not useful (addresses, authors, legal text)

Question: {question}
Answer: {answer}
Source: {source}

Respond with JSON only:
{{
    "score": <1-5>,
    "rationale": "<1-2 sentence explanation>"
}}
"""


class QuestionRegenerator:
    """Regenerates low-quality questions until all are 4-5."""
    
    def __init__(self):
        self.llm = ChatVertexAI(
            model_name=LLM_MODEL,
            project=GCP_PROJECT,
            location=GCP_LLM_LOCATION,
            temperature=0.7,  # Some creativity for question generation
            max_tokens=1000,
        )
        self.eval_llm = ChatVertexAI(
            model_name=LLM_MODEL,
            project=GCP_PROJECT,
            location=GCP_LLM_LOCATION,
            temperature=0.0,  # Deterministic for evaluation
            max_tokens=500,
        )
        self.corpus_data = self._load_corpus()
        self.metadata = self._load_metadata()
        self.doc_contents = self._load_doc_contents()
        
    def _load_corpus(self) -> Dict:
        with open(CORPUS_PATH) as f:
            return json.load(f)
    
    def _load_metadata(self) -> Dict[str, Dict]:
        """Load all document metadata."""
        metadata = {}
        for f in METADATA_DIR.glob("*.json"):
            with open(f) as fp:
                data = json.load(fp)
                doc_id = data.get("doc_id", f.stem)
                metadata[doc_id] = data
        return metadata
    
    def _load_doc_contents(self) -> Dict[str, str]:
        """Load document text content."""
        contents = {}
        for f in INPUTS_DIR.glob("*.txt"):
            doc_id = f.stem
            with open(f) as fp:
                contents[doc_id] = fp.read()
        return contents
    
    def _get_low_quality_distribution(self) -> Dict[Tuple[str, str], List[Dict]]:
        """Get distribution of low-quality questions by (hop_type, difficulty)."""
        dist = defaultdict(list)
        for q in self.corpus_data["questions"]:
            score = q.get("domain_relevance_score")
            if score is not None and score <= 3:
                key = (q.get("question_type", "single_hop"), q.get("difficulty", "medium"))
                dist[key].append(q)
        return dist
    
    def _generate_single_hop(self, difficulty: str, exclude_docs: set) -> Dict:
        """Generate a single-hop question."""
        # Pick a random document
        available_docs = [d for d in self.metadata.keys() if d not in exclude_docs and d in self.doc_contents]
        if not available_docs:
            available_docs = list(self.metadata.keys())
        
        doc_id = random.choice(available_docs)
        meta = self.metadata.get(doc_id, {})
        content = self.doc_contents.get(doc_id, "")[:8000]  # Limit content
        
        prompt = SINGLE_HOP_PROMPT.format(
            doc_title=meta.get("doc_title", doc_id),
            source_filename=meta.get("source_filename", f"{doc_id}.pdf"),
            content=content,
            difficulty=difficulty
        )
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_json(response.content)
            result["question_type"] = "single_hop"
            result["difficulty"] = difficulty
            result["source_doc_id"] = doc_id
            result["source_filename"] = meta.get("source_filename", f"{doc_id}.pdf")
            return result
        except Exception as e:
            return None
    
    def _generate_multi_hop(self, difficulty: str, exclude_docs: set) -> Dict:
        """Generate a multi-hop question."""
        available_docs = [d for d in self.metadata.keys() if d not in exclude_docs and d in self.doc_contents]
        if len(available_docs) < 2:
            available_docs = list(self.metadata.keys())
        
        doc1_id, doc2_id = random.sample(available_docs, 2)
        meta1 = self.metadata.get(doc1_id, {})
        meta2 = self.metadata.get(doc2_id, {})
        content1 = self.doc_contents.get(doc1_id, "")[:4000]
        content2 = self.doc_contents.get(doc2_id, "")[:4000]
        
        prompt = MULTI_HOP_PROMPT.format(
            doc1_title=meta1.get("doc_title", doc1_id),
            content1=content1,
            doc2_title=meta2.get("doc_title", doc2_id),
            content2=content2,
            difficulty=difficulty
        )
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_json(response.content)
            result["question_type"] = "multi_hop"
            result["difficulty"] = difficulty
            result["source_doc_id"] = doc1_id
            result["source_filename"] = meta1.get("source_filename", f"{doc1_id}.pdf")
            result["secondary_doc_id"] = doc2_id
            return result
        except Exception as e:
            return None
    
    def _evaluate_question(self, question: Dict) -> Tuple[int, str]:
        """Evaluate a question's relevance. Returns (score, rationale)."""
        prompt = RELEVANCE_PROMPT.format(
            question=question.get("question", ""),
            answer=question.get("ground_truth_answer", ""),
            source=question.get("source_filename", "Unknown")
        )
        
        try:
            response = self.eval_llm.invoke(prompt)
            result = self._parse_json(response.content)
            return result.get("score", 0), result.get("rationale", "")
        except Exception as e:
            return 0, f"Evaluation failed: {e}"
    
    def _parse_json(self, content: str) -> Dict:
        """Parse JSON from LLM response."""
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Regex fallback
            score_match = re.search(r'"score"\s*:\s*(\d)', content)
            rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', content)
            if score_match:
                return {
                    "score": int(score_match.group(1)),
                    "rationale": rationale_match.group(1) if rationale_match else ""
                }
            raise
    
    def run(self, max_iterations: int = 10) -> None:
        """Main loop: remove low-quality, regenerate, evaluate, repeat."""
        
        for iteration in range(1, max_iterations + 1):
            # Get current low-quality distribution
            low_quality_dist = self._get_low_quality_distribution()
            total_low = sum(len(v) for v in low_quality_dist.values())
            
            if total_low == 0:
                print(f"\n✅ All questions are high quality (4-5)!")
                break
            
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'='*70}")
            print(f"Low quality questions to replace: {total_low}")
            
            for (hop_type, difficulty), questions in low_quality_dist.items():
                print(f"  {hop_type} / {difficulty}: {len(questions)}")
            
            # Remove low-quality questions
            high_quality = [q for q in self.corpus_data["questions"] 
                          if q.get("domain_relevance_score") is not None and q.get("domain_relevance_score") >= 4]
            
            # Track which docs we've used recently
            used_docs = set(q.get("source_doc_id") for q in high_quality)
            
            # Generate replacements
            new_questions = []
            next_id = max(int(q.get("question_id", "q_0000")[2:]) for q in self.corpus_data["questions"]) + 1
            
            print(f"\nGenerating {total_low} replacement questions...", flush=True)
            
            for (hop_type, difficulty), questions in low_quality_dist.items():
                count_needed = len(questions)
                generated = 0
                attempts = 0
                max_attempts = count_needed * 5  # Allow more failures
                
                print(f"\n  {hop_type}/{difficulty}: generating {count_needed}...", flush=True)
                
                while generated < count_needed and attempts < max_attempts:
                    attempts += 1
                    
                    # Generate question
                    if hop_type == "single_hop":
                        q = self._generate_single_hop(difficulty, used_docs)
                    else:
                        q = self._generate_multi_hop(difficulty, used_docs)
                    
                    if q is None:
                        print(".", end="", flush=True)
                        continue
                    
                    # Evaluate immediately
                    score, rationale = self._evaluate_question(q)
                    
                    if score >= 4:
                        q["question_id"] = f"q_{next_id:04d}"
                        q["domain_relevance_score"] = score
                        q["domain_relevance_rationale"] = rationale
                        new_questions.append(q)
                        generated += 1
                        next_id += 1
                        print(f"✓{score}", end="", flush=True)
                    else:
                        print(f"x{score}", end="", flush=True)
                
                print(f" → {generated}/{count_needed}", flush=True)
                
                if generated < count_needed:
                    print(f"  ⚠️  Only generated {generated}/{count_needed} for {hop_type}/{difficulty}")
            
            # Update corpus
            self.corpus_data["questions"] = high_quality + new_questions
            
            # Save
            with open(CORPUS_PATH, 'w') as f:
                json.dump(self.corpus_data, f, indent=2)
            
            print(f"\n✓ Corpus updated: {len(self.corpus_data['questions'])} questions")
            
            # Check if we're done
            remaining_low = sum(1 for q in self.corpus_data["questions"] 
                               if q.get("domain_relevance_score", 0) < 4)
            if remaining_low == 0:
                print(f"\n✅ All questions are now high quality (4-5)!")
                break
        
        # Final summary
        self._print_summary()
    
    def _print_summary(self):
        """Print final distribution summary."""
        print(f"\n{'='*70}")
        print("FINAL CORPUS SUMMARY")
        print(f"{'='*70}")
        
        questions = self.corpus_data["questions"]
        
        # Score distribution
        scores = defaultdict(int)
        for q in questions:
            scores[q.get("domain_relevance_score", 0)] += 1
        
        print("\nRelevance Score Distribution:")
        for s in [5, 4, 3, 2, 1]:
            count = scores.get(s, 0)
            pct = count / len(questions) * 100 if questions else 0
            print(f"  {s}: {count} ({pct:.1f}%)")
        
        # Type/difficulty distribution
        print("\nQuestion Type/Difficulty Distribution:")
        dist = defaultdict(lambda: defaultdict(int))
        for q in questions:
            dist[q.get("question_type", "unknown")][q.get("difficulty", "unknown")] += 1
        
        for hop in ["single_hop", "multi_hop"]:
            for diff in ["easy", "medium", "hard"]:
                print(f"  {hop} / {diff}: {dist[hop][diff]}")
        
        print(f"\nTotal questions: {len(questions)}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", "-i", type=int, default=10)
    args = parser.parse_args()
    
    regenerator = QuestionRegenerator()
    regenerator.run(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
