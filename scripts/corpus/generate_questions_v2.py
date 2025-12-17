#!/usr/bin/env python3
"""
Question Generation Pipeline v2

Generates 500 evaluation questions leveraging document metadata:
- 250 single-hop, 250 multi-hop
- ~1/3 easy, ~1/3 medium, ~1/3 hard
- Uses doc metadata for targeted, high-quality questions

Model: Gemini 2.5 Flash with HIGH thinking
"""

import json
import random
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

from google import genai
from google.genai import types

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Paths
    "metadata_dir": "doc_metadata/json",
    "chunks_file": "data/all_chunks.json",
    "output_dir": "corpus",
    "output_file": "qa_corpus_v2.json",
    
    # GCP / Vertex AI
    "project": "bf-rag-sandbox-scott",
    "location": "us-central1",
    
    # Model Configuration
    "model": "gemini-2.5-flash",
    "temperature": 0.7,  # Slightly higher for creative question generation
    "max_output_tokens": 8192,
    "thinking_budget": 16384,  # HIGH thinking
    
    # Question targets
    "total_questions": 500,
    "single_hop_count": 250,
    "multi_hop_count": 250,
    
    # Difficulty distribution (per hop type)
    "difficulty_distribution": {
        "easy": 0.33,
        "medium": 0.34,
        "hard": 0.33
    },
    
    # Processing
    "max_retries": 5,
    "base_delay": 2.0,
    "questions_per_batch": 5,  # Generate multiple questions per LLM call for efficiency
}

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger("question_generator_v2")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
    
    return logger

# =============================================================================
# PROMPTS
# =============================================================================

SINGLE_HOP_PROMPT = """You are an expert at creating evaluation questions for a RAG (Retrieval Augmented Generation) system.
This is a SCADA/Solar/Electrical equipment technical corpus.

You have access to document metadata that describes what each document contains.
Generate {num_questions} SINGLE-HOP questions at {difficulty} difficulty level.

DIFFICULTY GUIDELINES:
- EASY: Direct factual recall. Answer is explicitly stated. Example: "What is the rated voltage of X?"
- MEDIUM: Requires understanding context or finding specific details. Example: "What PPE is required when performing X procedure?"
- HARD: Requires technical interpretation, calculations, or deep domain knowledge. Example: "Calculate the minimum conductor ampacity for X given Y conditions."

SINGLE-HOP means the question can be answered from ONE document/chunk.

DOCUMENT METADATA:
{metadata}

DOCUMENT TEXT (first 10 pages):
{text}

Generate exactly {num_questions} questions. Each question must:
1. Be answerable from this specific document
2. Match the {difficulty} difficulty level
3. Have a clear, factual ground truth answer
4. Be specific and technical (use model numbers, standards, specific values when available)

Respond with a JSON array:
[
  {{
    "question": "Your question here",
    "ground_truth_answer": "The correct answer",
    "difficulty": "{difficulty}",
    "question_type": "single_hop",
    "reasoning": "Why this matches the difficulty level"
  }},
  ...
]"""

MULTI_HOP_PROMPT = """You are an expert at creating evaluation questions for a RAG (Retrieval Augmented Generation) system.
This is a SCADA/Solar/Electrical equipment technical corpus.

Generate {num_questions} MULTI-HOP questions at {difficulty} difficulty level.
These questions MUST require information from MULTIPLE documents to answer completely.

DIFFICULTY GUIDELINES:
- EASY: Compare two related items across docs. Example: "Which inverter has higher efficiency, the CPS 125kW or the SMA Sunny Central?"
- MEDIUM: Synthesize information across docs for a procedure or requirement. Example: "What arc flash PPE is required per OSHA when working on a 480V Yotta BESS system?"
- HARD: Complex multi-step reasoning across 2-3 docs. Example: "Design the protection coordination for a 4.2MVA solar site using Shoals combiners and EPEC switchgear per NEC 2023."

DOCUMENT 1 METADATA:
{metadata1}

DOCUMENT 1 TEXT:
{text1}

DOCUMENT 2 METADATA:
{metadata2}

DOCUMENT 2 TEXT:
{text2}

Generate exactly {num_questions} questions. Each question must:
1. REQUIRE information from BOTH documents to answer
2. Match the {difficulty} difficulty level
3. Have a clear ground truth answer that synthesizes both sources
4. Be specific and technical

Respond with a JSON array:
[
  {{
    "question": "Your multi-hop question here",
    "ground_truth_answer": "Answer combining both documents",
    "difficulty": "{difficulty}",
    "question_type": "multi_hop",
    "reasoning": "Why this requires both docs and matches difficulty"
  }},
  ...
]"""

# =============================================================================
# UTILITIES
# =============================================================================

def retry_with_backoff(func, max_retries: int, base_delay: float, logger: logging.Logger):
    """Execute function with retry and exponential backoff."""
    import random as rand
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            error_str = str(e).lower()
            
            if "429" in str(e) or "resource exhausted" in error_str or "quota" in error_str:
                delay = base_delay * (2 ** attempt) + rand.uniform(0, 1)
                logger.warning(f"  Rate limited, retry {attempt+1}/{max_retries} in {delay:.1f}s...")
                time.sleep(delay)
            elif attempt < max_retries - 1:
                delay = base_delay + rand.uniform(0, 0.5)
                logger.warning(f"  Error: {e}, retry {attempt+1}/{max_retries} in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise
    
    raise last_exception


# =============================================================================
# QUESTION GENERATOR
# =============================================================================

class QuestionGeneratorV2:
    """Generate evaluation questions using document metadata."""
    
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize Vertex AI client
        self.client = genai.Client(
            vertexai=True,
            project=config["project"],
            location=config["location"]
        )
        
        self.model = config["model"]
        self.logger.info(f"Initialized QuestionGeneratorV2 with {self.model}")
    
    def load_metadata(self) -> List[Dict[str, Any]]:
        """Load all document metadata."""
        metadata_dir = Path(self.config["metadata_dir"])
        metadata_files = list(metadata_dir.glob("*.json"))
        
        metadata_list = []
        for f in metadata_files:
            with open(f, 'r') as fp:
                metadata_list.append(json.load(fp))
        
        self.logger.info(f"Loaded metadata for {len(metadata_list)} documents")
        return metadata_list
    
    def load_extracted_text(self, doc_id: str) -> str:
        """Load extracted text for a document."""
        text_file = Path("doc_metadata/inputs_first10") / f"{doc_id}.txt"
        if text_file.exists():
            with open(text_file, 'r') as f:
                return f.read()
        return ""
    
    def generate_questions_batch(
        self,
        prompt: str,
        num_questions: int
    ) -> List[Dict[str, Any]]:
        """Generate a batch of questions using LLM."""
        
        gen_config = types.GenerateContentConfig(
            temperature=self.config["temperature"],
            max_output_tokens=self.config["max_output_tokens"],
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.config["thinking_budget"]
            ),
            response_mime_type="application/json",
        )
        
        def do_generate():
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=gen_config
            )
            return response
        
        response = retry_with_backoff(
            do_generate,
            self.config["max_retries"],
            self.config["base_delay"],
            self.logger
        )
        
        # Parse response
        response_text = response.text
        if response_text is None:
            raise ValueError("Model returned empty response")
        
        response_text = response_text.strip()
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            response_text = "\n".join(lines)
        
        questions = json.loads(response_text)
        
        # Ensure it's a list
        if isinstance(questions, dict):
            questions = [questions]
        
        return questions[:num_questions]
    
    def generate_single_hop_questions(
        self,
        metadata_list: List[Dict],
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate single-hop questions across all documents."""
        
        questions = []
        difficulties = ["easy", "medium", "hard"]
        
        # Calculate questions per difficulty
        per_difficulty = {
            "easy": int(count * self.config["difficulty_distribution"]["easy"]),
            "medium": int(count * self.config["difficulty_distribution"]["medium"]),
            "hard": count - int(count * 0.33) - int(count * 0.34)  # Remainder goes to hard
        }
        
        self.logger.info(f"\nGenerating {count} single-hop questions:")
        self.logger.info(f"  Easy: {per_difficulty['easy']}, Medium: {per_difficulty['medium']}, Hard: {per_difficulty['hard']}")
        
        # Filter to high-confidence docs for better questions
        good_docs = [m for m in metadata_list if m.get("extraction_notes", {}).get("confidence") == "high"]
        if len(good_docs) < 10:
            good_docs = metadata_list  # Fall back to all if not enough high-confidence
        
        for difficulty in difficulties:
            target = per_difficulty[difficulty]
            generated = 0
            
            self.logger.info(f"\n  Generating {target} {difficulty} single-hop questions...")
            
            # Shuffle docs for variety
            random.shuffle(good_docs)
            doc_idx = 0
            
            while generated < target and doc_idx < len(good_docs) * 3:
                doc = good_docs[doc_idx % len(good_docs)]
                doc_idx += 1
                
                # Load text
                text = self.load_extracted_text(doc["doc_id"])
                if len(text) < 500:
                    continue
                
                # How many questions to generate this batch
                batch_size = min(3, target - generated)
                
                prompt = SINGLE_HOP_PROMPT.format(
                    num_questions=batch_size,
                    difficulty=difficulty,
                    metadata=json.dumps(doc, indent=2)[:3000],
                    text=text[:8000]
                )
                
                try:
                    batch = self.generate_questions_batch(prompt, batch_size)
                    
                    for q in batch:
                        q["source_doc_id"] = doc["doc_id"]
                        q["source_filename"] = doc.get("source", {}).get("filename", "unknown")
                        questions.append(q)
                        generated += 1
                        
                    self.logger.info(f"    ✓ {generated}/{target} {difficulty} questions")
                    
                except Exception as e:
                    self.logger.warning(f"    ✗ Failed for {doc['doc_id']}: {e}")
                    continue
        
        return questions
    
    def generate_multi_hop_questions(
        self,
        metadata_list: List[Dict],
        count: int
    ) -> List[Dict[str, Any]]:
        """Generate multi-hop questions requiring multiple documents."""
        
        questions = []
        difficulties = ["easy", "medium", "hard"]
        
        per_difficulty = {
            "easy": int(count * self.config["difficulty_distribution"]["easy"]),
            "medium": int(count * self.config["difficulty_distribution"]["medium"]),
            "hard": count - int(count * 0.33) - int(count * 0.34)
        }
        
        self.logger.info(f"\nGenerating {count} multi-hop questions:")
        self.logger.info(f"  Easy: {per_difficulty['easy']}, Medium: {per_difficulty['medium']}, Hard: {per_difficulty['hard']}")
        
        # Filter to high-confidence docs
        good_docs = [m for m in metadata_list if m.get("extraction_notes", {}).get("confidence") == "high"]
        if len(good_docs) < 10:
            good_docs = metadata_list
        
        # Create document pairs based on related topics
        def find_related_docs(doc: Dict, all_docs: List[Dict]) -> List[Dict]:
            """Find documents related to this one by topic/equipment overlap."""
            doc_keywords = set(doc.get("topics", {}).get("keywords", []))
            doc_equipment = set(doc.get("equipment", {}).get("equipment_family", []))
            
            scored = []
            for other in all_docs:
                if other["doc_id"] == doc["doc_id"]:
                    continue
                
                other_keywords = set(other.get("topics", {}).get("keywords", []))
                other_equipment = set(other.get("equipment", {}).get("equipment_family", []))
                
                # Score by overlap
                keyword_overlap = len(doc_keywords & other_keywords)
                equipment_overlap = len(doc_equipment & other_equipment)
                score = keyword_overlap + equipment_overlap * 2
                
                if score > 0:
                    scored.append((score, other))
            
            scored.sort(key=lambda x: -x[0])
            return [x[1] for x in scored[:5]]
        
        for difficulty in difficulties:
            target = per_difficulty[difficulty]
            generated = 0
            
            self.logger.info(f"\n  Generating {target} {difficulty} multi-hop questions...")
            
            random.shuffle(good_docs)
            doc_idx = 0
            
            while generated < target and doc_idx < len(good_docs) * 3:
                doc1 = good_docs[doc_idx % len(good_docs)]
                doc_idx += 1
                
                # Find related doc
                related = find_related_docs(doc1, good_docs)
                if not related:
                    # Fall back to random
                    candidates = [d for d in good_docs if d["doc_id"] != doc1["doc_id"]]
                    if not candidates:
                        continue
                    doc2 = random.choice(candidates)
                else:
                    doc2 = random.choice(related[:3]) if len(related) >= 3 else related[0]
                
                # Load texts
                text1 = self.load_extracted_text(doc1["doc_id"])
                text2 = self.load_extracted_text(doc2["doc_id"])
                
                if len(text1) < 500 or len(text2) < 500:
                    continue
                
                batch_size = min(2, target - generated)
                
                prompt = MULTI_HOP_PROMPT.format(
                    num_questions=batch_size,
                    difficulty=difficulty,
                    metadata1=json.dumps(doc1, indent=2)[:2000],
                    text1=text1[:5000],
                    metadata2=json.dumps(doc2, indent=2)[:2000],
                    text2=text2[:5000]
                )
                
                try:
                    batch = self.generate_questions_batch(prompt, batch_size)
                    
                    for q in batch:
                        q["source_doc_ids"] = [doc1["doc_id"], doc2["doc_id"]]
                        q["source_filenames"] = [
                            doc1.get("source", {}).get("filename", "unknown"),
                            doc2.get("source", {}).get("filename", "unknown")
                        ]
                        questions.append(q)
                        generated += 1
                    
                    self.logger.info(f"    ✓ {generated}/{target} {difficulty} questions")
                    
                except Exception as e:
                    self.logger.warning(f"    ✗ Failed for {doc1['doc_id']} + {doc2['doc_id']}: {e}")
                    continue
        
        return questions
    
    def generate_all(self) -> Dict[str, Any]:
        """Generate all questions."""
        
        metadata_list = self.load_metadata()
        
        # Generate single-hop
        single_hop = self.generate_single_hop_questions(
            metadata_list,
            self.config["single_hop_count"]
        )
        
        # Generate multi-hop
        multi_hop = self.generate_multi_hop_questions(
            metadata_list,
            self.config["multi_hop_count"]
        )
        
        # Combine and add IDs
        all_questions = []
        for i, q in enumerate(single_hop + multi_hop):
            q["question_id"] = f"q_{i+1:04d}"
            all_questions.append(q)
        
        # Summary stats
        stats = {
            "total": len(all_questions),
            "single_hop": len(single_hop),
            "multi_hop": len(multi_hop),
            "by_difficulty": {
                "easy": len([q for q in all_questions if q.get("difficulty") == "easy"]),
                "medium": len([q for q in all_questions if q.get("difficulty") == "medium"]),
                "hard": len([q for q in all_questions if q.get("difficulty") == "hard"]),
            },
            "generated_at": datetime.now().isoformat(),
            "model": self.config["model"]
        }
        
        output = {
            "metadata": stats,
            "questions": all_questions
        }
        
        # Save
        output_path = Path(self.config["output_dir"]) / self.config["output_file"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("QUESTION GENERATION COMPLETE")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Total Questions: {stats['total']}")
        self.logger.info(f"  Single-hop: {stats['single_hop']}")
        self.logger.info(f"  Multi-hop: {stats['multi_hop']}")
        self.logger.info(f"By Difficulty:")
        self.logger.info(f"  Easy: {stats['by_difficulty']['easy']}")
        self.logger.info(f"  Medium: {stats['by_difficulty']['medium']}")
        self.logger.info(f"  Hard: {stats['by_difficulty']['hard']}")
        self.logger.info(f"\nSaved to: {output_path}")
        
        return output


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("QUESTION GENERATION PIPELINE v2")
    logger.info("="*60)
    logger.info(f"Model: {CONFIG['model']}")
    logger.info(f"Target: {CONFIG['total_questions']} questions")
    logger.info(f"  Single-hop: {CONFIG['single_hop_count']}")
    logger.info(f"  Multi-hop: {CONFIG['multi_hop_count']}")
    logger.info(f"Difficulty: ~33% easy, ~34% medium, ~33% hard")
    logger.info("")
    
    generator = QuestionGeneratorV2(CONFIG, logger)
    result = generator.generate_all()
    
    return result


if __name__ == "__main__":
    main()
