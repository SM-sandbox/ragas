#!/usr/bin/env python3
"""
Fill gaps with better prompts and retry logic.
- Score 5 required for easy/medium
- Score 4 OK for hard questions
"""

import json
import random
import re
import sys
import time
from pathlib import Path

from langchain_google_vertexai import ChatVertexAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_config import GCP_PROJECT, GCP_LLM_LOCATION

CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_v2.json"
METADATA_DIR = Path(__file__).parent.parent / "doc_metadata" / "json"
INPUTS_DIR = Path(__file__).parent.parent / "doc_metadata" / "inputs_first10"

MODEL = "gemini-2.5-flash"
MAX_RETRIES = 5

GAPS = [
    ("single_hop", "medium", 5, 5),
    ("single_hop", "hard", 28, 4),
    ("multi_hop", "easy", 10, 5),
    ("multi_hop", "hard", 4, 4),
]

DIFFICULTY_DESC = {
    "easy": "Direct fact lookup",
    "medium": "Requires context understanding",
    "hard": "Requires synthesis or comparison"
}

def load_resources():
    metadata = {}
    for f in METADATA_DIR.glob("*.json"):
        with open(f) as fp:
            data = json.load(fp)
            metadata[data.get("doc_id", f.stem)] = data
    contents = {}
    for f in INPUTS_DIR.glob("*.txt"):
        with open(f) as fp:
            contents[f.stem] = fp.read()
    return metadata, contents

def extract_json(text):
    """Extract JSON from text, handling markdown code blocks."""
    text = text.strip()
    # Try to find JSON in code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{"):
                text = part
                break
    
    # Find JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON found")
    
    # Find matching closing brace
    depth = 0
    for i, c in enumerate(text[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i+1])
    
    raise ValueError("Incomplete JSON")

def generate_question(llm, hop_type, difficulty, metadata, contents):
    available = [d for d in metadata if d in contents]
    
    if hop_type == "single_hop":
        doc_id = random.choice(available)
        meta = metadata[doc_id]
        prompt = f"""You are generating a technical question for field technicians.

DOCUMENT: {meta.get("doc_title", doc_id)}
CONTENT:
{contents[doc_id][:5000]}

Generate ONE {difficulty.upper()} question about this document.
- {DIFFICULTY_DESC[difficulty]}
- Must be about: equipment specs, safety, troubleshooting, configuration
- Must NOT be about: revision numbers, authors, dates, form fields

OUTPUT FORMAT - Return ONLY this JSON, nothing else:
{{"question": "your technical question", "ground_truth_answer": "the factual answer", "reasoning": "why this is useful for field techs"}}"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = llm.invoke(prompt)
                result = extract_json(response.content)
                result["question_type"] = "single_hop"
                result["difficulty"] = difficulty
                result["source_doc_id"] = doc_id
                result["source_filename"] = meta.get("source_filename", f"{doc_id}.pdf")
                return result
            except Exception:
                time.sleep(1)
        return None
    else:
        doc1, doc2 = random.sample(available, 2)
        meta1, meta2 = metadata[doc1], metadata[doc2]
        prompt = f"""You are generating a technical question requiring info from BOTH documents.

DOCUMENT 1: {meta1.get("doc_title", doc1)}
{contents[doc1][:2500]}

DOCUMENT 2: {meta2.get("doc_title", doc2)}
{contents[doc2][:2500]}

Generate ONE {difficulty.upper()} multi-hop question.
- {DIFFICULTY_DESC[difficulty]}
- MUST require information from BOTH documents
- Must be about: equipment specs, safety, troubleshooting, configuration

OUTPUT FORMAT - Return ONLY this JSON, nothing else:
{{"question": "your technical question", "ground_truth_answer": "answer using both docs", "reasoning": "why both docs are needed"}}"""
        
        for attempt in range(MAX_RETRIES):
            try:
                response = llm.invoke(prompt)
                result = extract_json(response.content)
                result["question_type"] = "multi_hop"
                result["difficulty"] = difficulty
                result["source_doc_id"] = doc1
                result["source_filename"] = meta1.get("source_filename", f"{doc1}.pdf")
                result["secondary_doc_id"] = doc2
                return result
            except Exception:
                time.sleep(1)
        return None

def evaluate_question(llm, question):
    prompt = f"""Rate this question for field technician relevance (1-5):

5 = CRITICAL: Must-know technical knowledge (specs, safety, troubleshooting)
4 = RELEVANT: Useful domain knowledge (features, procedures)
3 = MARGINAL: Somewhat useful
2 = LOW VALUE: Trivial details
1 = IRRELEVANT: Not useful

QUESTION: {question.get("question", "")}
ANSWER: {question.get("ground_truth_answer", "")}

OUTPUT FORMAT - Return ONLY this JSON:
{{"score": 5, "rationale": "brief reason"}}"""
    
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(prompt)
            result = extract_json(response.content)
            return result.get("score", 0), result.get("rationale", "")
        except Exception:
            time.sleep(1)
    return 0, "Failed to evaluate"

def main():
    print("Loading resources...", flush=True)
    metadata, contents = load_resources()
    print(f"  {len(metadata)} docs", flush=True)
    
    print("Loading corpus...", flush=True)
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    
    print("Initializing LLM...", flush=True)
    llm = ChatVertexAI(
        model_name=MODEL,
        project=GCP_PROJECT,
        location=GCP_LLM_LOCATION,
        temperature=0.7,
        max_tokens=1000,
    )
    
    existing_ids = [int(q.get("question_id", "q_0000")[2:]) for q in corpus["questions"]]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    
    new_questions = []
    
    for hop_type, difficulty, count_needed, min_score in GAPS:
        print(f"\n{hop_type}/{difficulty}: need {count_needed} (min score {min_score})", flush=True)
        generated = 0
        attempts = 0
        max_attempts = count_needed * 5
        
        while generated < count_needed and attempts < max_attempts:
            attempts += 1
            
            q = generate_question(llm, hop_type, difficulty, metadata, contents)
            if q is None:
                print("!", end="", flush=True)
                continue
            
            score, rationale = evaluate_question(llm, q)
            q["domain_relevance_score"] = score
            q["domain_relevance_rationale"] = rationale
            
            if score >= min_score:
                q["question_id"] = f"q_{next_id:04d}"
                new_questions.append(q)
                corpus["questions"].append(q)
                generated += 1
                next_id += 1
                print(f"✓{score}", end="", flush=True)
                
                if generated % 5 == 0:
                    with open(CORPUS_PATH, 'w') as f:
                        json.dump(corpus, f, indent=2)
                    print(" [saved]", end="", flush=True)
            else:
                print(f"x{score}", end="", flush=True)
        
        print(f" → {generated}/{count_needed}", flush=True)
    
    with open(CORPUS_PATH, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"\n✓ Added {len(new_questions)} questions. Total: {len(corpus['questions'])}", flush=True)

if __name__ == "__main__":
    main()
