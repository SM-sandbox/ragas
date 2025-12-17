#!/usr/bin/env python3
"""
Fill gaps - using google.genai SDK with response_schema for reliable JSON.
- Score 5 required for easy/medium
- Score 4 OK for hard questions
"""

import json
import random
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_config import GCP_PROJECT, GCP_LLM_LOCATION

CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_v2.json"
METADATA_DIR = Path(__file__).parent.parent / "doc_metadata" / "json"
INPUTS_DIR = Path(__file__).parent.parent / "doc_metadata" / "inputs_first10"

MODEL = "gemini-2.5-flash"
MAX_RETRIES = 5

GAPS = [
    ("single_hop", "medium", 5, 5),   # need 5, min_score 5
    ("single_hop", "hard", 28, 4),    # need 28, min_score 4 (hard OK with Relevant)
    ("multi_hop", "easy", 10, 5),     # need 10, min_score 5
    ("multi_hop", "hard", 4, 4),      # need 4, min_score 4 (hard OK with Relevant)
]

DIFFICULTY_DESC = {
    "easy": "Direct fact lookup (e.g., 'What is the max voltage?')",
    "medium": "Requires context understanding (e.g., 'What safety step before maintenance?')",
    "hard": "Requires synthesis or comparison (e.g., 'How does X differ between modes?')"
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

def parse_json(content):
    content = content.strip()
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content.strip())

def generate_question(model, gen_config, hop_type, difficulty, metadata, contents):
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
- Difficulty: {DIFFICULTY_DESC[difficulty]}

Return JSON with keys: question, ground_truth_answer, reasoning"""
        
        response = model.generate_content(prompt, generation_config=gen_config)
        result = json.loads(response.text)
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
- Difficulty: {DIFFICULTY_DESC[difficulty]}

Return JSON with keys: question, ground_truth_answer, reasoning"""
        
        response = model.generate_content(prompt, generation_config=gen_config)
        result = json.loads(response.text)
        result["question_type"] = "multi_hop"
        result["difficulty"] = difficulty
        result["source_doc_id"] = doc1
        result["source_filename"] = meta1.get("source_filename", f"{doc1}.pdf")
        result["secondary_doc_id"] = doc2
        return result

def evaluate_question(model, eval_config, question):
    prompt = f"""Rate this question's DOMAIN RELEVANCE for field technicians (1-5):

5 = CRITICAL: Core knowledge field tech MUST know (specs, safety, troubleshooting)
4 = RELEVANT: Useful domain knowledge (features, procedures)
3 = MARGINAL: Somewhat useful
2 = LOW VALUE: Trivial or document-specific
1 = IRRELEVANT: Not useful

Question: {question.get("question", "")}
Answer: {question.get("ground_truth_answer", "")}

Return JSON with keys: score, rationale"""
    
    response = model.generate_content(prompt, generation_config=eval_config)
    result = json.loads(response.text)
    return result.get("score", 0), result.get("rationale", "")

def main():
    print("Loading resources...", flush=True)
    metadata, contents = load_resources()
    print(f"  {len(metadata)} docs, {len(contents)} content files", flush=True)
    
    print("Loading corpus...", flush=True)
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    
    print("Initializing Vertex AI with JSON mode...", flush=True)
    vertexai.init(project=GCP_PROJECT, location=GCP_LLM_LOCATION)
    
    model = GenerativeModel(MODEL)
    gen_config = GenerationConfig(
        temperature=0.7,
        max_output_tokens=1000,
        response_mime_type="application/json",
    )
    eval_config = GenerationConfig(
        temperature=0.0,
        max_output_tokens=300,
        response_mime_type="application/json",
    )
    
    # Get next ID
    existing_ids = [int(q.get("question_id", "q_0000")[2:]) for q in corpus["questions"]]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    
    new_questions = []
    
    for hop_type, difficulty, count_needed, min_score in GAPS:
        print(f"\n{hop_type}/{difficulty}: need {count_needed} (min score {min_score})", flush=True)
        generated = 0
        attempts = 0
        max_attempts = count_needed * 6
        
        while generated < count_needed and attempts < max_attempts:
            attempts += 1
            
            try:
                q = generate_question(model, gen_config, hop_type, difficulty, metadata, contents)
                score, rationale = evaluate_question(model, eval_config, q)
                
                q["domain_relevance_score"] = score
                q["domain_relevance_rationale"] = rationale
                
                if score >= min_score:
                    q["question_id"] = f"q_{next_id:04d}"
                    new_questions.append(q)
                    corpus["questions"].append(q)
                    generated += 1
                    next_id += 1
                    print(f"✓{score}", end="", flush=True)
                    
                    # Save every 5
                    if generated % 5 == 0:
                        with open(CORPUS_PATH, 'w') as f:
                            json.dump(corpus, f, indent=2)
                        print(" [saved]", end="", flush=True)
                else:
                    print(f"x{score}", end="", flush=True)
                    
            except json.JSONDecodeError as e:
                print("J", end="", flush=True)  # JSON error
                time.sleep(1)
            except Exception as e:
                print("!", end="", flush=True)
                time.sleep(2)
        
        print(f" → {generated}/{count_needed}", flush=True)
    
    # Final save
    with open(CORPUS_PATH, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"\n✓ Added {len(new_questions)} questions. Total: {len(corpus['questions'])}", flush=True)

if __name__ == "__main__":
    main()
