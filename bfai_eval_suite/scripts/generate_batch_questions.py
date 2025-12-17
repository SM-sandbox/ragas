#!/usr/bin/env python3
"""
Generate batch of questions to fill gaps in corpus distribution.
Only keeps questions that score 4-5 on relevance.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from langchain_google_vertexai import ChatVertexAI

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_config import GCP_PROJECT, GCP_LLM_LOCATION, LLM_MODEL

CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_v2.json"
METADATA_DIR = Path(__file__).parent.parent / "doc_metadata" / "json"
INPUTS_DIR = Path(__file__).parent.parent / "doc_metadata" / "inputs_first10"

# Generation prompts - emphasize technical relevance
SINGLE_HOP_PROMPT = """Generate a {difficulty} single-hop question for a SCADA/solar/electrical equipment RAG system.

Document: {doc_title}
Content:
{content}

Requirements:
- Question must be answerable from this document
- Must be HIGHLY RELEVANT to field technicians (equipment specs, safety, troubleshooting, configuration)
- NOT about document metadata (revision numbers, authors, form fields, dates)
- {difficulty_desc}

Return JSON only:
{{"question": "...", "ground_truth_answer": "...", "reasoning": "..."}}
"""

MULTI_HOP_PROMPT = """Generate a {difficulty} multi-hop question requiring BOTH documents.

Doc 1: {doc1_title}
{content1}

Doc 2: {doc2_title}
{content2}

Requirements:
- Must require info from BOTH documents
- HIGHLY RELEVANT to field technicians (specs, safety, troubleshooting)
- NOT about document metadata
- {difficulty_desc}

Return JSON only:
{{"question": "...", "ground_truth_answer": "...", "reasoning": "..."}}
"""

EVAL_PROMPT = """Rate this question's relevance for field technicians (1-5):
5=Critical (specs, safety, troubleshooting), 4=Relevant (procedures, features), 3=Marginal, 2=Low value, 1=Irrelevant

Question: {question}
Answer: {answer}

Return JSON: {{"score": <1-5>, "rationale": "..."}}
"""

DIFFICULTY_DESC = {
    "easy": "Direct fact lookup (e.g., 'What is the max voltage?')",
    "medium": "Requires context understanding (e.g., 'What safety step is needed before X?')",
    "hard": "Requires synthesis/comparison (e.g., 'How does X differ between modes?')"
}


def load_resources():
    """Load metadata and document contents."""
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
    """Parse JSON from LLM response."""
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
            return {"score": int(score_match.group(1)), "rationale": rationale_match.group(1) if rationale_match else ""}
        raise


def generate_and_evaluate(llm, eval_llm, hop_type, difficulty, metadata, contents, used_docs):
    """Generate one question and evaluate it. Returns (question_dict, score) or (None, 0)."""
    available = [d for d in metadata if d in contents and d not in used_docs]
    if not available:
        available = [d for d in metadata if d in contents]
    
    try:
        if hop_type == "single_hop":
            doc_id = random.choice(available)
            meta = metadata[doc_id]
            prompt = SINGLE_HOP_PROMPT.format(
                difficulty=difficulty,
                doc_title=meta.get("doc_title", doc_id),
                content=contents[doc_id][:6000],
                difficulty_desc=DIFFICULTY_DESC[difficulty]
            )
            response = llm.invoke(prompt)
            result = parse_json(response.content)
            result["question_type"] = "single_hop"
            result["difficulty"] = difficulty
            result["source_doc_id"] = doc_id
            result["source_filename"] = meta.get("source_filename", f"{doc_id}.pdf")
        else:
            doc1, doc2 = random.sample(available if len(available) >= 2 else list(metadata.keys()), 2)
            meta1, meta2 = metadata.get(doc1, {}), metadata.get(doc2, {})
            prompt = MULTI_HOP_PROMPT.format(
                difficulty=difficulty,
                doc1_title=meta1.get("doc_title", doc1),
                content1=contents.get(doc1, "")[:3000],
                doc2_title=meta2.get("doc_title", doc2),
                content2=contents.get(doc2, "")[:3000],
                difficulty_desc=DIFFICULTY_DESC[difficulty]
            )
            response = llm.invoke(prompt)
            result = parse_json(response.content)
            result["question_type"] = "multi_hop"
            result["difficulty"] = difficulty
            result["source_doc_id"] = doc1
            result["source_filename"] = meta1.get("source_filename", f"{doc1}.pdf")
            result["secondary_doc_id"] = doc2
        
        # Evaluate
        eval_prompt = EVAL_PROMPT.format(
            question=result.get("question", ""),
            answer=result.get("ground_truth_answer", "")
        )
        eval_response = eval_llm.invoke(eval_prompt)
        eval_result = parse_json(eval_response.content)
        score = eval_result.get("score", 0)
        
        result["domain_relevance_score"] = score
        result["domain_relevance_rationale"] = eval_result.get("rationale", "")
        
        return result, score
    except Exception as e:
        print(f"!", end="", flush=True)
        return None, 0


def main():
    print("Loading resources...", flush=True)
    metadata, contents = load_resources()
    
    print("Loading corpus...", flush=True)
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)
    
    questions = corpus["questions"]
    
    # Calculate what we need
    dist = defaultdict(lambda: defaultdict(int))
    for q in questions:
        dist[q.get("question_type")][q.get("difficulty")] += 1
    
    needs = []
    for hop in ["single_hop", "multi_hop"]:
        for diff in ["easy", "medium", "hard"]:
            target = 83 if diff != "hard" else 84
            current = dist[hop][diff]
            needed = max(0, target - current)
            if needed > 0:
                needs.append((hop, diff, needed))
    
    total_needed = sum(n[2] for n in needs)
    print(f"Need to generate {total_needed} questions", flush=True)
    
    if total_needed == 0:
        print("Corpus already complete!")
        return
    
    # Initialize LLMs
    print("Initializing LLMs...", flush=True)
    llm = ChatVertexAI(model_name=LLM_MODEL, project=GCP_PROJECT, location=GCP_LLM_LOCATION, temperature=0.7, max_tokens=1000)
    eval_llm = ChatVertexAI(model_name=LLM_MODEL, project=GCP_PROJECT, location=GCP_LLM_LOCATION, temperature=0.0, max_tokens=500)
    
    # Track used docs
    used_docs = set(q.get("source_doc_id") for q in questions)
    
    # Get next question ID
    next_id = max(int(q.get("question_id", "q_0000")[2:]) for q in questions) + 1
    
    new_questions = []
    
    for hop_type, difficulty, count_needed in needs:
        print(f"\n{hop_type}/{difficulty}: need {count_needed}", flush=True)
        generated = 0
        attempts = 0
        max_attempts = count_needed * 8  # Allow many retries
        
        while generated < count_needed and attempts < max_attempts:
            attempts += 1
            q, score = generate_and_evaluate(llm, eval_llm, hop_type, difficulty, metadata, contents, used_docs)
            
            if q and score >= 4:
                q["question_id"] = f"q_{next_id:04d}"
                new_questions.append(q)
                generated += 1
                next_id += 1
                print(f"✓", end="", flush=True)
            elif q:
                print(f"x{score}", end="", flush=True)
            # else: already printed !
        
        print(f" → {generated}/{count_needed}", flush=True)
    
    # Add new questions to corpus
    corpus["questions"].extend(new_questions)
    
    # Save
    with open(CORPUS_PATH, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"\n✓ Added {len(new_questions)} questions. Total: {len(corpus['questions'])}", flush=True)
    
    # Show final distribution
    print("\nFinal distribution:")
    dist = defaultdict(lambda: defaultdict(int))
    for q in corpus["questions"]:
        dist[q.get("question_type")][q.get("difficulty")] += 1
    for hop in ["single_hop", "multi_hop"]:
        for diff in ["easy", "medium", "hard"]:
            print(f"  {hop}/{diff}: {dist[hop][diff]}")


if __name__ == "__main__":
    main()
