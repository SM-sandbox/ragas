#!/usr/bin/env python3
"""
Generate Q&A corpus directly from Knowledge Graph using LLM.
Simple approach that bypasses Ragas transform issues.
"""
import json
import random
from pathlib import Path

from langchain_google_vertexai import ChatVertexAI
from ragas.testset.graph import KnowledgeGraph

from config import config


def load_knowledge_graph() -> KnowledgeGraph:
    """Load the knowledge graph from file"""
    kg_path = Path(__file__).parent / config.OUTPUT_DIR / "knowledge_graph.json"
    print(f"Loading KG from: {kg_path}")
    return KnowledgeGraph.load(kg_path)


def extract_documents_from_kg(kg: KnowledgeGraph) -> list[dict]:
    """Extract document content from knowledge graph nodes"""
    documents = []
    for node in kg.nodes:
        if node.type.value == "document":
            content = node.properties.get("page_content", "")
            metadata = node.properties.get("document_metadata", {})
            if content and len(content) > 100:
                documents.append({
                    "content": content,
                    "source": metadata.get("source_document", "unknown"),
                    "chunk_id": metadata.get("chunk_id", ""),
                })
    return documents


def generate_qa_from_document(llm, doc: dict) -> list[dict]:
    """Generate Q&A pairs from a single document using LLM"""
    prompt = f"""Based on the following document content, generate 2-3 high-quality question-answer pairs.
The questions should be specific, factual, and answerable from the content.
The answers should be comprehensive and directly supported by the content.

Document Source: {doc['source']}

Content:
{doc['content'][:3000]}

Generate Q&A pairs in this exact JSON format (return ONLY the JSON array, no other text):
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]
"""
    
    try:
        response = llm.invoke(prompt)
        # Parse JSON from response
        response_text = response.content.strip()
        # Handle markdown code blocks
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        qa_pairs = json.loads(response_text)
        
        # Add metadata
        for qa in qa_pairs:
            qa["source_document"] = doc["source"]
            qa["context"] = doc["content"][:1500]
        
        return qa_pairs
    except Exception as e:
        print(f"  Warning: Failed to generate Q&A for {doc['source']}: {e}")
        return []


def main():
    print("="*60)
    print("GENERATING Q&A CORPUS FROM KNOWLEDGE GRAPH")
    print("="*60)
    
    # Load KG
    print("\n[1/4] Loading Knowledge Graph...")
    kg = load_knowledge_graph()
    print(f"✓ Loaded KG with {len(kg.nodes)} nodes")
    
    # Extract documents
    print("\n[2/4] Extracting documents from KG...")
    documents = extract_documents_from_kg(kg)
    print(f"✓ Extracted {len(documents)} documents")
    
    # Sample documents for Q&A generation
    sample_size = min(30, len(documents))
    sampled_docs = random.sample(documents, sample_size)
    print(f"✓ Sampled {sample_size} documents for Q&A generation")
    
    # Initialize LLM
    print("\n[3/4] Initializing LLM...")
    llm = ChatVertexAI(
        model_name="gemini-2.0-flash",
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LLM_LOCATION,
        temperature=0.7,
    )
    print("✓ LLM initialized")
    
    # Generate Q&A pairs
    print("\n[4/4] Generating Q&A pairs...")
    all_qa_pairs = []
    
    for i, doc in enumerate(sampled_docs):
        print(f"  Processing {i+1}/{sample_size}: {doc['source'][:50]}...")
        qa_pairs = generate_qa_from_document(llm, doc)
        all_qa_pairs.extend(qa_pairs)
        print(f"    Generated {len(qa_pairs)} Q&A pairs")
    
    # Add IDs
    for i, qa in enumerate(all_qa_pairs):
        qa["id"] = i + 1
    
    # Save results
    output_dir = Path(__file__).parent / config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "qa_corpus_from_kg.json"
    with open(output_path, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"\n✓ Saved {len(all_qa_pairs)} Q&A pairs to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Q&A CORPUS SUMMARY")
    print("="*60)
    print(f"Total Q&A pairs: {len(all_qa_pairs)}")
    print(f"Source documents: {len(set(qa['source_document'] for qa in all_qa_pairs))}")
    
    # Show samples
    if all_qa_pairs:
        print("\n--- Sample Q&A Pairs ---")
        for sample in all_qa_pairs[:3]:
            print(f"\nQ: {sample['question']}")
            print(f"A: {sample['answer'][:200]}...")
            print(f"Source: {sample['source_document']}")


if __name__ == "__main__":
    main()
