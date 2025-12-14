#!/usr/bin/env python3
"""
Generate 200+ Q&A pairs from Knowledge Graph with multi-hop support.
Uses KG relationships for multi-hop question generation.
"""
import json
import random
from pathlib import Path
from collections import defaultdict

from langchain_google_vertexai import ChatVertexAI
from ragas.testset.graph import KnowledgeGraph

from config import config


def load_knowledge_graph() -> KnowledgeGraph:
    """Load the knowledge graph from file"""
    kg_path = Path(__file__).parent / config.OUTPUT_DIR / "knowledge_graph.json"
    print(f"Loading KG from: {kg_path}")
    return KnowledgeGraph.load(kg_path)


def analyze_kg_structure(kg: KnowledgeGraph) -> dict:
    """Analyze KG for multi-hop capabilities"""
    analysis = {
        "total_nodes": len(kg.nodes),
        "total_relationships": len(kg.relationships),
        "node_types": defaultdict(int),
        "relationship_types": defaultdict(int),
        "connected_pairs": [],
    }
    
    # Count node types
    node_map = {}
    for node in kg.nodes:
        analysis["node_types"][node.type.value] += 1
        node_map[node.id] = node
    
    # Count relationship types and find connected pairs
    for rel in kg.relationships:
        analysis["relationship_types"][rel.type] += 1
        if rel.source in node_map and rel.target in node_map:
            analysis["connected_pairs"].append({
                "source": node_map[rel.source],
                "target": node_map[rel.target],
                "type": rel.type,
            })
    
    return analysis, node_map


def extract_documents_from_kg(kg: KnowledgeGraph) -> list[dict]:
    """Extract document content from knowledge graph nodes"""
    documents = []
    for node in kg.nodes:
        if node.type.value == "document":
            content = node.properties.get("page_content", "")
            metadata = node.properties.get("document_metadata", {})
            if content and len(content) > 100:
                documents.append({
                    "id": node.id,
                    "content": content,
                    "source": metadata.get("source_document", "unknown"),
                    "chunk_id": metadata.get("chunk_id", ""),
                })
    return documents


def generate_single_hop_qa(llm, doc: dict) -> list[dict]:
    """Generate single-hop Q&A pairs from a single document"""
    prompt = f"""Based on the following technical document about SCADA/Solar/Electrical equipment, generate 2 high-quality question-answer pairs.

Requirements:
- Questions should be specific and factual
- Answers should be directly supported by the content
- Focus on technical details, specifications, procedures, or safety information

Document Source: {doc['source']}

Content:
{doc['content'][:3500]}

Generate Q&A pairs in this exact JSON format (return ONLY the JSON array):
[
  {{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}},
  {{"question": "...", "answer": "...", "difficulty": "easy|medium|hard"}}
]
"""
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        qa_pairs = json.loads(response_text)
        
        for qa in qa_pairs:
            qa["source_document"] = doc["source"]
            qa["context"] = doc["content"][:1500]
            qa["question_type"] = "single_hop"
            qa["node_id"] = doc["id"]
        
        return qa_pairs
    except Exception as e:
        print(f"  Warning: Failed for {doc['source'][:30]}: {e}")
        return []


def generate_multi_hop_qa(llm, doc1: dict, doc2: dict, relationship_type: str) -> list[dict]:
    """Generate multi-hop Q&A that requires both documents"""
    prompt = f"""You are creating evaluation questions for a RAG system on SCADA/Solar/Electrical equipment.

Given TWO related documents (relationship: {relationship_type}), create 1-2 MULTI-HOP questions that:
1. REQUIRE information from BOTH documents to answer completely
2. Test ability to synthesize across sources
3. Are specific and technical

Document 1 ({doc1['source']}):
{doc1['content'][:2000]}

Document 2 ({doc2['source']}):
{doc2['content'][:2000]}

Generate multi-hop Q&A in this exact JSON format (return ONLY the JSON array):
[
  {{"question": "...", "answer": "...", "reasoning": "Why both docs needed"}}
]
"""
    
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
        
        qa_pairs = json.loads(response_text)
        
        for qa in qa_pairs:
            qa["source_documents"] = [doc1["source"], doc2["source"]]
            qa["contexts"] = [doc1["content"][:1000], doc2["content"][:1000]]
            qa["question_type"] = "multi_hop"
            qa["relationship_type"] = relationship_type
            qa["node_ids"] = [doc1["id"], doc2["id"]]
        
        return qa_pairs
    except Exception as e:
        print(f"  Warning: Multi-hop failed: {e}")
        return []


def main():
    print("="*60)
    print("GENERATING 200+ Q&A CORPUS WITH MULTI-HOP SUPPORT")
    print("="*60)
    
    # Load KG
    print("\n[1/5] Loading Knowledge Graph...")
    kg = load_knowledge_graph()
    
    # Analyze KG structure
    print("\n[2/5] Analyzing KG structure for multi-hop...")
    analysis, node_map = analyze_kg_structure(kg)
    
    print(f"  Total nodes: {analysis['total_nodes']}")
    print(f"  Total relationships: {analysis['total_relationships']}")
    print(f"  Node types: {dict(analysis['node_types'])}")
    print(f"  Relationship types: {dict(analysis['relationship_types'])}")
    print(f"  Connected pairs available for multi-hop: {len(analysis['connected_pairs'])}")
    
    # Extract documents
    print("\n[3/5] Extracting documents from KG...")
    documents = extract_documents_from_kg(kg)
    print(f"✓ Extracted {len(documents)} documents")
    
    # Create document lookup
    doc_by_id = {doc["id"]: doc for doc in documents}
    
    # Initialize LLM
    print("\n[4/5] Initializing LLM...")
    llm = ChatVertexAI(
        model_name="gemini-2.0-flash",
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LLM_LOCATION,
        temperature=0.7,
    )
    print("✓ LLM initialized")
    
    # Generate Q&A pairs
    print("\n[5/5] Generating Q&A pairs...")
    all_qa_pairs = []
    
    # --- SINGLE-HOP: Generate ~160 questions from 80 documents ---
    print("\n--- Generating Single-Hop Questions (target: 160) ---")
    sample_size = min(80, len(documents))
    sampled_docs = random.sample(documents, sample_size)
    
    for i, doc in enumerate(sampled_docs):
        print(f"  Single-hop {i+1}/{sample_size}: {doc['source'][:40]}...")
        qa_pairs = generate_single_hop_qa(llm, doc)
        all_qa_pairs.extend(qa_pairs)
    
    single_hop_count = len(all_qa_pairs)
    print(f"✓ Generated {single_hop_count} single-hop Q&A pairs")
    
    # --- MULTI-HOP: Generate ~80 questions from KG relationships ---
    print("\n--- Generating Multi-Hop Questions (target: 80) ---")
    
    # Use KG relationships for multi-hop
    multi_hop_pairs = []
    for pair in analysis["connected_pairs"]:
        source_id = pair["source"].id
        target_id = pair["target"].id
        if source_id in doc_by_id and target_id in doc_by_id:
            multi_hop_pairs.append({
                "doc1": doc_by_id[source_id],
                "doc2": doc_by_id[target_id],
                "rel_type": pair["type"],
            })
    
    # Also create cross-document pairs for more multi-hop questions
    if len(multi_hop_pairs) < 60:
        print(f"  Adding cross-document pairs (have {len(multi_hop_pairs)} from KG)...")
        for _ in range(60 - len(multi_hop_pairs)):
            if len(documents) >= 2:
                doc1, doc2 = random.sample(documents, 2)
                multi_hop_pairs.append({
                    "doc1": doc1,
                    "doc2": doc2,
                    "rel_type": "cross_document",
                })
    
    random.shuffle(multi_hop_pairs)
    multi_hop_target = min(60, len(multi_hop_pairs))
    
    for i, pair in enumerate(multi_hop_pairs[:multi_hop_target]):
        print(f"  Multi-hop {i+1}/{multi_hop_target}: {pair['rel_type']}...")
        qa_pairs = generate_multi_hop_qa(llm, pair["doc1"], pair["doc2"], pair["rel_type"])
        all_qa_pairs.extend(qa_pairs)
    
    multi_hop_count = len(all_qa_pairs) - single_hop_count
    print(f"✓ Generated {multi_hop_count} multi-hop Q&A pairs")
    
    # Add IDs
    for i, qa in enumerate(all_qa_pairs):
        qa["id"] = i + 1
    
    # Save results
    output_dir = Path(__file__).parent / config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert UUIDs to strings for JSON serialization
    def convert_uuids(obj):
        if isinstance(obj, dict):
            return {k: convert_uuids(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_uuids(item) for item in obj]
        elif hasattr(obj, 'hex'):  # UUID object
            return str(obj)
        return obj
    
    all_qa_pairs = convert_uuids(all_qa_pairs)
    
    output_path = output_dir / "qa_corpus_200.json"
    with open(output_path, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"\n✓ Saved {len(all_qa_pairs)} Q&A pairs to: {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("Q&A CORPUS SUMMARY")
    print("="*60)
    print(f"Total Q&A pairs: {len(all_qa_pairs)}")
    print(f"  Single-hop: {single_hop_count}")
    print(f"  Multi-hop: {multi_hop_count}")
    
    # Count by difficulty if available
    difficulties = defaultdict(int)
    for qa in all_qa_pairs:
        difficulties[qa.get("difficulty", "unknown")] += 1
    if difficulties:
        print(f"  By difficulty: {dict(difficulties)}")
    
    # Show samples
    print("\n--- Sample Single-Hop Q&A ---")
    single_samples = [qa for qa in all_qa_pairs if qa.get("question_type") == "single_hop"][:2]
    for s in single_samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answer'][:150]}...")
        print()
    
    print("--- Sample Multi-Hop Q&A ---")
    multi_samples = [qa for qa in all_qa_pairs if qa.get("question_type") == "multi_hop"][:2]
    for s in multi_samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answer'][:150]}...")
        if "reasoning" in s:
            print(f"Reasoning: {s['reasoning'][:100]}...")
        print()


if __name__ == "__main__":
    main()
