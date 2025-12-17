#!/usr/bin/env python3
"""
Build a Knowledge Graph from the BrightFox document corpus using Ragas.
Simple one-layer deep KG with document nodes and basic relationships.
"""
import json
from pathlib import Path
from langchain_core.documents import Document as LCDocument
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.synthesizers.generate import TestsetGenerator
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from config import config


def load_chunks() -> list:
    """Load chunks from the data file"""
    chunks_file = Path(__file__).parent / config.DATA_DIR / "all_chunks.json"
    with open(chunks_file, 'r') as f:
        return json.load(f)


def build_simple_kg():
    """Build a simple one-layer knowledge graph from document chunks"""
    print("="*60)
    print("BUILDING KNOWLEDGE GRAPH")
    print("="*60)
    
    # Load chunks
    print("\n[1/4] Loading chunks...")
    chunks = load_chunks()
    print(f"✓ Loaded {len(chunks)} chunks")
    
    # Sample a subset for faster KG building (use first 100 chunks from different docs)
    print("\n[2/4] Sampling documents for KG...")
    docs_seen = set()
    sampled_chunks = []
    for chunk in chunks:
        doc = chunk.get('source_document', 'unknown')
        if doc not in docs_seen and len(sampled_chunks) < 50:
            docs_seen.add(doc)
            sampled_chunks.append(chunk)
    
    # Also add a few more chunks per document for relationships
    for chunk in chunks:
        doc = chunk.get('source_document', 'unknown')
        if doc in docs_seen and len([c for c in sampled_chunks if c.get('source_document') == doc]) < 3:
            sampled_chunks.append(chunk)
            if len(sampled_chunks) >= 100:
                break
    
    print(f"✓ Sampled {len(sampled_chunks)} chunks from {len(docs_seen)} documents")
    
    # Convert to LangChain documents
    print("\n[3/4] Converting to LangChain documents...")
    lc_documents = []
    for chunk in sampled_chunks:
        text = chunk.get('text', chunk.get('content', ''))
        if text and len(text) > 100:
            metadata = {
                'source_document': chunk.get('source_document', 'unknown'),
                'chunk_id': chunk.get('id', chunk.get('chunk_id', '')),
            }
            lc_documents.append(LCDocument(page_content=text, metadata=metadata))
    
    print(f"✓ Created {len(lc_documents)} LangChain documents")
    
    # Initialize LLM and embeddings for Ragas
    print("\n[4/4] Building Knowledge Graph with Ragas...")
    
    # Use Vertex AI
    llm = ChatVertexAI(
        model_name="gemini-2.5-flash",
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LLM_LOCATION,
        temperature=0.1,
    )
    
    embeddings = VertexAIEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
    )
    
    # Wrap for Ragas
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    # Create nodes from documents
    nodes = []
    for doc in lc_documents:
        node = Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
            },
        )
        nodes.append(node)
    
    # Create the knowledge graph
    kg = KnowledgeGraph(nodes=nodes)
    print(f"✓ Created KG with {len(kg.nodes)} nodes")
    
    # Apply simple transforms (summary extraction, embeddings)
    print("\nApplying transforms to build relationships...")
    
    transforms = default_transforms(
        documents=lc_documents,
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
    )
    
    run_config = RunConfig(max_workers=4, timeout=120)
    
    try:
        apply_transforms(kg, transforms, run_config=run_config)
        print(f"✓ Transforms applied. KG now has {len(kg.nodes)} nodes and {len(kg.relationships)} relationships")
    except Exception as e:
        print(f"⚠ Transform error (continuing with basic KG): {e}")
    
    # Save the knowledge graph
    output_dir = Path(__file__).parent / config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    kg_path = output_dir / "knowledge_graph.json"
    kg.save(kg_path)
    print(f"\n✓ Knowledge Graph saved to: {kg_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH SUMMARY")
    print("="*60)
    print(f"Total Nodes: {len(kg.nodes)}")
    print(f"Total Relationships: {len(kg.relationships)}")
    
    # Count node types
    node_types = {}
    for node in kg.nodes:
        nt = str(node.type)
        node_types[nt] = node_types.get(nt, 0) + 1
    print(f"Node Types: {node_types}")
    
    # Count relationship types
    if kg.relationships:
        rel_types = {}
        for rel in kg.relationships:
            rt = rel.type
            rel_types[rt] = rel_types.get(rt, 0) + 1
        print(f"Relationship Types: {rel_types}")
    
    return kg


if __name__ == "__main__":
    kg = build_simple_kg()
