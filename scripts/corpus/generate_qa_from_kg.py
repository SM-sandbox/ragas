#!/usr/bin/env python3
"""
Generate Q&A corpus from the Knowledge Graph using Ragas TestsetGenerator.
"""
import json
from pathlib import Path

from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph

from config import config


def load_knowledge_graph() -> KnowledgeGraph:
    """Load the knowledge graph from file"""
    kg_path = Path(__file__).parent / config.OUTPUT_DIR / "knowledge_graph.json"
    print(f"Loading KG from: {kg_path}")
    return KnowledgeGraph.load(kg_path)


def extract_documents_from_kg(kg: KnowledgeGraph):
    """Extract LangChain documents from knowledge graph nodes"""
    from langchain_core.documents import Document as LCDocument
    
    documents = []
    for node in kg.nodes:
        if node.type.value == "document":
            content = node.properties.get("page_content", "")
            metadata = node.properties.get("document_metadata", {})
            if content:
                documents.append(LCDocument(page_content=content, metadata=metadata))
    
    return documents


def generate_qa_corpus(kg: KnowledgeGraph, num_questions: int = 50):
    """Generate Q&A pairs from the knowledge graph"""
    print("="*60)
    print("GENERATING Q&A CORPUS FROM KNOWLEDGE GRAPH")
    print("="*60)
    
    # Extract documents from KG
    print("\n[1/4] Extracting documents from KG...")
    documents = extract_documents_from_kg(kg)
    print(f"✓ Extracted {len(documents)} documents from KG")
    
    # Initialize LLM and embeddings
    print("\n[2/4] Initializing LLM and embeddings...")
    
    llm = ChatVertexAI(
        model_name="gemini-2.0-flash",
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LLM_LOCATION,
        temperature=0.7,
    )
    
    embeddings = VertexAIEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        project=config.GCP_PROJECT_ID,
        location=config.GCP_LOCATION,
    )
    
    # Wrap for Ragas
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)
    
    print("✓ LLM and embeddings initialized")
    
    # Create testset generator
    print("\n[3/4] Creating TestsetGenerator...")
    generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_embeddings)
    print("✓ Generator created")
    
    # Generate testset from documents
    print(f"\n[4/4] Generating {num_questions} Q&A pairs...")
    print("This may take a few minutes...")
    
    testset = generator.generate_with_langchain_docs(
        documents=documents,
        testset_size=num_questions,
        with_debugging_logs=True,
    )
    
    print(f"✓ Generated {len(testset)} Q&A pairs")
    
    return testset


def save_qa_corpus(testset, output_path: Path):
    """Save the Q&A corpus to JSON"""
    qa_pairs = []
    
    for i, sample in enumerate(testset.samples):
        qa_pair = {
            "id": i + 1,
            "question": sample.eval_sample.user_input,
            "ground_truth": sample.eval_sample.reference,
            "contexts": sample.eval_sample.reference_contexts if hasattr(sample.eval_sample, 'reference_contexts') else [],
            "synthesizer": sample.synthesizer_name if hasattr(sample, 'synthesizer_name') else "unknown",
        }
        qa_pairs.append(qa_pair)
    
    with open(output_path, 'w') as f:
        json.dump(qa_pairs, f, indent=2)
    
    print(f"\n✓ Q&A corpus saved to: {output_path}")
    return qa_pairs


def main():
    # Load knowledge graph
    print("\nLoading Knowledge Graph...")
    kg = load_knowledge_graph()
    print(f"✓ Loaded KG with {len(kg.nodes)} nodes and {len(kg.relationships)} relationships")
    
    # Generate Q&A corpus
    testset = generate_qa_corpus(kg, num_questions=50)
    
    # Save results
    output_dir = Path(__file__).parent / config.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    qa_path = output_dir / "qa_corpus_from_kg.json"
    qa_pairs = save_qa_corpus(testset, qa_path)
    
    # Print summary
    print("\n" + "="*60)
    print("Q&A CORPUS SUMMARY")
    print("="*60)
    print(f"Total Q&A pairs: {len(qa_pairs)}")
    
    # Show sample
    if qa_pairs:
        print("\n--- Sample Q&A Pair ---")
        sample = qa_pairs[0]
        print(f"Q: {sample['question'][:200]}...")
        print(f"A: {sample['ground_truth'][:200]}...")


if __name__ == "__main__":
    main()
