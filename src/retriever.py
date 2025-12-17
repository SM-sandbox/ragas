"""RAG Retriever that combines Vector Search with chunk lookup"""
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from config import config
from vector_search import VectorSearchClient


class RAGRetriever:
    """Retriever that uses Vertex AI Vector Search and returns chunk content"""
    
    def __init__(self):
        self.vector_client = VectorSearchClient()
        self.data_dir = Path(__file__).parent / config.DATA_DIR
        self.chunk_index = self._build_chunk_index()
        
    def _build_chunk_index(self) -> Dict[str, Dict]:
        """Build an index of chunk_id -> chunk data for fast lookup"""
        chunks_file = self.data_dir / "all_chunks.json"
        if not chunks_file.exists():
            print(f"Warning: Chunks file not found: {chunks_file}")
            return {}
        
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        
        index = {}
        for chunk in chunks:
            chunk_id = chunk.get('id', chunk.get('chunk_id', ''))
            if chunk_id:
                index[chunk_id] = chunk
        
        print(f"Built chunk index with {len(index)} entries")
        return index
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query"""
        # Query vector search
        results = self.vector_client.query(query, num_neighbors=top_k)
        
        # Enrich with chunk content
        enriched_results = []
        for result in results:
            chunk_id = result['id']
            chunk_data = self.chunk_index.get(chunk_id, {})
            
            enriched_results.append({
                'chunk_id': chunk_id,
                'distance': result['distance'],
                'text': chunk_data.get('text', chunk_data.get('content', '')),
                'source_document': chunk_data.get('source_document', ''),
                'metadata': chunk_data.get('metadata', {})
            })
        
        return enriched_results
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieve and format context as a single string"""
        results = self.retrieve(query, top_k)
        
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(f"[Document {i+1}: {result['source_document']}]\n{result['text']}")
        
        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    retriever = RAGRetriever()
    
    # Test retrieval
    query = "What is the voltage rating for a transformer?"
    results = retriever.retrieve(query, top_k=3)
    
    print(f"\nQuery: {query}")
    print(f"Results:")
    for r in results:
        print(f"  - {r['source_document']}: {r['text'][:100]}...")
