#!/usr/bin/env python3
"""
RAG Retriever with Google Reranker.

Uses direct Vector Search (which works) + Google Ranking API for reranking.
This tests the reranker without requiring the full orchestrator infrastructure.
"""
import json
from typing import List, Dict, Any
from pathlib import Path

from google.cloud import discoveryengine_v1 as discoveryengine

from config import config
from vector_search import VectorSearchClient


class RAGRetrieverWithReranker:
    """Retriever with Vector Search + Google Ranking API reranking."""
    
    def __init__(self):
        self.vector_client = VectorSearchClient()
        self.data_dir = Path(__file__).parent / config.DATA_DIR
        self.chunk_index = self._build_chunk_index()
        
        # Initialize reranker client
        self._rank_client = None
        self.ranking_model = config.RANKING_MODEL
        self.project_id = config.GCP_PROJECT_ID
    
    def _get_rank_client(self):
        """Lazy init ranking client."""
        if self._rank_client is None:
            self._rank_client = discoveryengine.RankServiceClient()
        return self._rank_client
    
    def _build_chunk_index(self) -> Dict[str, Dict]:
        """Build index of chunk_id -> chunk data."""
        chunks_file = self.data_dir / "all_chunks.json"
        if not chunks_file.exists():
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
    
    def retrieve(
        self,
        query: str,
        recall_k: int = 100,
        precision_k: int = 10,
        enable_reranking: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve with recall phase + precision reranking.
        
        Args:
            query: Search query
            recall_k: Number of candidates in recall phase
            precision_k: Number of results after reranking
            enable_reranking: Whether to use Google Ranking API
            
        Returns:
            List of enriched chunk results
        """
        # Phase 1: RECALL - Get candidates from vector search
        results = self.vector_client.query(query, num_neighbors=recall_k)
        
        # Enrich with chunk content
        enriched = []
        for result in results:
            chunk_id = result['id']
            chunk_data = self.chunk_index.get(chunk_id, {})
            
            enriched.append({
                'chunk_id': chunk_id,
                'distance': result['distance'],
                'text': chunk_data.get('text', chunk_data.get('content', '')),
                'source_document': chunk_data.get('source_document', ''),
                'metadata': chunk_data.get('metadata', {}),
            })
        
        if not enable_reranking or len(enriched) <= precision_k:
            return enriched[:precision_k]
        
        # Phase 2: PRECISION - Rerank with Google Ranking API
        try:
            reranked = self._rerank(query, enriched, precision_k)
            return reranked
        except Exception as e:
            print(f"Reranking failed: {e}, falling back to vector search order")
            return enriched[:precision_k]
    
    def _rerank(
        self,
        query: str,
        chunks: List[Dict],
        top_n: int,
    ) -> List[Dict]:
        """Rerank chunks using Google Ranking API."""
        client = self._get_rank_client()
        
        # Build ranking config path
        ranking_config = f"projects/{self.project_id}/locations/global/rankingConfigs/default_ranking_config"
        
        # Build records
        records = []
        for i, chunk in enumerate(chunks):
            record = discoveryengine.RankingRecord(
                id=str(i),  # Use index as ID
                content=chunk['text'][:1000],  # Truncate to token limit
            )
            records.append(record)
        
        # Create request
        request = discoveryengine.RankRequest(
            ranking_config=ranking_config,
            model=self.ranking_model,
            query=query,
            records=records,
            top_n=top_n,
        )
        
        # Call API
        response = client.rank(request=request)
        
        # Reorder chunks by ranking
        reranked = []
        for record in response.records:
            idx = int(record.id)
            chunk = chunks[idx].copy()
            chunk['ranking_score'] = record.score
            reranked.append(chunk)
        
        return reranked
    
    def retrieve_context(
        self,
        query: str,
        recall_k: int = 100,
        precision_k: int = 10,
        enable_reranking: bool = True,
    ) -> str:
        """Retrieve and format context as string."""
        results = self.retrieve(query, recall_k, precision_k, enable_reranking)
        
        context_parts = []
        for i, result in enumerate(results):
            context_parts.append(
                f"[Document {i+1}: {result['source_document']}]\n{result['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)


if __name__ == "__main__":
    # Test
    retriever = RAGRetrieverWithReranker()
    
    query = "What is the voltage rating for a transformer?"
    
    print(f"\nQuery: {query}")
    print("\n--- Without Reranking ---")
    results = retriever.retrieve(query, recall_k=20, precision_k=5, enable_reranking=False)
    for r in results:
        print(f"  {r['source_document']}: {r['text'][:80]}...")
    
    print("\n--- With Reranking ---")
    results = retriever.retrieve(query, recall_k=20, precision_k=5, enable_reranking=True)
    for r in results:
        score = r.get('ranking_score', 'N/A')
        print(f"  [score={score:.3f}] {r['source_document']}: {r['text'][:60]}...")
