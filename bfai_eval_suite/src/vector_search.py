"""Vertex AI Vector Search client for querying embeddings"""
import json
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import MatchingEngineIndexEndpoint
import vertexai
from vertexai.language_models import TextEmbeddingModel

from config import config


class VectorSearchClient:
    """Client for Vertex AI Vector Search"""
    
    def __init__(self):
        # Initialize Vertex AI
        vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION)
        aiplatform.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION)
        
        # Initialize embedding model
        self.embedding_model = TextEmbeddingModel.from_pretrained(config.EMBEDDING_MODEL)
        
        # Initialize index endpoint
        self.endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=config.VECTOR_SEARCH_ENDPOINT_ID,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LOCATION,
        )
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string"""
        embeddings = self.embedding_model.get_embeddings([text])
        return embeddings[0].values
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = self.embedding_model.get_embeddings(texts)
        return [e.values for e in embeddings]
    
    def query(
        self, 
        query_text: str, 
        num_neighbors: int = 10,
        filter_expr: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector search index with text"""
        # Get query embedding
        query_embedding = self.get_embedding(query_text)
        
        # Query the index
        response = self.endpoint.find_neighbors(
            deployed_index_id=config.VECTOR_SEARCH_DEPLOYED_INDEX_ID,
            queries=[query_embedding],
            num_neighbors=num_neighbors,
        )
        
        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                results.append({
                    "id": neighbor.id,
                    "distance": neighbor.distance,
                })
        
        return results
    
    def query_with_embedding(
        self,
        embedding: List[float],
        num_neighbors: int = 10
    ) -> List[Dict[str, Any]]:
        """Query the vector search index with a pre-computed embedding"""
        response = self.endpoint.find_neighbors(
            deployed_index_id=config.VECTOR_SEARCH_DEPLOYED_INDEX_ID,
            queries=[embedding],
            num_neighbors=num_neighbors,
        )
        
        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                results.append({
                    "id": neighbor.id,
                    "distance": neighbor.distance,
                })
        
        return results


if __name__ == "__main__":
    # Test the client
    client = VectorSearchClient()
    
    # Test query
    results = client.query("What is a transformer?", num_neighbors=5)
    print("Query results:")
    for r in results:
        print(f"  ID: {r['id']}, Distance: {r['distance']}")
