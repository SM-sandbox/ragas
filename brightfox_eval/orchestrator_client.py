#!/usr/bin/env python3
"""
Client for calling the RAG Orchestrator API.

This calls the real sm-dev-01 RAG infrastructure instead of bypassing it
with direct vector search. This tests the full pipeline including:
- Hybrid search (semantic + keyword)
- Google Ranking API reranking
- Full orchestration layer
"""
import requests
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator client."""
    # API endpoint
    api_base_url: str = "http://localhost:8000"
    
    # Job configuration
    job_id: str = "brightfoxai__evaldocs66"
    
    # Retrieval settings
    recall_top_k: int = 100  # Number of candidates in recall phase
    precision_top_n: int = 12  # Number after reranking (precision)
    
    # Hybrid search settings (semantic vs keyword blend)
    enable_hybrid: bool = True
    rrf_ranking_alpha: float = 0.5  # 0.5 = 50% semantic, 50% keyword
    
    # Reranking settings
    enable_reranking: bool = True
    ranking_model: str = "semantic-ranker-default@latest"
    
    # Generation settings
    model: str = "gemini-2.5-flash"
    temperature: float = 0.0


class OrchestratorClient:
    """Client for the RAG Orchestrator API."""
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
    
    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        enable_hybrid: Optional[bool] = None,
        enable_reranking: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Query the RAG orchestrator.
        
        Args:
            query: The question to ask
            top_k: Override precision_top_n (number of final results)
            enable_hybrid: Override hybrid search setting
            enable_reranking: Override reranking setting
            
        Returns:
            Response dict with answer, sources, timing, metadata
        """
        url = f"{self.config.api_base_url}/query"
        
        payload = {
            "query": query,
            "job_id": self.config.job_id,
            "top_k": top_k or self.config.precision_top_n,
            "enable_hybrid": enable_hybrid if enable_hybrid is not None else self.config.enable_hybrid,
            "enable_reranking": enable_reranking if enable_reranking is not None else self.config.enable_reranking,
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to orchestrator at {self.config.api_base_url}. "
                "Make sure the RAG API is running:\n"
                "  cd /Users/scottmacon/Documents/GitHub/sm-dev-01\n"
                "  source .venv/bin/activate\n"
                "  python -m uvicorn services.api.app:app --host 0.0.0.0 --port 8000"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Orchestrator API error: {e.response.text}")
    
    def health_check(self) -> bool:
        """Check if the orchestrator is running."""
        try:
            response = requests.get(f"{self.config.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_answer_and_context(self, query: str, top_k: int = 12) -> tuple[str, str, List[Dict]]:
        """
        Get answer and context from orchestrator.
        
        Returns:
            Tuple of (answer, context_string, sources_list)
        """
        result = self.query(query, top_k=top_k)
        
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        
        # Build context string from sources
        context_parts = []
        for i, source in enumerate(sources):
            name = source.get("name", "Unknown")
            snippet = source.get("snippet", "")
            page = source.get("page", "")
            context_parts.append(f"[Document {i+1}: {name} (Page {page})]\n{snippet}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        return answer, context, sources


def create_orchestrator_client(
    api_url: str = "http://localhost:8000",
    job_id: str = "brightfoxai__evaldocs66",
    recall: int = 100,
    precision: int = 12,
    semantic_weight: float = 0.5,
    enable_reranking: bool = True,
) -> OrchestratorClient:
    """
    Factory function to create an orchestrator client.
    
    Args:
        api_url: Base URL of the orchestrator API
        job_id: Job ID for the document corpus
        recall: Number of candidates in recall phase
        precision: Number of results after reranking
        semantic_weight: Weight for semantic vs keyword (0.5 = 50-50)
        enable_reranking: Whether to use Google Ranking API
        
    Returns:
        Configured OrchestratorClient
    """
    config = OrchestratorConfig(
        api_base_url=api_url,
        job_id=job_id,
        recall_top_k=recall,
        precision_top_n=precision,
        rrf_ranking_alpha=semantic_weight,
        enable_reranking=enable_reranking,
    )
    return OrchestratorClient(config)


if __name__ == "__main__":
    # Test the client
    client = create_orchestrator_client()
    
    if client.health_check():
        print("✓ Orchestrator is running")
        
        # Test query
        result = client.query("What is the voltage rating for a transformer?")
        print(f"\nAnswer: {result['answer'][:200]}...")
        print(f"Sources: {len(result['sources'])}")
        print(f"Timing: {result['timing']}")
    else:
        print("✗ Orchestrator is not running")
        print("Start it with:")
        print("  cd /Users/scottmacon/Documents/GitHub/sm-dev-01")
        print("  source .venv/bin/activate")
        print("  python -m uvicorn services.api.app:app --host 0.0.0.0 --port 8000")
