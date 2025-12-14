"""Configuration for BrightFox RAG Evaluation"""
import os
from dataclasses import dataclass

@dataclass
class Config:
    # GCP Configuration
    GCP_PROJECT_ID: str = "civic-athlete-473921-c0"  # For Vector Search
    GCP_LOCATION: str = "us-east1"  # For Vector Search
    GCP_LLM_PROJECT: str = "bf-rag-sandbox-scott"  # For Gemini LLM
    GCP_LLM_LOCATION: str = "us-central1"  # For Gemini LLM
    
    # Vertex AI Vector Search
    VECTOR_SEARCH_ENDPOINT_ID: str = "1807654290668388352"
    VECTOR_SEARCH_DEPLOYED_INDEX_ID: str = "idx_brightfoxai_evalv3_autoscale"
    VECTOR_SEARCH_PUBLIC_ENDPOINT: str = "669919480.us-east1-689311309499.vdb.vertexai.goog"
    INDEX_RESOURCE: str = "projects/689311309499/locations/us-east1/indexes/8315830741241954304"
    
    # Embedding Model
    EMBEDDING_MODEL: str = "text-embedding-005"
    EMBEDDING_DIMENSIONS: int = 768
    
    # GCS Bucket
    GCS_BUCKET: str = "brightfoxai-documents"
    GCS_CHUNKS_PATH: str = "BRIGHTFOXAI/EVALV3/processed"
    
    # LLM Configuration
    LLM_MODEL: str = "gemini-2.5-flash"  # Gemini 2.5 Flash
    
    # Evaluation settings
    NUM_SINGLE_HOP_QUESTIONS: int = 25
    NUM_MULTI_HOP_QUESTIONS: int = 25
    MIN_QUALITY_SCORE: int = 4  # Keep questions rated 4 or 5
    
    # Paths
    DATA_DIR: str = "data"
    OUTPUT_DIR: str = "output"


config = Config()
