#!/usr/bin/env python3
"""
Pure Semantic Retrieval Test - Direct vs Orchestrator

Tests retrieval quality using ONLY semantic search (no keyword/sparse):
- 3 GCP embedding configs: text-embedding-005, gemini-768, gemini-1536
- 2 paths: Direct API and Orchestrator
- Metrics: Recall@K, MRR@K at K=5,10,15,20,25,50,100
- Timing: Per-question and average retrieval time

Usage:
    python semantic_retrieval_test.py --mode direct
    python semantic_retrieval_test.py --mode orchestrator
    python semantic_retrieval_test.py --mode both
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from google.cloud import aiplatform

try:
    from google import genai
    from google.genai.types import EmbedContentConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from vertexai.language_models import TextEmbeddingModel

# Paths
CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_200.json"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

# K values for metrics
K_VALUES = [5, 10, 15, 20, 25, 50, 100]

# GCP Project
GCP_PROJECT = "civic-athlete-473921-c0"
GCP_PROJECT_NUMBER = "689311309499"
GCP_LOCATION = "us-east1"

# Orchestrator
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8080")

# 3 GCP Embedding Configs to test
EMBEDDING_CONFIGS = [
    {
        "name": "text-embedding-005",
        "description": "text-embedding-005, 768 dim, no task type",
        "embedding_model": "text-embedding-005",
        "task_type": None,
        "dimensions": 768,
        "endpoint_id": "1807654290668388352",
        "deployed_index_id": "idx_brightfoxai_evalv3_autoscale",
        "job_id": "brightfoxai__evalv3",
    },
    {
        "name": "gemini-768-RETRIEVAL_QUERY",
        "description": "gemini-embedding-001, 768 dim, RETRIEVAL_QUERY task type",
        "embedding_model": "gemini-embedding-001",
        "task_type": "RETRIEVAL_QUERY",
        "dimensions": 768,
        "endpoint_id": "4639292556377587712",
        "deployed_index_id": "idx_bfai_eval66_g1_768_tt",
        "job_id": "bfai__eval66_g1_768_tt",
    },
    {
        "name": "gemini-1536-RETRIEVAL_QUERY",
        "description": "gemini-embedding-001, 1536 dim, RETRIEVAL_QUERY task type",
        "embedding_model": "gemini-embedding-001",
        "task_type": "RETRIEVAL_QUERY",
        "dimensions": 1536,
        "endpoint_id": "3594457442827632640",
        "deployed_index_id": "idx_bfai_eval66a_g1_1536_tt",
        "job_id": "bfai__eval66a_g1_1536_tt",
    },
    {
        "name": "gemini-3072-RETRIEVAL_QUERY",
        "description": "gemini-embedding-001, 3072 dim, RETRIEVAL_QUERY task type",
        "embedding_model": "gemini-embedding-001",
        "task_type": "RETRIEVAL_QUERY",
        "dimensions": 3072,
        "endpoint_id": "6820160675931750400",
        "deployed_index_id": "idx_bfai_eval66a_g1_3072_tt",
        "job_id": "bfai__eval66_g1_3072",  # Use existing job (no _tt suffix in orchestrator)
    },
]


class DirectRetriever:
    """Direct retriever using pure semantic search (no keyword/sparse)."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.embedding_model_name = config["embedding_model"]
        self.task_type = config.get("task_type")
        self.dimensions = config["dimensions"]
        self.endpoint_id = config["endpoint_id"]
        self.deployed_index_id = config["deployed_index_id"]
        
        # Initialize AI Platform
        aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION)
        
        # Initialize embedding model
        if not self.task_type:
            self.embedding_model = TextEmbeddingModel.from_pretrained(self.embedding_model_name)
            self.genai_client = None
        else:
            self.embedding_model = None
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", GCP_PROJECT)
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
            os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
            if GENAI_AVAILABLE:
                self.genai_client = genai.Client()
            else:
                raise RuntimeError("google-genai not available for task_type embeddings")
        
        # Get Vector Search endpoint
        endpoint_resource = f"projects/{GCP_PROJECT_NUMBER}/locations/{GCP_LOCATION}/indexEndpoints/{self.endpoint_id}"
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_resource
        )
        
        # Load chunk metadata for source document matching
        self.chunk_index = self._load_chunks()
    
    def _load_chunks(self) -> Dict[str, Dict]:
        """Load chunk data from local file."""
        chunks_file = Path(__file__).parent / "data" / "all_chunks.json"
        if not chunks_file.exists():
            return {}
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        return {c.get('id', c.get('chunk_id', '')): c for c in chunks if c.get('id') or c.get('chunk_id')}
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text."""
        if self.task_type and self.genai_client:
            response = self.genai_client.models.embed_content(
                model=self.embedding_model_name,
                contents=[text],
                config=EmbedContentConfig(
                    task_type=self.task_type,
                    output_dimensionality=self.dimensions,
                ),
            )
            return list(response.embeddings[0].values)
        else:
            embeddings = self.embedding_model.get_embeddings([text])
            return embeddings[0].values
    
    def retrieve(self, query: str, top_k: int = 100) -> tuple:
        """
        Retrieve using PURE SEMANTIC search (no keyword/sparse).
        Returns (results, retrieval_time_seconds).
        """
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Query Vector Search - PURE SEMANTIC (dense only, no hybrid)
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=top_k,
        )
        
        retrieval_time = time.time() - start_time
        
        # Extract results with source documents
        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                chunk_id = neighbor.id
                chunk_data = self.chunk_index.get(chunk_id, {})
                
                # Parse source document from chunk ID if not in metadata
                source_doc = chunk_data.get('source_document', '')
                if not source_doc:
                    parts = chunk_id.rsplit("_chunk_", 1)
                    source_doc = parts[0] if len(parts) > 1 else chunk_id
                
                results.append({
                    'chunk_id': chunk_id,
                    'source_document': source_doc,
                    'distance': neighbor.distance,
                })
        
        return results, retrieval_time


class OrchestratorRetriever:
    """
    Fast retriever that imports orchestrator components directly.
    Skips HTTP overhead and LLM generation for pure retrieval testing.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.job_id = config["job_id"]
        self._retriever = None
        self._init_retriever()
    
    def _init_retriever(self):
        """Initialize the VectorSearchRetriever directly."""
        import sys
        sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")
        
        from libs.core.gcp_config import get_jobs_config
        from services.api.retrieval.vector_search import VectorSearchRetriever
        
        jobs = get_jobs_config()
        job_config = jobs.get(self.job_id, {})
        job_config["job_id"] = self.job_id
        
        self._retriever = VectorSearchRetriever(job_config)
    
    def health_check(self) -> bool:
        return self._retriever is not None
    
    def retrieve(self, query: str, top_k: int = 100) -> tuple:
        """
        Retrieve using orchestrator's VectorSearchRetriever directly.
        Returns (results, retrieval_time_seconds, internal_time).
        """
        from services.api.core.config import QueryConfig
        
        start_time = time.time()
        
        # Create config for pure semantic search
        config = QueryConfig(
            recall_top_k=top_k,
            enable_hybrid=False,  # PURE SEMANTIC
            rrf_ranking_alpha=1.0,  # 100% dense
        )
        
        try:
            result = self._retriever.retrieve(query, config)
            retrieval_time = time.time() - start_time
            
            # Extract results
            results = []
            for chunk in result.chunks:
                # Extract source_document from doc_name or chunk_id
                source_doc = chunk.doc_name or ""
                if not source_doc and chunk.chunk_id:
                    parts = chunk.chunk_id.rsplit("_chunk_", 1)
                    source_doc = parts[0] if len(parts) > 1 else chunk.chunk_id
                
                results.append({
                    'chunk_id': chunk.chunk_id,
                    'source_document': source_doc,
                    'score': chunk.score,
                })
            
            return results, retrieval_time, result.retrieval_time_seconds
            
        except Exception as e:
            print(f"Retrieval error: {e}")
            return [], time.time() - start_time, 0


class SemanticRetrievalTester:
    """Tests pure semantic retrieval across configs and paths."""
    
    def __init__(self):
        self.qa_corpus = self._load_corpus()
    
    def _load_corpus(self) -> List[Dict]:
        with open(CORPUS_PATH) as f:
            corpus = json.load(f)
        # Only include questions with source_document for retrieval testing
        return [q for q in corpus if q.get("source_document")]
    
    def evaluate_config_direct(self, config: Dict, max_questions: Optional[int] = None) -> Dict:
        """Evaluate a config using direct API."""
        retriever = DirectRetriever(config)
        corpus = self.qa_corpus[:max_questions] if max_questions else self.qa_corpus
        
        results = []
        total_retrieval_time = 0
        
        pbar = tqdm(corpus, desc=f"Direct: {config['name'][:25]}", unit="q")
        
        for qa in pbar:
            question = qa["question"]
            expected_source = qa.get("source_document", "")
            
            # Retrieve
            retrieved, retrieval_time = retriever.retrieve(question, top_k=100)
            total_retrieval_time += retrieval_time
            
            # Calculate metrics
            first_relevant_rank = None
            relevant_counts = {k: 0 for k in K_VALUES}
            
            for i, result in enumerate(retrieved):
                rank = i + 1
                is_relevant = self._is_relevant(result["source_document"], expected_source)
                
                if is_relevant:
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    for k in K_VALUES:
                        if rank <= k:
                            relevant_counts[k] += 1
            
            results.append({
                "question_id": qa.get("id", hash(question)),
                "question": question,
                "expected_source": expected_source,
                "first_relevant_rank": first_relevant_rank,
                "relevant_counts": relevant_counts,
                "retrieval_time": retrieval_time,
                "total_results": len(retrieved),
            })
            
            # Update progress
            if results:
                hits = sum(1 for r in results if r["first_relevant_rank"] and r["first_relevant_rank"] <= 10)
                pbar.set_postfix_str(f"R@10={hits/len(results):.1%}, t={retrieval_time:.2f}s")
        
        return self._summarize_results(results, config, "direct", total_retrieval_time)
    
    def evaluate_config_orchestrator(self, config: Dict, max_questions: Optional[int] = None) -> Dict:
        """Evaluate a config using orchestrator API."""
        retriever = OrchestratorRetriever(config)
        
        if not retriever.health_check():
            raise RuntimeError(f"Orchestrator not reachable at {ORCHESTRATOR_URL}")
        
        corpus = self.qa_corpus[:max_questions] if max_questions else self.qa_corpus
        
        results = []
        total_retrieval_time = 0
        total_internal_time = 0
        
        pbar = tqdm(corpus, desc=f"Orch: {config['name'][:25]}", unit="q")
        
        for qa in pbar:
            question = qa["question"]
            expected_source = qa.get("source_document", "")
            
            # Retrieve
            retrieved, retrieval_time, internal_time = retriever.retrieve(question, top_k=100)
            total_retrieval_time += retrieval_time
            total_internal_time += internal_time
            
            # Calculate metrics
            first_relevant_rank = None
            relevant_counts = {k: 0 for k in K_VALUES}
            
            for i, result in enumerate(retrieved):
                rank = i + 1
                is_relevant = self._is_relevant(result["source_document"], expected_source)
                
                if is_relevant:
                    if first_relevant_rank is None:
                        first_relevant_rank = rank
                    for k in K_VALUES:
                        if rank <= k:
                            relevant_counts[k] += 1
            
            results.append({
                "question_id": qa.get("id", hash(question)),
                "question": question,
                "expected_source": expected_source,
                "first_relevant_rank": first_relevant_rank,
                "relevant_counts": relevant_counts,
                "retrieval_time": retrieval_time,
                "internal_retrieval_time": internal_time,
                "total_results": len(retrieved),
            })
            
            # Update progress
            if results:
                hits = sum(1 for r in results if r["first_relevant_rank"] and r["first_relevant_rank"] <= 10)
                pbar.set_postfix_str(f"R@10={hits/len(results):.1%}, t={retrieval_time:.2f}s")
        
        summary = self._summarize_results(results, config, "orchestrator", total_retrieval_time)
        summary["timing"]["total_internal_retrieval"] = total_internal_time
        summary["timing"]["avg_internal_retrieval"] = total_internal_time / len(results) if results else 0
        return summary
    
    def _is_relevant(self, retrieved_source: str, expected_source: str) -> bool:
        """Check if retrieved source matches expected source."""
        if not expected_source:
            return False
        return (
            expected_source.lower() in retrieved_source.lower() or
            retrieved_source.lower() in expected_source.lower()
        )
    
    def _summarize_results(self, results: List[Dict], config: Dict, mode: str, total_time: float) -> Dict:
        """Summarize evaluation results."""
        n = len(results)
        if n == 0:
            return {"config": config, "mode": mode, "metrics": {}, "timing": {}}
        
        metrics = {}
        
        # Recall@K
        for k in K_VALUES:
            hits = sum(1 for r in results if r["relevant_counts"].get(k, 0) > 0)
            metrics[f"recall@{k}"] = hits / n
        
        # MRR and MRR@K
        reciprocal_ranks = []
        for r in results:
            if r["first_relevant_rank"]:
                reciprocal_ranks.append(1.0 / r["first_relevant_rank"])
            else:
                reciprocal_ranks.append(0.0)
        
        metrics["mrr"] = sum(reciprocal_ranks) / n
        
        for k in K_VALUES:
            rr_at_k = []
            for r in results:
                rank = r["first_relevant_rank"]
                if rank and rank <= k:
                    rr_at_k.append(1.0 / rank)
                else:
                    rr_at_k.append(0.0)
            metrics[f"mrr@{k}"] = sum(rr_at_k) / n
        
        # Timing
        timing = {
            "total_retrieval": total_time,
            "avg_retrieval": total_time / n,
            "min_retrieval": min(r["retrieval_time"] for r in results),
            "max_retrieval": max(r["retrieval_time"] for r in results),
        }
        
        return {
            "config": config,
            "mode": mode,
            "total_questions": n,
            "metrics": metrics,
            "timing": timing,
            "results": results,
        }
    
    def run_evaluation(
        self,
        mode: str = "both",
        configs: List[Dict] = None,
        max_questions: Optional[int] = None,
    ) -> Dict:
        """Run full evaluation."""
        if configs is None:
            configs = EMBEDDING_CONFIGS
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = OUTPUT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_semantic_retrieval"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        data_dir = experiment_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        print("=" * 70)
        print("PURE SEMANTIC RETRIEVAL TEST")
        print("=" * 70)
        print(f"Mode: {mode}")
        print(f"Questions: {max_questions or len(self.qa_corpus)}")
        print(f"Configs: {[c['name'] for c in configs]}")
        print(f"Search: PURE SEMANTIC (no keyword/sparse)")
        print("=" * 70)
        
        all_results = {"direct": {}, "orchestrator": {}}
        
        # Direct tests
        if mode in ["direct", "both"]:
            print("\n>>> DIRECT API TESTS")
            for config in configs:
                print(f"\n  Testing: {config['name']}")
                result = self.evaluate_config_direct(config, max_questions)
                all_results["direct"][config["name"]] = result
                
                print(f"    Recall@10: {result['metrics'].get('recall@10', 0):.1%}")
                print(f"    MRR: {result['metrics'].get('mrr', 0):.3f}")
                print(f"    Avg retrieval: {result['timing']['avg_retrieval']:.3f}s")
        
        # Orchestrator tests
        if mode in ["orchestrator", "both"]:
            print("\n>>> ORCHESTRATOR API TESTS")
            for config in configs:
                print(f"\n  Testing: {config['name']}")
                try:
                    result = self.evaluate_config_orchestrator(config, max_questions)
                    all_results["orchestrator"][config["name"]] = result
                    
                    print(f"    Recall@10: {result['metrics'].get('recall@10', 0):.1%}")
                    print(f"    MRR: {result['metrics'].get('mrr', 0):.3f}")
                    print(f"    Avg retrieval: {result['timing']['avg_retrieval']:.3f}s")
                except Exception as e:
                    print(f"    ERROR: {e}")
        
        # Save results
        output_file = data_dir / f"semantic_retrieval_{mode}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "mode": mode,
                "k_values": K_VALUES,
                "total_questions": max_questions or len(self.qa_corpus),
                "results": all_results,
            }, f, indent=2, default=str)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Generate report
        self._generate_report(all_results, mode, max_questions or len(self.qa_corpus))
        
        return all_results
    
    def _generate_report(self, all_results: Dict, mode: str, total_questions: int) -> None:
        """Generate markdown report."""
        lines = [
            "# Pure Semantic Retrieval Test Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Questions:** {total_questions}",
            f"**Mode:** {mode}",
            f"**Search:** Pure Semantic (no keyword/sparse)",
            "",
        ]
        
        # Summary table
        if mode == "both" and all_results["direct"] and all_results["orchestrator"]:
            lines.extend([
                "## Summary Comparison",
                "",
                "| Config | Mode | Recall@10 | MRR | Avg Time |",
                "|--------|------|-----------|-----|----------|",
            ])
            
            for config_name in all_results["direct"]:
                direct = all_results["direct"].get(config_name, {})
                orch = all_results["orchestrator"].get(config_name, {})
                
                d_r10 = direct.get("metrics", {}).get("recall@10", 0)
                d_mrr = direct.get("metrics", {}).get("mrr", 0)
                d_time = direct.get("timing", {}).get("avg_retrieval", 0)
                
                o_r10 = orch.get("metrics", {}).get("recall@10", 0)
                o_mrr = orch.get("metrics", {}).get("mrr", 0)
                o_time = orch.get("timing", {}).get("avg_retrieval", 0)
                
                lines.append(f"| {config_name} | Direct | {d_r10:.1%} | {d_mrr:.3f} | {d_time:.3f}s |")
                lines.append(f"| {config_name} | Orchestrator | {o_r10:.1%} | {o_mrr:.3f} | {o_time:.3f}s |")
            
            lines.append("")
        
        # Recall@K tables
        for path in ["direct", "orchestrator"]:
            if not all_results.get(path):
                continue
            
            lines.extend([
                f"## Recall@K ({path.title()})",
                "",
                "| Config | R@5 | R@10 | R@15 | R@20 | R@25 | R@50 | R@100 |",
                "|--------|-----|------|------|------|------|------|-------|",
            ])
            
            for config_name, data in all_results[path].items():
                metrics = data.get("metrics", {})
                row = f"| {config_name} |"
                for k in K_VALUES:
                    row += f" {metrics.get(f'recall@{k}', 0):.1%} |"
                lines.append(row)
            
            lines.append("")
        
        # MRR tables
        for path in ["direct", "orchestrator"]:
            if not all_results.get(path):
                continue
            
            lines.extend([
                f"## MRR@K ({path.title()})",
                "",
                "| Config | MRR | MRR@5 | MRR@10 | MRR@20 | MRR@50 | MRR@100 |",
                "|--------|-----|-------|--------|--------|--------|---------|",
            ])
            
            for config_name, data in all_results[path].items():
                metrics = data.get("metrics", {})
                lines.append(f"| {config_name} | {metrics.get('mrr', 0):.3f} | {metrics.get('mrr@5', 0):.3f} | {metrics.get('mrr@10', 0):.3f} | {metrics.get('mrr@20', 0):.3f} | {metrics.get('mrr@50', 0):.3f} | {metrics.get('mrr@100', 0):.3f} |")
            
            lines.append("")
        
        # Timing table
        lines.extend([
            "## Timing",
            "",
            "| Config | Mode | Avg | Min | Max | Total |",
            "|--------|------|-----|-----|-----|-------|",
        ])
        
        for path in ["direct", "orchestrator"]:
            if not all_results.get(path):
                continue
            for config_name, data in all_results[path].items():
                timing = data.get("timing", {})
                lines.append(f"| {config_name} | {path} | {timing.get('avg_retrieval', 0):.3f}s | {timing.get('min_retrieval', 0):.3f}s | {timing.get('max_retrieval', 0):.3f}s | {timing.get('total_retrieval', 0):.1f}s |")
        
        lines.append("")
        
        # Save report
        report_path = REPORTS_DIR / "Semantic_Retrieval_Test_Report.md"
        REPORTS_DIR.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"✓ Report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Pure semantic retrieval test")
    parser.add_argument("--mode", choices=["direct", "orchestrator", "both"], default="both")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Test specific config only")
    
    args = parser.parse_args()
    
    tester = SemanticRetrievalTester()
    
    configs = EMBEDDING_CONFIGS
    if args.config:
        configs = [c for c in EMBEDDING_CONFIGS if args.config.lower() in c["name"].lower()]
        if not configs:
            print(f"Config not found: {args.config}")
            print(f"Available: {[c['name'] for c in EMBEDDING_CONFIGS]}")
            sys.exit(1)
    
    tester.run_evaluation(
        mode=args.mode,
        configs=configs,
        max_questions=args.max_questions,
    )


if __name__ == "__main__":
    main()
