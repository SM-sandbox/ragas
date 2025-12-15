#!/usr/bin/env python3
"""
Retrieval Metrics Evaluator

Calculates pure retrieval quality metrics for all embedding configurations:
- Recall@K: % of questions where the source document appears in top-K results
- Precision@K: % of top-K results that are from the correct source document
- MRR (Mean Reciprocal Rank): Average of 1/rank of first relevant result

Tests at K = 5, 10, 15, 20, 25, 50, 100

Usage:
    python retrieval_metrics_evaluator.py [--workers 15] [--max-questions N]
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.eval_config import (
    GCP_PROJECT, GCP_LOCATION,
    DEFAULT_WORKERS, DEFAULT_RETRIES,
    PROGRESS_BAR_CONFIG
)

# Paths
CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_200.json"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

# K values to test
K_VALUES = [5, 10, 15, 20, 25, 50, 100]

# Azure Configuration
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "https://asosearch-stg.search.windows.net")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", "")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "bf-demo")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://sponsored-eastus-oai.openai.azure.com/")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "eastus-text-embedding-3-large")
AZURE_API_VERSION = "2024-06-01"

# GCP Embedding configurations to test (endpoint IDs from working evaluations)
GCP_CONFIGS = [
    {
        "name": "text-embedding-005",
        "description": "text-embedding-005, 768 dim, no task type",
        "platform": "gcp",
        "embedding_model": "text-embedding-005",
        "task_type": None,
        "dimensions": 768,
        "endpoint_id": "1807654290668388352",
        "deployed_index_id": "idx_brightfoxai_evalv3_autoscale"
    },
    {
        "name": "gemini-SEMANTIC_SIMILARITY",
        "description": "gemini-embedding-001, 768 dim, SEMANTIC_SIMILARITY task type",
        "platform": "gcp",
        "embedding_model": "gemini-embedding-001",
        "task_type": "SEMANTIC_SIMILARITY",
        "dimensions": 768,
        "endpoint_id": "740301178981580800",
        "deployed_index_id": "idx_bfai_eval66_g1_768"
    },
    {
        "name": "gemini-RETRIEVAL_QUERY",
        "description": "gemini-embedding-001, 768 dim, RETRIEVAL_QUERY task type",
        "platform": "gcp",
        "embedding_model": "gemini-embedding-001",
        "task_type": "RETRIEVAL_QUERY",
        "dimensions": 768,
        "endpoint_id": "4639292556377587712",
        "deployed_index_id": "idx_bfai_eval66_g1_768_tt"
    },
]

# Azure configuration
AZURE_CONFIG = {
    "name": "azure-text-embedding-3-large",
    "description": "Azure text-embedding-3-large via Azure AI Search",
    "platform": "azure",
    "embedding_model": "text-embedding-3-large",
}

# All configs
EMBEDDING_CONFIGS = GCP_CONFIGS + [AZURE_CONFIG]


class RetrievalMetricsEvaluator:
    """Evaluates retrieval quality metrics across embedding configs."""
    
    def __init__(self):
        self.qa_corpus = self._load_corpus()
        # Filter to questions with source_document
        self.qa_corpus = [q for q in self.qa_corpus if q.get("source_document")]
        self.results = {}
        
    def _load_corpus(self) -> List[Dict]:
        """Load the Q&A corpus."""
        with open(CORPUS_PATH) as f:
            return json.load(f)
    
    def _get_gcp_embeddings(self, text: str, config: Dict) -> List[float]:
        """Get embeddings for text using GCP Vertex AI."""
        from google.cloud import aiplatform
        from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
        
        aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION)
        
        model = TextEmbeddingModel.from_pretrained(config["embedding_model"])
        
        if config["task_type"]:
            inputs = [TextEmbeddingInput(text=text, task_type=config["task_type"])]
            embeddings = model.get_embeddings(inputs, output_dimensionality=config["dimensions"])
        else:
            embeddings = model.get_embeddings([text], output_dimensionality=config["dimensions"])
        
        return embeddings[0].values
    
    def _get_azure_embeddings(self, text: str) -> List[float]:
        """Get embeddings using Azure OpenAI."""
        from openai import AzureOpenAI
        
        client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        response = client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING_DEPLOYMENT
        )
        return response.data[0].embedding
    
    def _query_gcp_vector_search(self, query_embedding: List[float], config: Dict, top_k: int = 100) -> List[Dict]:
        """Query GCP Vector Search endpoint and return results with source documents."""
        from google.cloud import aiplatform
        
        aiplatform.init(project=GCP_PROJECT, location=GCP_LOCATION)
        
        endpoint = aiplatform.MatchingEngineIndexEndpoint(config["endpoint_id"])
        
        response = endpoint.find_neighbors(
            deployed_index_id=config["deployed_index_id"],
            queries=[query_embedding],
            num_neighbors=top_k,
        )
        
        results = []
        for neighbor in response[0]:
            # Extract source document from chunk ID
            # Chunk IDs are formatted as: doc_name_chunk_N
            chunk_id = neighbor.id
            # Parse source document from chunk ID
            parts = chunk_id.rsplit("_chunk_", 1)
            source_doc = parts[0] if len(parts) > 1 else chunk_id
            
            results.append({
                "chunk_id": chunk_id,
                "source_document": source_doc,
                "distance": neighbor.distance
            })
        
        return results
    
    def _query_azure_search(self, query_embedding: List[float], top_k: int = 100) -> List[Dict]:
        """Query Azure AI Search and return results with source documents."""
        import requests
        
        search_url = f"{AZURE_SEARCH_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX}/docs/search?api-version=2023-11-01"
        
        search_body = {
            "vectorQueries": [{
                "kind": "vector",
                "vector": query_embedding,
                "fields": "embedding3",
                "k": top_k
            }],
            "select": "id,content,sourcefile,sourcepage",
            "top": top_k
        }
        
        headers = {
            "api-key": AZURE_SEARCH_KEY,
            "Content-Type": "application/json"
        }
        
        response = requests.post(search_url, headers=headers, json=search_body)
        
        if not response.ok:
            raise Exception(f"Azure search failed: {response.text}")
        
        results = []
        for r in response.json().get("value", []):
            source_file = r.get("sourcefile", "")
            # Extract document name from sourcefile path
            source_doc = Path(source_file).stem if source_file else r.get("id", "")
            
            results.append({
                "chunk_id": r.get("id", ""),
                "source_document": source_doc,
                "score": r.get("@search.score", 0)
            })
        
        return results
    
    def evaluate_single_question(
        self, 
        qa_item: Dict, 
        config: Dict,
        max_k: int = 100,
        retries: int = DEFAULT_RETRIES
    ) -> Dict:
        """Evaluate retrieval for a single question."""
        question_id = qa_item.get("id", hash(qa_item["question"]))
        expected_source = qa_item["source_document"]
        start_time = time.time()
        
        for attempt in range(retries):
            try:
                # Get query embedding and search based on platform
                if config.get("platform") == "azure":
                    query_embedding = self._get_azure_embeddings(qa_item["question"])
                    results = self._query_azure_search(query_embedding, top_k=max_k)
                else:
                    query_embedding = self._get_gcp_embeddings(qa_item["question"], config)
                    results = self._query_gcp_vector_search(query_embedding, config, top_k=max_k)
                
                # Find rank of first relevant result
                first_relevant_rank = None
                relevant_counts = {k: 0 for k in K_VALUES}
                
                for i, result in enumerate(results):
                    rank = i + 1
                    # Check if this result is from the expected source document
                    # Use fuzzy matching since chunk IDs may have variations
                    is_relevant = (
                        expected_source.lower() in result["source_document"].lower() or
                        result["source_document"].lower() in expected_source.lower()
                    )
                    
                    if is_relevant:
                        if first_relevant_rank is None:
                            first_relevant_rank = rank
                        
                        # Count relevant results at each K
                        for k in K_VALUES:
                            if rank <= k:
                                relevant_counts[k] += 1
                
                elapsed = time.time() - start_time
                
                return {
                    "question_id": question_id,
                    "question": qa_item["question"],
                    "expected_source": expected_source,
                    "first_relevant_rank": first_relevant_rank,
                    "relevant_counts": relevant_counts,
                    "total_results": len(results),
                    "success": True,
                    "time_seconds": round(elapsed, 2)
                }
                
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return {
                        "question_id": question_id,
                        "question": qa_item["question"],
                        "expected_source": expected_source,
                        "first_relevant_rank": None,
                        "relevant_counts": {k: 0 for k in K_VALUES},
                        "total_results": 0,
                        "success": False,
                        "error": str(e),
                        "time_seconds": round(time.time() - start_time, 2)
                    }
        
        return {
            "question_id": question_id,
            "success": False,
            "error": "Max retries exceeded"
        }
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate Recall@K, Precision@K, and MRR@K from results."""
        successful = [r for r in results if r["success"]]
        n = len(successful)
        
        if n == 0:
            return {}
        
        metrics = {}
        
        # Recall@K: % of questions where source doc appears in top-K
        for k in K_VALUES:
            hits = sum(1 for r in successful if r["relevant_counts"].get(k, 0) > 0)
            metrics[f"recall@{k}"] = hits / n
        
        # Precision@K: avg % of top-K results that are relevant
        for k in K_VALUES:
            precisions = [r["relevant_counts"].get(k, 0) / k for r in successful]
            metrics[f"precision@{k}"] = sum(precisions) / n
        
        # MRR: Mean Reciprocal Rank
        reciprocal_ranks = []
        for r in successful:
            if r["first_relevant_rank"]:
                reciprocal_ranks.append(1.0 / r["first_relevant_rank"])
            else:
                reciprocal_ranks.append(0.0)
        
        metrics["mrr"] = sum(reciprocal_ranks) / n
        
        # MRR@K: MRR considering only top-K
        for k in K_VALUES:
            rr_at_k = []
            for r in successful:
                rank = r["first_relevant_rank"]
                if rank and rank <= k:
                    rr_at_k.append(1.0 / rank)
                else:
                    rr_at_k.append(0.0)
            metrics[f"mrr@{k}"] = sum(rr_at_k) / n
        
        return metrics
    
    def run_evaluation(
        self,
        configs: List[Dict] = None,
        max_questions: Optional[int] = None,
        parallel_workers: int = DEFAULT_WORKERS
    ) -> Dict:
        """Run retrieval metrics evaluation for all configs."""
        
        if configs is None:
            configs = EMBEDDING_CONFIGS
        
        # Setup output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = OUTPUT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_retrieval_metrics"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        data_dir = experiment_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        corpus = self.qa_corpus[:max_questions] if max_questions else self.qa_corpus
        
        print("=" * 70)
        print("RETRIEVAL METRICS EVALUATION")
        print("=" * 70)
        print(f"Questions with source docs: {len(corpus)}")
        print(f"K values: {K_VALUES}")
        print(f"Configs to test: {[c['name'] for c in configs]}")
        print("=" * 70)
        
        all_results = {}
        checkpoint_interval = 10  # Save every 10 questions
        run_log = []  # Track timing and failures per config
        
        for config in configs:
            config_start_time = time.time()
            print(f"\n>>> Evaluating: {config['name']}")
            print(f"    {config['description']}")
            
            # Checkpoint file for this config
            checkpoint_file = data_dir / f"checkpoint_{config['name']}.json"
            
            # Load checkpoint if exists
            results = []
            completed_ids = set()
            if checkpoint_file.exists():
                with open(checkpoint_file) as f:
                    checkpoint_data = json.load(f)
                    results = checkpoint_data.get("results", [])
                    completed_ids = {r["question_id"] for r in results}
                print(f"    Loaded checkpoint: {len(results)} questions already done")
            
            # Filter to pending questions
            pending = [q for q in corpus if q.get("id", hash(q["question"])) not in completed_ids]
            
            if not pending:
                print(f"    All questions already evaluated!")
            else:
                pbar = tqdm(
                    total=len(pending),
                    desc=f"  {config['name'][:20]}",
                    **PROGRESS_BAR_CONFIG
                )
                
                # Sequential for now to avoid rate limits on embedding API
                for i, qa_item in enumerate(pending):
                    result = self.evaluate_single_question(qa_item, config)
                    results.append(result)
                    
                    # Update progress
                    successful = sum(1 for r in results if r["success"])
                    if successful > 0:
                        hits_at_10 = sum(1 for r in results if r["success"] and r["relevant_counts"].get(10, 0) > 0)
                        recall_10 = hits_at_10 / successful
                        pbar.set_postfix_str(f"R@10={recall_10:.2%}")
                    
                    pbar.update(1)
                    
                    # Checkpoint every 10 questions
                    if (i + 1) % checkpoint_interval == 0:
                        with open(checkpoint_file, 'w') as f:
                            json.dump({"config": config, "results": results}, f, indent=2)
                        
                pbar.close()
                
                # Final checkpoint save
                with open(checkpoint_file, 'w') as f:
                    json.dump({"config": config, "results": results}, f, indent=2)
            
            # Calculate metrics
            metrics = self.calculate_metrics(results)
            
            all_results[config['name']] = {
                "config": config,
                "metrics": metrics,
                "results": results
            }
            
            # Calculate timing and failures for this config
            config_elapsed = time.time() - config_start_time
            successful = sum(1 for r in results if r.get("success"))
            failed = len(results) - successful
            
            run_log.append({
                "config": config["name"],
                "questions": len(results),
                "successful": successful,
                "failed": failed,
                "time_seconds": round(config_elapsed, 2),
                "recall@10": metrics.get("recall@10", 0),
                "mrr": metrics.get("mrr", 0)
            })
            
            # Print summary
            print(f"    Recall@10: {metrics.get('recall@10', 0):.2%}")
            print(f"    MRR: {metrics.get('mrr', 0):.3f}")
            print(f"    Time: {config_elapsed:.1f}s | Success: {successful} | Failed: {failed}")
        
        # Print run log summary
        print("\n" + "=" * 70)
        print("RUN LOG SUMMARY")
        print("=" * 70)
        print(f"{'Config':<35} {'Questions':>10} {'Success':>10} {'Failed':>8} {'Time':>10} {'R@10':>8} {'MRR':>8}")
        print("-" * 70)
        total_time = 0
        total_success = 0
        total_failed = 0
        for entry in run_log:
            print(f"{entry['config']:<35} {entry['questions']:>10} {entry['successful']:>10} {entry['failed']:>8} {entry['time_seconds']:>8.1f}s {entry['recall@10']:>7.1%} {entry['mrr']:>8.3f}")
            total_time += entry['time_seconds']
            total_success += entry['successful']
            total_failed += entry['failed']
        print("-" * 70)
        print(f"{'TOTAL':<35} {total_success + total_failed:>10} {total_success:>10} {total_failed:>8} {total_time:>8.1f}s")
        print("=" * 70)
        
        # Save results
        output_file = data_dir / f"retrieval_metrics_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "k_values": K_VALUES,
                "total_questions": len(corpus),
                "run_log": run_log,
                "results": all_results
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Generate report
        self._generate_report(all_results, corpus)
        
        return all_results
    
    def _generate_report(self, all_results: Dict, corpus: List) -> None:
        """Generate markdown report."""
        lines = [
            "# Retrieval Metrics Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Questions Evaluated:** {len(corpus)}",
            "",
            "## Recall@K",
            "",
            "Percentage of questions where the source document appears in top-K results.",
            "",
        ]
        
        # Recall table header
        header = "| Config |"
        separator = "|--------|"
        for k in K_VALUES:
            header += f" R@{k} |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)
        
        for config_name, data in sorted(all_results.items()):
            row = f"| {config_name} |"
            for k in K_VALUES:
                val = data['metrics'].get(f'recall@{k}', 0)
                row += f" {val:.1%} |"
            lines.append(row)
        
        lines.extend([
            "",
            "## Precision@K",
            "",
            "Average percentage of top-K results that are from the correct source document.",
            "",
        ])
        
        # Precision table
        header = "| Config |"
        separator = "|--------|"
        for k in K_VALUES:
            header += f" P@{k} |"
            separator += "------|"
        lines.append(header)
        lines.append(separator)
        
        for config_name, data in sorted(all_results.items()):
            row = f"| {config_name} |"
            for k in K_VALUES:
                val = data['metrics'].get(f'precision@{k}', 0)
                row += f" {val:.1%} |"
            lines.append(row)
        
        lines.extend([
            "",
            "## MRR (Mean Reciprocal Rank)",
            "",
            "Average of 1/rank of the first relevant result.",
            "",
        ])
        
        # MRR table
        header = "| Config | MRR |"
        separator = "|--------|-----|"
        for k in K_VALUES:
            header += f" MRR@{k} |"
            separator += "--------|"
        lines.append(header)
        lines.append(separator)
        
        for config_name, data in sorted(all_results.items()):
            row = f"| {config_name} | {data['metrics'].get('mrr', 0):.3f} |"
            for k in K_VALUES:
                val = data['metrics'].get(f'mrr@{k}', 0)
                row += f" {val:.3f} |"
            lines.append(row)
        
        lines.extend([
            "",
            "## Metric Definitions",
            "",
            "- **Recall@K**: Did the correct source document appear anywhere in the top-K results?",
            "- **Precision@K**: What fraction of the top-K results came from the correct source?",
            "- **MRR**: How highly was the first relevant result ranked? (1/rank, averaged)",
            "- **MRR@K**: MRR but only considering results in top-K (0 if not in top-K)",
        ])
        
        # Save report
        REPORTS_DIR.mkdir(exist_ok=True)
        output_file = REPORTS_DIR / "Retrieval_Metrics_Report.md"
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        
        print(f"✓ Report saved to: {output_file}")


def main():
    """Run the retrieval metrics evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieval Metrics Evaluator")
    parser.add_argument("--workers", "-w", type=int, default=DEFAULT_WORKERS,
                        help=f"Parallel workers (default: {DEFAULT_WORKERS})")
    parser.add_argument("--max-questions", "-n", type=int, default=None,
                        help="Limit questions to evaluate")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Specific config to test (e.g., gemini-RETRIEVAL_QUERY)")
    args = parser.parse_args()
    
    evaluator = RetrievalMetricsEvaluator()
    
    configs = None
    if args.config:
        configs = [c for c in EMBEDDING_CONFIGS if c['name'] == args.config]
        if not configs:
            print(f"Config '{args.config}' not found. Available: {[c['name'] for c in EMBEDDING_CONFIGS]}")
            return
    
    evaluator.run_evaluation(
        configs=configs,
        max_questions=args.max_questions,
        parallel_workers=args.workers
    )


if __name__ == "__main__":
    main()
