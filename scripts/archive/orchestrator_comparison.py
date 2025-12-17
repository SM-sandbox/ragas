#!/usr/bin/env python3
"""
Orchestrator vs Direct Comparison Test

Compares RAG performance between:
1. Direct mode: Hitting Vector Search + LLM directly (existing approach)
2. Orchestrator mode: Going through the RAG Orchestrator API

This measures the overhead added by the orchestrator layer and validates
that results are consistent between both approaches.

Usage:
    python orchestrator_comparison.py --mode orchestrator --job-id bfai__eval66_g1_768_tt
    python orchestrator_comparison.py --mode direct --config gemini-RETRIEVAL_QUERY
    python orchestrator_comparison.py --mode both  # Run both and compare
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from config import config

# Import direct retriever components
from google.cloud import aiplatform
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import HybridQuery
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import ChatVertexAI

try:
    from google import genai
    from google.genai.types import EmbedContentConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Paths
CORPUS_PATH = Path(__file__).parent.parent / "corpus" / "qa_corpus_200.json"
OUTPUT_DIR = Path(__file__).parent.parent / "experiments"
REPORTS_DIR = Path(__file__).parent.parent / "reports"

# Orchestrator config
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8080")
DEFAULT_JOB_ID = "bfai__eval66_g1_768_tt"  # gemini-RETRIEVAL_QUERY 768

# Direct config matching orchestrator job
DIRECT_CONFIG = {
    "name": "gemini-RETRIEVAL_QUERY",
    "description": "gemini-embedding-001, 768 dim, RETRIEVAL_QUERY task type",
    "embedding_model": "gemini-embedding-001",
    "task_type": "RETRIEVAL_QUERY",
    "dimensions": 768,
    "endpoint_id": "4639292556377587712",
    "deployed_index_id": "idx_bfai_eval66_g1_768_tt",
}


class SimpleSparseEmbedding:
    """Simple BM25-style sparse embedding generator."""
    
    def __init__(self):
        import re
        self.tokenize_pattern = re.compile(r'\b\w+\b')
    
    def generate_sparse_embedding(self, text: str) -> tuple:
        tokens = self.tokenize_pattern.findall(text.lower())
        if not tokens:
            return [], []
        
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        dimensions = []
        values = []
        for token, count in token_counts.items():
            dim = hash(token) % 1000000
            tf = count / len(tokens)
            dimensions.append(dim)
            values.append(float(tf))
        
        return dimensions, values


class OrchestratorClient:
    """Client for the RAG Orchestrator API."""
    
    def __init__(self, base_url: str = ORCHESTRATOR_URL, job_id: str = DEFAULT_JOB_ID):
        self.base_url = base_url.rstrip("/")
        self.job_id = job_id
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if orchestrator is running."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            return resp.ok
        except Exception:
            return False
    
    def query(self, question: str, top_k: int = 10) -> Dict:
        """Send query to orchestrator and get response with timing."""
        start_time = time.time()
        
        payload = {
            "query": question,
            "job_id": self.job_id,
            "top_k": top_k,
            "enable_hybrid": True,
            "enable_reranking": True,
        }
        
        try:
            resp = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            result = resp.json()
            
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "timing": result.get("timing", {}),
                "metadata": result.get("metadata", {}),
                "client_total_time": total_time,
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "client_total_time": time.time() - start_time,
            }


class DirectRetriever:
    """Direct retriever (same as embedding_comparison_direct.py)."""
    
    def __init__(
        self,
        embedding_model: str,
        endpoint_id: str,
        deployed_index_id: str,
        task_type: Optional[str] = None,
        dimensions: int = 768,
        project_id: str = "civic-athlete-473921-c0",
        location: str = "us-east1",
        recall_k: int = 100,
        precision_k: int = 10,
        enable_reranking: bool = True,
        enable_hybrid: bool = True,
        rrf_alpha: float = 0.5,
    ):
        self.embedding_model_name = embedding_model
        self.endpoint_id = endpoint_id
        self.deployed_index_id = deployed_index_id
        self.task_type = task_type
        self.dimensions = dimensions
        self.project_id = project_id
        self.location = location
        self.recall_k = recall_k
        self.precision_k = precision_k
        self.enable_reranking = enable_reranking
        self.enable_hybrid = enable_hybrid
        self.rrf_alpha = rrf_alpha
        
        aiplatform.init(project=project_id, location=location)
        
        if not task_type:
            self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model)
            self.genai_client = None
        else:
            self.embedding_model = None
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
            os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
            # Cache genai client for reuse
            if GENAI_AVAILABLE:
                self.genai_client = genai.Client()
            else:
                self.genai_client = None
        
        endpoint_resource = f"projects/689311309499/locations/{location}/indexEndpoints/{endpoint_id}"
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_resource
        )
        
        if enable_hybrid:
            self.sparse_generator = SimpleSparseEmbedding()
        
        if enable_reranking:
            self.rank_client = discoveryengine.RankServiceClient()
            self.ranking_config = f"projects/{project_id}/locations/global/rankingConfigs/default_ranking_config"
        
        self.chunk_index = self._load_chunks()
    
    def _load_chunks(self) -> Dict[str, Dict]:
        chunks_file = Path(__file__).parent / "data" / "all_chunks.json"
        if not chunks_file.exists():
            return {}
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        return {c.get('id', c.get('chunk_id', '')): c for c in chunks if c.get('id') or c.get('chunk_id')}
    
    def _get_embedding(self, text: str) -> List[float]:
        if self.task_type and self.genai_client:
            # Use cached genai client
            response = self.genai_client.models.embed_content(
                model=self.embedding_model_name,
                contents=[text],
                config=EmbedContentConfig(
                    task_type=self.task_type,
                    output_dimensionality=self.dimensions,
                ),
            )
            return list(response.embeddings[0].values)
        elif self.embedding_model_name.startswith("gemini") and self.genai_client:
            # Use cached genai client
            response = self.genai_client.models.embed_content(
                model=self.embedding_model_name,
                contents=[text],
                config=EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=self.dimensions,
                ),
            )
            return list(response.embeddings[0].values)
        else:
            embeddings = self.embedding_model.get_embeddings([text])
            return embeddings[0].values
    
    def _rerank(self, query: str, chunks: List[Dict], top_n: int) -> List[Dict]:
        if not chunks:
            return []
        
        records = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            if text:
                records.append(discoveryengine.RankingRecord(id=str(i), content=text[:10000]))
        
        if not records:
            return chunks[:top_n]
        
        request = discoveryengine.RankRequest(
            ranking_config=self.ranking_config,
            model="semantic-ranker-default@latest",
            query=query,
            records=records,
            top_n=top_n,
        )
        
        try:
            response = self.rank_client.rank(request=request)
            reranked = []
            for record in response.records:
                idx = int(record.id)
                chunk = chunks[idx].copy()
                chunk['rerank_score'] = record.score
                reranked.append(chunk)
            return reranked
        except Exception:
            return chunks[:top_n]
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        recall_k = self.recall_k
        query_embedding = self._get_embedding(query)
        
        if self.enable_hybrid:
            sparse_dims, sparse_vals = self.sparse_generator.generate_sparse_embedding(query)
            if sparse_dims and sparse_vals:
                hybrid_query = HybridQuery(
                    dense_embedding=query_embedding,
                    sparse_embedding_values=sparse_vals,
                    sparse_embedding_dimensions=sparse_dims,
                    rrf_ranking_alpha=self.rrf_alpha,
                )
                queries = [hybrid_query]
            else:
                queries = [query_embedding]
        else:
            queries = [query_embedding]
        
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=queries,
            num_neighbors=recall_k,
        )
        
        results = []
        if response and len(response) > 0:
            for neighbor in response[0]:
                chunk_id = neighbor.id
                chunk_data = self.chunk_index.get(chunk_id, {})
                results.append({
                    'chunk_id': chunk_id,
                    'distance': neighbor.distance,
                    'text': chunk_data.get('text', chunk_data.get('content', '')),
                    'source_document': chunk_data.get('source_document', ''),
                })
        
        if self.enable_reranking and results:
            precision_k = top_k if top_k else self.precision_k
            results = self._rerank(query, results, precision_k)
        elif top_k:
            results = results[:top_k]
        
        return results
    
    def retrieve_context(self, query: str, top_k: int = 10) -> str:
        results = self.retrieve(query, top_k)
        context_parts = []
        for i, r in enumerate(results):
            source = r.get('source_document', 'Unknown')
            text = r.get('text', '')
            context_parts.append(f"[{i+1}] {source}:\n{text}")
        return "\n\n---\n\n".join(context_parts)


class ComparisonEvaluator:
    """Evaluates and compares orchestrator vs direct performance."""
    
    def __init__(self, job_id: str = DEFAULT_JOB_ID):
        self.job_id = job_id
        self.qa_corpus = self._load_corpus()
        
        # Initialize judge LLM
        self.judge_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LLM_LOCATION,
            temperature=0.0,
        )
        
        # Initialize RAG LLM (for direct mode)
        self.rag_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LLM_LOCATION,
            temperature=0.0,
        )
        
        self.max_retries = 5
    
    def _load_corpus(self) -> List[Dict]:
        with open(CORPUS_PATH) as f:
            return json.load(f)
    
    def _generate_answer(self, question: str, context: str) -> str:
        prompt = f"""You are a technical assistant. Answer the question based ONLY on the provided context.
Be specific and accurate. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        response = self.rag_llm.invoke(prompt)
        return response.content
    
    def _judge_answer(self, question: str, ground_truth: str, rag_answer: str) -> Dict:
        prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG System Answer: {rag_answer}

Rate the RAG answer on these criteria (1-5 scale):
1. Correctness: Does it match the ground truth factually?
2. Completeness: Does it cover all key points from ground truth?
3. Relevance: Does it directly answer the question?

Respond in JSON format:
{{
    "correctness": <1-5>,
    "completeness": <1-5>,
    "relevance": <1-5>,
    "overall_score": <1-5>,
    "verdict": "pass" | "partial" | "fail"
}}

Rules: "pass" = overall_score >= 4, "partial" = 3, "fail" <= 2
"""
        for attempt in range(3):
            try:
                response = self.judge_llm.invoke(prompt)
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                return json.loads(content.strip())
            except Exception as e:
                if attempt == 2:
                    return {"error": str(e), "overall_score": 0, "verdict": "error"}
                time.sleep(1)
    
    def evaluate_orchestrator(
        self,
        max_questions: Optional[int] = None,
        orchestrator_url: str = ORCHESTRATOR_URL,
    ) -> Dict:
        """Run evaluation using orchestrator."""
        client = OrchestratorClient(orchestrator_url, self.job_id)
        
        if not client.health_check():
            raise RuntimeError(f"Orchestrator not reachable at {orchestrator_url}")
        
        corpus = self.qa_corpus[:max_questions] if max_questions else self.qa_corpus
        results = []
        
        print(f"\n>>> Orchestrator Evaluation: {len(corpus)} questions")
        print(f"    URL: {orchestrator_url}")
        print(f"    Job ID: {self.job_id}")
        
        pbar = tqdm(corpus, desc="Orchestrator", unit="q")
        
        for qa in pbar:
            question = qa["question"]
            ground_truth = qa.get("answer", "")
            
            start_time = time.time()
            
            # Query orchestrator
            response = client.query(question, top_k=10)
            
            if response["success"]:
                rag_answer = response["answer"]
                
                # Judge the answer
                judge_start = time.time()
                judgment = self._judge_answer(question, ground_truth, rag_answer)
                judge_time = time.time() - judge_start
                
                results.append({
                    "question_id": qa.get("id", hash(question)),
                    "question": question,
                    "success": True,
                    "rag_answer": rag_answer,
                    "judgment": judgment,
                    "timing": {
                        "total": time.time() - start_time,
                        "orchestrator": response.get("timing", {}),
                        "client_total": response["client_total_time"],
                        "judge": judge_time,
                    },
                })
            else:
                results.append({
                    "question_id": qa.get("id", hash(question)),
                    "question": question,
                    "success": False,
                    "error": response.get("error", "Unknown error"),
                    "timing": {"total": time.time() - start_time},
                })
            
            # Update progress
            successful = [r for r in results if r["success"]]
            if successful:
                avg_score = sum(r["judgment"].get("overall_score", 0) for r in successful) / len(successful)
                pbar.set_postfix_str(f"avg={avg_score:.2f}")
        
        return self._summarize_results(results, "orchestrator")
    
    def evaluate_direct(
        self,
        max_questions: Optional[int] = None,
    ) -> Dict:
        """Run evaluation using direct retriever."""
        retriever = DirectRetriever(
            embedding_model=DIRECT_CONFIG["embedding_model"],
            endpoint_id=DIRECT_CONFIG["endpoint_id"],
            deployed_index_id=DIRECT_CONFIG["deployed_index_id"],
            task_type=DIRECT_CONFIG["task_type"],
            dimensions=DIRECT_CONFIG["dimensions"],
        )
        
        corpus = self.qa_corpus[:max_questions] if max_questions else self.qa_corpus
        results = []
        
        print(f"\n>>> Direct Evaluation: {len(corpus)} questions")
        print(f"    Config: {DIRECT_CONFIG['name']}")
        
        pbar = tqdm(corpus, desc="Direct", unit="q")
        
        for qa in pbar:
            question = qa["question"]
            ground_truth = qa.get("answer", "")
            
            for attempt in range(self.max_retries):
                try:
                    start_time = time.time()
                    
                    # Retrieve context
                    retrieval_start = time.time()
                    context = retriever.retrieve_context(question, top_k=10)
                    retrieval_time = time.time() - retrieval_start
                    
                    # Generate answer
                    generation_start = time.time()
                    rag_answer = self._generate_answer(question, context)
                    generation_time = time.time() - generation_start
                    
                    # Judge answer
                    judge_start = time.time()
                    judgment = self._judge_answer(question, ground_truth, rag_answer)
                    judge_time = time.time() - judge_start
                    
                    results.append({
                        "question_id": qa.get("id", hash(question)),
                        "question": question,
                        "success": True,
                        "rag_answer": rag_answer,
                        "judgment": judgment,
                        "timing": {
                            "total": time.time() - start_time,
                            "retrieval": retrieval_time,
                            "generation": generation_time,
                            "judge": judge_time,
                        },
                    })
                    break
                    
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        results.append({
                            "question_id": qa.get("id", hash(question)),
                            "question": question,
                            "success": False,
                            "error": str(e),
                            "timing": {"total": time.time() - start_time},
                        })
                    else:
                        time.sleep(2 ** attempt)
            
            # Update progress
            successful = [r for r in results if r["success"]]
            if successful:
                avg_score = sum(r["judgment"].get("overall_score", 0) for r in successful) / len(successful)
                pbar.set_postfix_str(f"avg={avg_score:.2f}")
        
        return self._summarize_results(results, "direct")
    
    def _summarize_results(self, results: List[Dict], mode: str) -> Dict:
        """Summarize evaluation results."""
        successful = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]
        
        if not successful:
            return {
                "mode": mode,
                "total": len(results),
                "successful": 0,
                "failed": len(failed),
                "metrics": {},
                "timing": {},
                "results": results,
            }
        
        # Calculate metrics
        scores = [r["judgment"].get("overall_score", 0) for r in successful]
        verdicts = [r["judgment"].get("verdict", "error") for r in successful]
        
        metrics = {
            "avg_score": sum(scores) / len(scores),
            "pass_rate": verdicts.count("pass") / len(verdicts),
            "partial_rate": verdicts.count("partial") / len(verdicts),
            "fail_rate": verdicts.count("fail") / len(verdicts),
        }
        
        # Calculate timing
        timing = {
            "avg_total": sum(r["timing"]["total"] for r in successful) / len(successful),
        }
        
        if mode == "orchestrator":
            client_times = [r["timing"].get("client_total", 0) for r in successful]
            timing["avg_client_total"] = sum(client_times) / len(client_times) if client_times else 0
            
            # Extract orchestrator internal timing
            orch_timings = [r["timing"].get("orchestrator", {}) for r in successful]
            if orch_timings and orch_timings[0]:
                for key in ["retrieval", "ranking", "generation", "processing", "total"]:
                    vals = [t.get(key, 0) for t in orch_timings if t.get(key)]
                    if vals:
                        timing[f"orch_{key}"] = sum(vals) / len(vals)
        else:
            timing["avg_retrieval"] = sum(r["timing"].get("retrieval", 0) for r in successful) / len(successful)
            timing["avg_generation"] = sum(r["timing"].get("generation", 0) for r in successful) / len(successful)
        
        timing["avg_judge"] = sum(r["timing"].get("judge", 0) for r in successful) / len(successful)
        
        return {
            "mode": mode,
            "total": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "metrics": metrics,
            "timing": timing,
            "results": results,
        }
    
    def generate_comparison_report(
        self,
        orchestrator_results: Dict,
        direct_results: Dict,
    ) -> str:
        """Generate markdown comparison report."""
        lines = [
            "# Orchestrator vs Direct Comparison Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Job ID:** {self.job_id}",
            f"**Config:** {DIRECT_CONFIG['name']}",
            "",
            "## Summary",
            "",
            "| Metric | Orchestrator | Direct | Difference |",
            "|--------|--------------|--------|------------|",
        ]
        
        # Compare metrics
        orch_m = orchestrator_results.get("metrics", {})
        direct_m = direct_results.get("metrics", {})
        
        for metric in ["avg_score", "pass_rate"]:
            orch_val = orch_m.get(metric, 0)
            direct_val = direct_m.get(metric, 0)
            diff = orch_val - direct_val
            
            if "rate" in metric:
                lines.append(f"| {metric} | {orch_val:.1%} | {direct_val:.1%} | {diff:+.1%} |")
            else:
                lines.append(f"| {metric} | {orch_val:.2f} | {direct_val:.2f} | {diff:+.2f} |")
        
        lines.extend([
            "",
            "## Timing Comparison",
            "",
            "| Phase | Orchestrator | Direct | Overhead |",
            "|-------|--------------|--------|----------|",
        ])
        
        orch_t = orchestrator_results.get("timing", {})
        direct_t = direct_results.get("timing", {})
        
        # Total time
        orch_total = orch_t.get("avg_client_total", orch_t.get("avg_total", 0))
        direct_total = direct_t.get("avg_total", 0)
        overhead = orch_total - direct_total
        lines.append(f"| Total (client) | {orch_total:.2f}s | {direct_total:.2f}s | {overhead:+.2f}s |")
        
        # Retrieval
        orch_ret = orch_t.get("orch_retrieval", 0)
        direct_ret = direct_t.get("avg_retrieval", 0)
        lines.append(f"| Retrieval | {orch_ret:.2f}s | {direct_ret:.2f}s | {orch_ret - direct_ret:+.2f}s |")
        
        # Generation
        orch_gen = orch_t.get("orch_generation", 0)
        direct_gen = direct_t.get("avg_generation", 0)
        lines.append(f"| Generation | {orch_gen:.2f}s | {direct_gen:.2f}s | {orch_gen - direct_gen:+.2f}s |")
        
        lines.extend([
            "",
            "## Success/Failure",
            "",
            f"- **Orchestrator:** {orchestrator_results['successful']}/{orchestrator_results['total']} successful",
            f"- **Direct:** {direct_results['successful']}/{direct_results['total']} successful",
            "",
            "## Conclusion",
            "",
        ])
        
        # Determine if results are within margin of error
        score_diff = abs(orch_m.get("avg_score", 0) - direct_m.get("avg_score", 0))
        if score_diff < 0.2:
            lines.append("✅ **Results are within margin of error** - Orchestrator is working correctly.")
        else:
            lines.append(f"⚠️ **Score difference of {score_diff:.2f}** - May need investigation.")
        
        if overhead > 0:
            lines.append(f"\n📊 **Orchestrator adds {overhead:.2f}s overhead** per query on average.")
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare orchestrator vs direct RAG performance")
    parser.add_argument("--mode", choices=["orchestrator", "direct", "both"], default="both",
                        help="Evaluation mode")
    parser.add_argument("--job-id", default=DEFAULT_JOB_ID, help="Job ID for orchestrator")
    parser.add_argument("--orchestrator-url", default=ORCHESTRATOR_URL, help="Orchestrator URL")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit questions")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = OUTPUT_DIR / f"{datetime.now().strftime('%Y-%m-%d')}_orchestrator_comparison"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    data_dir = experiment_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    evaluator = ComparisonEvaluator(job_id=args.job_id)
    
    print("=" * 70)
    print("ORCHESTRATOR VS DIRECT COMPARISON")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Questions: {args.max_questions or 'all'}")
    print("=" * 70)
    
    orchestrator_results = None
    direct_results = None
    
    if args.mode in ["orchestrator", "both"]:
        try:
            orchestrator_results = evaluator.evaluate_orchestrator(
                max_questions=args.max_questions,
                orchestrator_url=args.orchestrator_url,
            )
            
            # Save results
            with open(data_dir / f"orchestrator_{timestamp}.json", 'w') as f:
                json.dump(orchestrator_results, f, indent=2, default=str)
            
            print(f"\n✓ Orchestrator: {orchestrator_results['successful']}/{orchestrator_results['total']} successful")
            print(f"  Avg score: {orchestrator_results['metrics'].get('avg_score', 0):.2f}")
            print(f"  Pass rate: {orchestrator_results['metrics'].get('pass_rate', 0):.1%}")
        except Exception as e:
            print(f"\n✗ Orchestrator evaluation failed: {e}")
    
    if args.mode in ["direct", "both"]:
        direct_results = evaluator.evaluate_direct(max_questions=args.max_questions)
        
        # Save results
        with open(data_dir / f"direct_{timestamp}.json", 'w') as f:
            json.dump(direct_results, f, indent=2, default=str)
        
        print(f"\n✓ Direct: {direct_results['successful']}/{direct_results['total']} successful")
        print(f"  Avg score: {direct_results['metrics'].get('avg_score', 0):.2f}")
        print(f"  Pass rate: {direct_results['metrics'].get('pass_rate', 0):.1%}")
    
    # Generate comparison report if both modes ran
    if orchestrator_results and direct_results:
        report = evaluator.generate_comparison_report(orchestrator_results, direct_results)
        
        report_path = REPORTS_DIR / "Orchestrator_vs_Direct_Comparison.md"
        REPORTS_DIR.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\n✓ Comparison report saved to: {report_path}")
        print("\n" + report)


if __name__ == "__main__":
    main()
