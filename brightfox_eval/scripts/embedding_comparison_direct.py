#!/usr/bin/env python3
"""
Direct Embedding Comparison Test - No Orchestrator.

Tests RAG retrieval quality across different embedding configurations
by hitting Vector Search directly. This is faster and avoids orchestrator complexity.

Configurations tested:
1. text-embedding-005 (no task type)
2. gemini-embedding-001 (no task type)  
3. gemini-embedding-001 with RETRIEVAL_QUERY task type
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from tqdm import tqdm

from google.cloud import aiplatform
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import HybridQuery
from vertexai.language_models import TextEmbeddingModel
from langchain_google_vertexai import ChatVertexAI

from config import config

# For Gemini embeddings with task_type
try:
    from google import genai
    from google.genai.types import EmbedContentConfig
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-genai not available, task_type embeddings disabled")


class SimpleSparseEmbedding:
    """Simple BM25-style sparse embedding generator for hybrid search."""
    
    def __init__(self):
        import re
        self.tokenize_pattern = re.compile(r'\b\w+\b')
    
    def generate_sparse_embedding(self, text: str) -> tuple:
        """Generate sparse embedding (token indices and values)."""
        import re
        # Simple tokenization
        tokens = self.tokenize_pattern.findall(text.lower())
        
        if not tokens:
            return [], []
        
        # Count token frequencies
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Convert to sparse format (hash tokens to indices, use TF as values)
        dimensions = []
        values = []
        for token, count in token_counts.items():
            # Hash token to dimension (0 to 999999)
            dim = hash(token) % 1000000
            tf = count / len(tokens)  # Term frequency
            dimensions.append(dim)
            values.append(float(tf))
        
        return dimensions, values


# The 3 index configurations to test
EMBEDDING_CONFIGS = [
    {
        "name": "text-embedding-005",
        "description": "text-embedding-005, no task type (baseline)",
        "embedding_model": "text-embedding-005",
        "task_type": None,
        "dimensions": 768,
        "endpoint_id": "1807654290668388352",
        "deployed_index_id": "idx_brightfoxai_evalv3_autoscale",
    },
    {
        "name": "gemini-SEMANTIC_SIMILARITY",
        "description": "gemini-embedding-001, 768 dim, SEMANTIC_SIMILARITY task type",
        "embedding_model": "gemini-embedding-001",
        "task_type": "SEMANTIC_SIMILARITY",  # Try SEMANTIC_SIMILARITY for the non-retrieval index
        "dimensions": 768,
        "endpoint_id": "740301178981580800",
        "deployed_index_id": "idx_bfai_eval66_g1_768",
    },
    {
        "name": "gemini-RETRIEVAL_QUERY",
        "description": "gemini-embedding-001, 768 dim, WITH RETRIEVAL_QUERY task type",
        "embedding_model": "gemini-embedding-001",
        "task_type": "RETRIEVAL_QUERY",
        "dimensions": 768,
        "endpoint_id": "4639292556377587712",
        "deployed_index_id": "idx_bfai_eval66_g1_768_tt",
    },
    {
        "name": "gemini-1536-RETRIEVAL_QUERY",
        "description": "gemini-embedding-001, 1536 dim, WITH RETRIEVAL_QUERY task type",
        "embedding_model": "gemini-embedding-001",
        "task_type": "RETRIEVAL_QUERY",
        "dimensions": 1536,
        "endpoint_id": "3594457442827632640",
        "deployed_index_id": "idx_bfai_eval66a_g1_1536_tt",
    },
]


class DirectRetriever:
    """Direct retriever using Vector Search + Google Ranking API reranking."""
    
    def __init__(
        self,
        embedding_model: str,
        endpoint_id: str,
        deployed_index_id: str,
        task_type: Optional[str] = None,
        dimensions: int = 768,
        project_id: str = "civic-athlete-473921-c0",
        location: str = "us-east1",
        recall_k: int = 100,  # Recall phase: get more candidates
        precision_k: int = 5,  # Precision phase: rerank to top 5
        enable_reranking: bool = True,
        ranking_model: str = "semantic-ranker-default@latest",
        enable_hybrid: bool = True,  # Enable hybrid search (dense + sparse)
        rrf_alpha: float = 0.5,  # RRF fusion alpha: 0.5 = 50/50 dense/sparse
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
        self.ranking_model = ranking_model
        self.enable_hybrid = enable_hybrid
        self.rrf_alpha = rrf_alpha
        
        # Initialize AI Platform
        aiplatform.init(project=project_id, location=location)
        
        # Load embedding model (for non-task-type queries)
        if not task_type:
            self.embedding_model = TextEmbeddingModel.from_pretrained(embedding_model)
            self.genai_client = None
        else:
            self.embedding_model = None
            # Set up genai environment
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
            os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
            # Cache genai client for reuse (avoid re-init overhead per call)
            if GENAI_AVAILABLE:
                self.genai_client = genai.Client()
            else:
                self.genai_client = None
        
        # Get Vector Search endpoint
        endpoint_resource = f"projects/689311309499/locations/{location}/indexEndpoints/{endpoint_id}"
        self.endpoint = aiplatform.MatchingEngineIndexEndpoint(
            index_endpoint_name=endpoint_resource
        )
        
        # Initialize sparse embedding generator for hybrid search
        if enable_hybrid:
            self.sparse_generator = SimpleSparseEmbedding()
        
        # Initialize Google Ranking API client
        if enable_reranking:
            self.rank_client = discoveryengine.RankServiceClient()
            self.ranking_config = f"projects/{project_id}/locations/global/rankingConfigs/default_ranking_config"
        
        # Load chunk metadata
        self.chunk_index = self._load_chunks()
    
    def _load_chunks(self) -> Dict[str, Dict]:
        """Load chunk data from local file."""
        chunks_file = Path(__file__).parent / "data" / "all_chunks.json"
        if not chunks_file.exists():
            print(f"Warning: {chunks_file} not found")
            return {}
        
        with open(chunks_file, 'r') as f:
            chunks = json.load(f)
        
        index = {}
        for chunk in chunks:
            chunk_id = chunk.get('id', chunk.get('chunk_id', ''))
            if chunk_id:
                index[chunk_id] = chunk
        
        return index
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text."""
        if self.task_type and self.genai_client:
            # Use cached genai client for task_type support
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
            # Gemini model without task_type - still need to control dimensions
            response = self.genai_client.models.embed_content(
                model=self.embedding_model_name,
                contents=[text],
                config=EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",  # Default task type
                    output_dimensionality=self.dimensions,
                ),
            )
            return list(response.embeddings[0].values)
        else:
            # Standard embedding (text-embedding-005)
            embeddings = self.embedding_model.get_embeddings([text])
            return embeddings[0].values
    
    def _rerank(self, query: str, chunks: List[Dict], top_n: int) -> List[Dict]:
        """Rerank chunks using Google Ranking API."""
        if not chunks:
            return []
        
        # Build ranking records
        records = []
        for i, chunk in enumerate(chunks):
            text = chunk.get('text', '')
            if text:
                records.append(discoveryengine.RankingRecord(
                    id=str(i),
                    content=text[:10000],  # API limit
                ))
        
        if not records:
            return chunks[:top_n]
        
        # Call Ranking API
        request = discoveryengine.RankRequest(
            ranking_config=self.ranking_config,
            model=self.ranking_model,
            query=query,
            records=records,
            top_n=top_n,
        )
        
        try:
            response = self.rank_client.rank(request=request)
            
            # Reorder chunks based on ranking
            reranked = []
            for record in response.records:
                idx = int(record.id)
                chunk = chunks[idx].copy()
                chunk['rerank_score'] = record.score
                reranked.append(chunk)
            
            return reranked
        except Exception as e:
            # If reranking fails, fall back to original order
            print(f" [Rerank failed: {str(e)[:30]}]", end="")
            return chunks[:top_n]
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """Retrieve chunks with recall + reranking precision."""
        # Phase 1: RECALL - ALWAYS get recall_k candidates from Vector Search
        # This is the same whether reranking is on or off
        recall_k = self.recall_k  # Always 100 (or whatever recall_k is set to)
        
        # Get query embedding (dense)
        query_embedding = self._get_embedding(query)
        
        # Build query - hybrid or dense-only
        if self.enable_hybrid:
            # Generate sparse embedding for hybrid search
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
                # Fallback to dense-only if sparse fails
                queries = [query_embedding]
        else:
            queries = [query_embedding]
        
        # Query Vector Search
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=queries,
            num_neighbors=recall_k,
        )
        
        # Enrich results with chunk content
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
        
        # Phase 2: PRECISION - rerank to top_k using Google Ranking API
        if self.enable_reranking and results:
            precision_k = top_k if top_k else self.precision_k
            results = self._rerank(query, results, precision_k)
        elif top_k:
            results = results[:top_k]
        
        return results
    
    def retrieve_context(self, query: str, top_k: int = 10) -> str:
        """Retrieve and format context string."""
        results = self.retrieve(query, top_k)
        
        context_parts = []
        for i, r in enumerate(results):
            source = r.get('source_document', 'Unknown')
            text = r.get('text', '')
            context_parts.append(f"[{i+1}] {source}:\n{text}")
        
        return "\n\n---\n\n".join(context_parts)


class EmbeddingComparisonTester:
    """Test RAG quality across different embedding configurations."""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / config.OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize judge LLM
        self.judge_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LLM_LOCATION,
            temperature=0.0,
        )
        
        # Initialize RAG LLM for answer generation
        self.rag_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=config.GCP_PROJECT_ID,
            location=config.GCP_LLM_LOCATION,
            temperature=0.0,
        )
        
        self.max_retries = 5  # Increased for high parallelism with exponential backoff
    
    def load_qa_corpus(self, filename: str = "qa_corpus_200.json") -> List[Dict]:
        """Load the Q&A corpus."""
        qa_path = self.output_dir / filename
        if not qa_path.exists():
            raise FileNotFoundError(f"Q&A corpus not found: {qa_path}")
        
        with open(qa_path, 'r') as f:
            return json.load(f)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate RAG answer from context."""
        prompt = f"""You are a technical assistant. Answer the question based ONLY on the provided context.
Be specific and accurate. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        response = self.rag_llm.invoke(prompt)
        return response.content
    
    def judge_answer(self, question: str, ground_truth: str, rag_answer: str, context: str) -> Dict:
        """Judge the RAG answer against ground truth."""
        prompt = f"""You are an expert evaluator for a RAG system.
Evaluate the RAG answer against the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG System Answer: {rag_answer}

Retrieved Context (for reference):
{context[:2000]}

Rate the RAG answer on these criteria (1-5 scale):
1. Correctness: Does it match the ground truth factually?
2. Completeness: Does it cover all key points from ground truth?
3. Faithfulness: Is it grounded in the retrieved context?
4. Relevance: Does it directly answer the question?
5. Clarity: Is it well-written and clear?

Respond in JSON format:
{{
    "correctness": <1-5>,
    "completeness": <1-5>,
    "faithfulness": <1-5>,
    "relevance": <1-5>,
    "clarity": <1-5>,
    "overall_score": <1-5>,
    "verdict": "pass" | "partial" | "fail",
    "explanation": "<brief explanation>"
}}

Rules: "pass" = overall_score >= 4, "partial" = 3, "fail" <= 2
"""
        response = self.judge_llm.invoke(prompt)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except Exception as e:
            return {"error": str(e), "overall_score": 0, "verdict": "error"}
    
    def _process_question(self, retriever: DirectRetriever, qa: Dict, top_k: int = 10) -> Dict:
        """Process a single question."""
        question = qa.get("question", "")
        ground_truth = qa.get("answer", qa.get("ground_truth", ""))
        
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                
                # Retrieve context
                retrieval_start = time.time()
                context = retriever.retrieve_context(question, top_k)
                retrieval_time = round(time.time() - retrieval_start, 3)
                
                # Generate answer
                gen_start = time.time()
                rag_answer = self._generate_answer(question, context)
                gen_time = round(time.time() - gen_start, 3)
                
                # Judge answer
                judge_start = time.time()
                judgment = self.judge_answer(question, ground_truth, rag_answer, context)
                judge_time = round(time.time() - judge_start, 3)
                
                total_time = round(time.time() - start_time, 3)
                
                return {
                    "question": question,
                    "ground_truth": ground_truth,
                    "rag_answer": rag_answer,
                    "answer_length": len(rag_answer),
                    "judgment": judgment,
                    "attempts": attempt + 1,
                    "timing": {
                        "retrieval_seconds": retrieval_time,
                        "generation_seconds": gen_time,
                        "judge_seconds": judge_time,
                        "total_seconds": total_time,
                    },
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Exponential backoff: 2, 4, 8, 16... seconds
                    backoff = 2 ** (attempt + 1)
                    # Check if rate limited (429) or server error (5xx)
                    err_str = str(e).lower()
                    if "429" in err_str or "rate" in err_str or "quota" in err_str:
                        backoff = min(backoff * 2, 60)  # Double backoff for rate limits, max 60s
                    print(f" [Retry {attempt + 2}/{self.max_retries}, backoff {backoff}s]", end="")
                    time.sleep(backoff)
                else:
                    return {"question": question, "error": str(e), "attempts": self.max_retries}
        
        return {"question": question, "error": "Max retries exceeded"}
    
    def test_config(
        self,
        cfg: Dict,
        qa_corpus: List[Dict],
        max_questions: Optional[int] = None,
        parallel_workers: int = 1,
        top_k: int = 10,
    ) -> Dict:
        """Test a single embedding configuration."""
        print(f"\n{'='*60}")
        print(f"TESTING: {cfg['name']}")
        print(f"  Model: {cfg['embedding_model']}")
        print(f"  Task Type: {cfg['task_type']}")
        print(f"  Endpoint: {cfg['endpoint_id']}")
        print(f"  Workers: {parallel_workers}")
        print(f"{'='*60}")
        
        # Create retriever for this config
        enable_rerank = getattr(self, 'enable_reranking', True)
        recall = getattr(self, 'recall_k', 100)
        alpha = getattr(self, 'rrf_alpha', 0.5)
        
        retriever = DirectRetriever(
            embedding_model=cfg['embedding_model'],
            endpoint_id=cfg['endpoint_id'],
            deployed_index_id=cfg['deployed_index_id'],
            task_type=cfg['task_type'],
            dimensions=cfg['dimensions'],
            recall_k=recall,
            precision_k=top_k,
            enable_reranking=enable_rerank,
            ranking_model="semantic-ranker-default@latest",
            enable_hybrid=True,
            rrf_alpha=alpha,
        )
        print(f"  ✓ Retriever initialized ({len(retriever.chunk_index)} chunks)")
        if enable_rerank:
            print(f"  ✓ Hybrid recall={recall} → rerank → precision={top_k}")
        else:
            print(f"  ✓ Hybrid recall only: top {top_k} (no reranking)")
        
        if max_questions:
            qa_corpus = qa_corpus[:max_questions]
        
        results = []
        start_time = time.time()
        
        # Progress bar
        pbar = tqdm(total=len(qa_corpus), desc=f"  {cfg['name'][:20]}", ncols=80)
        passes = 0
        fails = 0
        
        # Checkpoint save interval
        save_interval = getattr(self, 'save_interval', 10)
        checkpoint_path = self.output_dir / f"checkpoint_{cfg['name']}.json"
        
        if parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(self._process_question, retriever, qa, top_k): i
                    for i, qa in enumerate(qa_corpus)
                }
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    if "judgment" in result:
                        verdict = result["judgment"].get("verdict", "?")
                        if verdict == "pass":
                            passes += 1
                        elif verdict == "fail":
                            fails += 1
                    
                    pbar.update(1)
                    pbar.set_postfix({"pass": passes, "fail": fails})
                    
                    # Save checkpoint every N results
                    if len(results) % save_interval == 0:
                        with open(checkpoint_path, 'w') as f:
                            json.dump({"config": cfg["name"], "completed": len(results), "results": results}, f)
        else:
            for i, qa in enumerate(qa_corpus):
                result = self._process_question(retriever, qa, top_k)
                results.append(result)
                
                if "judgment" in result:
                    verdict = result["judgment"].get("verdict", "?")
                    if verdict == "pass":
                        passes += 1
                    elif verdict == "fail":
                        fails += 1
                
                pbar.update(1)
                pbar.set_postfix({"pass": passes, "fail": fails})
                
                # Save checkpoint every N results
                if len(results) % save_interval == 0:
                    with open(checkpoint_path, 'w') as f:
                        json.dump({"config": cfg["name"], "completed": len(results), "results": results}, f)
        
        pbar.close()
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        valid = [r for r in results if "judgment" in r]
        if valid:
            metrics = {
                "total": len(qa_corpus),
                "successful": len(valid),
                "failed": len(qa_corpus) - len(valid),
                "overall_score": round(sum(r["judgment"]["overall_score"] for r in valid) / len(valid), 2),
                "pass_rate": round(sum(1 for r in valid if r["judgment"]["verdict"] == "pass") / len(valid), 2),
                "correctness": round(sum(r["judgment"].get("correctness", 0) for r in valid) / len(valid), 2),
                "completeness": round(sum(r["judgment"].get("completeness", 0) for r in valid) / len(valid), 2),
                "faithfulness": round(sum(r["judgment"].get("faithfulness", 0) for r in valid) / len(valid), 2),
                "relevance": round(sum(r["judgment"].get("relevance", 0) for r in valid) / len(valid), 2),
                "clarity": round(sum(r["judgment"].get("clarity", 0) for r in valid) / len(valid), 2),
                "avg_time_per_query": round(sum(r["timing"]["total_seconds"] for r in valid) / len(valid), 2),
                "avg_answer_chars": round(sum(r.get("answer_length", 0) for r in valid) / len(valid), 0),
                "total_time": round(elapsed, 2),
            }
        else:
            metrics = {"error": "No valid results"}
        
        return {"config": cfg, "metrics": metrics, "results": results}
    
    def run_comparison(
        self,
        max_questions: Optional[int] = None,
        parallel_workers: int = 8,
        top_k: int = 5,
        configs: Optional[List[Dict]] = None,
        enable_reranking: bool = True,
        recall_k: int = 100,
        rrf_alpha: float = 0.5,
    ) -> Dict:
        """Run comparison across all embedding configurations."""
        configs = configs or EMBEDDING_CONFIGS
        self.enable_reranking = enable_reranking
        self.recall_k = recall_k
        self.rrf_alpha = rrf_alpha
        
        print("="*70)
        mode = "Recall + Rerank" if enable_reranking else "Recall Only"
        print(f"EMBEDDING COMPARISON TEST ({mode})")
        print("="*70)
        print(f"Configurations: {len(configs)}")
        print(f"Workers: {parallel_workers}")
        print(f"Search Type: HYBRID (dense + sparse)")
        print(f"RRF Alpha: {rrf_alpha} ({int(rrf_alpha*100)}% dense / {int((1-rrf_alpha)*100)}% sparse)")
        print(f"Recall: {recall_k} candidates")
        print(f"Precision: top {top_k}")
        print(f"Reranking: {'ON - Google Ranking API' if enable_reranking else 'OFF'}")
        
        # Load Q&A corpus
        qa_corpus = self.load_qa_corpus()
        print(f"✓ Loaded {len(qa_corpus)} Q&A pairs")
        
        if max_questions:
            print(f"  (Testing with {max_questions} questions)")
        
        # Test each configuration
        all_results = {}
        for cfg in configs:
            result = self.test_config(cfg, qa_corpus, max_questions, parallel_workers, top_k)
            all_results[cfg["name"]] = result
            
            # Save intermediate
            self._save_intermediate(cfg["name"], result)
        
        # Generate report
        report = self._generate_report(all_results)
        
        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rerank_suffix = "rerank" if enable_reranking else "recall_only"
        output_path = self.output_dir / f"embedding_comparison_{rerank_suffix}_{timestamp}.json"
        with open(output_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "settings": {
                    "search_type": "HYBRID",
                    "rrf_alpha": rrf_alpha,
                    "alpha_description": f"{int(rrf_alpha*100)}% dense / {int((1-rrf_alpha)*100)}% sparse",
                    "recall_k": recall_k,
                    "precision_k": top_k,
                    "reranking": enable_reranking,
                    "rerank_model": "semantic-ranker-default@latest" if enable_reranking else None,
                    "parallel_workers": parallel_workers,
                    "max_questions": max_questions,
                },
                "comparison": report,
                "detailed_results": all_results,
            }, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
        self._print_report(report)
        
        return report
    
    def _save_intermediate(self, name: str, result: Dict):
        """Save intermediate results."""
        path = self.output_dir / f"embedding_{name}_intermediate.json"
        with open(path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _generate_report(self, all_results: Dict) -> Dict:
        """Generate comparison report."""
        report = {"by_config": {}, "winner": None}
        
        best_score = 0
        best_config = None
        
        for name, data in all_results.items():
            metrics = data.get("metrics", {})
            report["by_config"][name] = {
                "description": data["config"]["description"],
                "embedding_model": data["config"]["embedding_model"],
                "task_type": data["config"]["task_type"],
                "overall_score": metrics.get("overall_score", 0),
                "pass_rate": metrics.get("pass_rate", 0),
                "correctness": metrics.get("correctness", 0),
                "completeness": metrics.get("completeness", 0),
                "faithfulness": metrics.get("faithfulness", 0),
                "relevance": metrics.get("relevance", 0),
                "clarity": metrics.get("clarity", 0),
                "avg_time": metrics.get("avg_time", 0),
            }
            
            score = metrics.get("overall_score", 0)
            if score > best_score:
                best_score = score
                best_config = name
        
        report["winner"] = {"name": best_config, "score": best_score}
        return report
    
    def _print_report(self, report: Dict):
        """Print comparison report."""
        print("\n" + "="*90)
        print("EMBEDDING COMPARISON RESULTS")
        print("="*90)
        
        print(f"{'Config':<25} {'Score':<8} {'Pass%':<8} {'Correct':<8} {'Complete':<8} {'Faithful':<8} {'Time':<8}")
        print("-"*90)
        
        for name, data in report["by_config"].items():
            print(f"{name:<25} ", end="")
            print(f"{data['overall_score']:<8.2f} ", end="")
            print(f"{data['pass_rate']*100:<8.0f} ", end="")
            print(f"{data['correctness']:<8.2f} ", end="")
            print(f"{data['completeness']:<8.2f} ", end="")
            print(f"{data['faithfulness']:<8.2f} ", end="")
            print(f"{data['avg_time']:<8.2f}")
        
        print("-"*90)
        
        winner = report.get("winner", {})
        print(f"\n🏆 WINNER: {winner.get('name')} (score: {winner.get('score', 0):.2f})")
        print("="*90)


def main():
    """Run the embedding comparison test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare embedding configs (direct retrieval)")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--max-questions", "-n", type=int, default=None, help="Limit questions")
    parser.add_argument("--top-k", "-k", type=int, default=12, help="Top-K chunks to pass to LLM (default: 12)")
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranking (recall only)")
    parser.add_argument("--recall-k", type=int, default=100, help="Recall candidates (default: 100)")
    parser.add_argument("--alpha", "-a", type=float, default=1.0, help="RRF alpha: 1.0=100%% dense, 0.5=50/50, 0.0=100%% sparse")
    parser.add_argument("--save-interval", "-s", type=int, default=10, help="Checkpoint save interval (default: 10)")
    parser.add_argument("--config", "-c", type=str, default=None, help="Run only specific config by name (e.g., gemini-RETRIEVAL_QUERY)")
    args = parser.parse_args()
    
    # Filter configs if specified
    configs = None
    if args.config:
        configs = [c for c in EMBEDDING_CONFIGS if c["name"] == args.config]
        if not configs:
            print(f"Error: Config '{args.config}' not found. Available: {[c['name'] for c in EMBEDDING_CONFIGS]}")
            return
    
    tester = EmbeddingComparisonTester()
    tester.save_interval = args.save_interval  # Set checkpoint interval
    tester.run_comparison(
        max_questions=args.max_questions,
        parallel_workers=args.workers,
        top_k=args.top_k,  # Same top_k for both modes
        enable_reranking=not args.no_rerank,
        recall_k=args.recall_k,
        rrf_alpha=args.alpha,
        configs=configs,
    )


if __name__ == "__main__":
    main()
