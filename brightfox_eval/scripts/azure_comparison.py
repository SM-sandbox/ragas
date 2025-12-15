#!/usr/bin/env python3
"""
Azure RAG Evaluation Script
Mirrors the GCP embedding_comparison_direct.py methodology but uses Azure AI Search + Azure OpenAI.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from openai import AzureOpenAI
from langchain_google_vertexai import ChatVertexAI

# Azure Configuration - load from environment variables
AZURE_SEARCH_ENDPOINT = os.environ.get("AZURE_SEARCH_ENDPOINT", "https://asosearch-stg.search.windows.net")
AZURE_SEARCH_KEY = os.environ.get("AZURE_SEARCH_KEY", "")
AZURE_SEARCH_INDEX = os.environ.get("AZURE_SEARCH_INDEX", "bf-demo")

AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://sponsored-eastus-oai.openai.azure.com/")
AZURE_OPENAI_KEY = os.environ.get("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")
AZURE_EMBEDDING_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "eastus-text-embedding-3-large")
AZURE_API_VERSION = "2024-06-01"

# Gemini for judging (same as GCP tests)
GCP_PROJECT = os.environ.get("GCP_PROJECT", "civic-athlete-473921-c0")


class AzureRetriever:
    """Azure AI Search retriever with vector search."""
    
    def __init__(self, top_k: int = 12):
        self.search_endpoint = AZURE_SEARCH_ENDPOINT
        self.search_key = AZURE_SEARCH_KEY
        self.index_name = AZURE_SEARCH_INDEX
        self.top_k = top_k
        
        # Azure OpenAI client for embeddings
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        # Get document count
        import requests
        count_url = f"{self.search_endpoint}/indexes/{self.index_name}/docs/$count?api-version=2023-11-01"
        resp = requests.get(count_url, headers={"api-key": self.search_key})
        self.doc_count = int(resp.text) if resp.ok else 0
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using Azure OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=AZURE_EMBEDDING_DEPLOYMENT
        )
        return response.data[0].embedding
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """Retrieve documents using Azure AI Search vector search."""
        import requests
        
        k = top_k or self.top_k
        
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Vector search request
        search_url = f"{self.search_endpoint}/indexes/{self.index_name}/docs/search?api-version=2023-11-01"
        
        search_body = {
            "vectorQueries": [{
                "kind": "vector",
                "vector": query_embedding,
                "fields": "embedding3",
                "k": k
            }],
            "select": "id,content,sourcefile,sourcepage",
            "top": k
        }
        
        headers = {
            "api-key": self.search_key,
            "Content-Type": "application/json"
        }
        
        response = requests.post(search_url, headers=headers, json=search_body)
        
        if not response.ok:
            raise Exception(f"Search failed: {response.text}")
        
        results = response.json().get("value", [])
        
        return [{
            "content": r.get("content", ""),
            "source": r.get("sourcefile", ""),
            "page": r.get("sourcepage", ""),
            "score": r.get("@search.score", 0)
        } for r in results]


class AzureRAGTester:
    """Azure RAG evaluation tester."""
    
    def __init__(self):
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load Q&A corpus (same as GCP tests)
        self.qa_corpus = self._load_qa_corpus()
        
        # Azure OpenAI client for generation and judging
        self.openai_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        # For judging, we'll use the same Gemini judge as GCP for fair comparison
        # Or use Azure OpenAI - let's use Azure for full Azure comparison
        self.save_interval = 10
    
    def _load_qa_corpus(self) -> List[Dict]:
        """Load Q&A pairs from the same corpus used in GCP tests."""
        qa_file = Path("output/qa_corpus_200.json")
        if not qa_file.exists():
            qa_file = Path("qa_pairs_with_answers.json")
        if not qa_file.exists():
            qa_file = Path("qa_pairs.json")
        
        if not qa_file.exists():
            raise FileNotFoundError("Q&A corpus not found")
        
        with open(qa_file) as f:
            data = json.load(f)
        
        if isinstance(data, dict) and "qa_pairs" in data:
            return data["qa_pairs"]
        return data
    
    def generate_answer(self, question: str, context: str, max_retries: int = 3) -> str:
        """Generate answer using Azure OpenAI with retry logic."""
        prompt = f"""You are a helpful assistant answering questions based on the provided context.

Context:
{context}

Question: {question}

Answer the question based only on the provided context. If the context doesn't contain enough information, say so."""

        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                if content and content.strip():
                    return content
                # Empty response, retry
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise e
        
        return "Failed to generate answer after retries"
    
    def judge_answer(self, question: str, ground_truth: str, rag_answer: str, context: str) -> Dict:
        """Judge the RAG answer using Gemini 2.5 Flash (same as GCP tests)."""
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
        # Use Gemini 2.5 Flash for judging (same as GCP)
        judge_llm = ChatVertexAI(
            model_name="gemini-2.5-flash",
            project=GCP_PROJECT,
            location="us-east1",
            temperature=0,
        )
        response = judge_llm.invoke(prompt)
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())
        except Exception as e:
            return {"error": str(e), "overall_score": 0, "verdict": "error"}
    
    def _process_question(self, retriever: AzureRetriever, qa: Dict, top_k: int = 12) -> Dict:
        """Process a single question."""
        start_time = time.time()
        
        question = qa.get("question", "")
        ground_truth = qa.get("answer", qa.get("ground_truth", ""))
        
        result = {
            "question": question,
            "ground_truth": ground_truth,
            "timing": {}
        }
        
        try:
            # Retrieval
            retrieval_start = time.time()
            chunks = retriever.retrieve(question, top_k=top_k)
            result["timing"]["retrieval_seconds"] = time.time() - retrieval_start
            result["chunks_retrieved"] = len(chunks)
            
            # Build context
            context = "\n\n---\n\n".join([
                f"[Source: {c.get('source', 'unknown')}]\n{c.get('content', '')}"
                for c in chunks
            ])
            result["context_length"] = len(context)
            
            # Generation
            gen_start = time.time()
            answer = self.generate_answer(question, context)
            result["timing"]["generation_seconds"] = time.time() - gen_start
            result["answer"] = answer
            result["answer_length"] = len(answer)
            
            # Judging
            judge_start = time.time()
            judgment = self.judge_answer(question, ground_truth, answer, context)
            result["timing"]["judge_seconds"] = time.time() - judge_start
            result["judgment"] = judgment
            
            result["timing"]["total_seconds"] = time.time() - start_time
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            result["success"] = False
            result["timing"]["total_seconds"] = time.time() - start_time
        
        return result
    
    def run_evaluation(
        self,
        max_questions: Optional[int] = None,
        parallel_workers: int = 8,
        top_k: int = 12
    ):
        """Run the Azure RAG evaluation."""
        
        qa_corpus = self.qa_corpus[:max_questions] if max_questions else self.qa_corpus
        
        print("=" * 70)
        print("AZURE RAG EVALUATION")
        print("=" * 70)
        print(f"Search Index: {AZURE_SEARCH_INDEX}")
        print(f"LLM Model: {AZURE_OPENAI_DEPLOYMENT}")
        print(f"Embedding Model: {AZURE_EMBEDDING_DEPLOYMENT}")
        print(f"Workers: {parallel_workers}")
        print(f"Top-K: {top_k}")
        print(f"Questions: {len(qa_corpus)}")
        print("=" * 70)
        
        # Initialize retriever
        retriever = AzureRetriever(top_k=top_k)
        print(f"✓ Retriever initialized ({retriever.doc_count} chunks)")
        
        results = []
        checkpoint_path = self.output_dir / "checkpoint_azure.json"
        
        if parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                futures = {
                    executor.submit(self._process_question, retriever, qa, top_k): i
                    for i, qa in enumerate(qa_corpus)
                }
                
                pbar = tqdm(
                    total=len(qa_corpus),
                    desc="Azure RAG",
                    ncols=80
                )
                
                pass_count = 0
                fail_count = 0
                
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    if result.get("success") and result.get("judgment", {}).get("verdict") == "pass":
                        pass_count += 1
                    elif result.get("success"):
                        fail_count += 1
                    
                    pbar.set_postfix({"pass": pass_count, "fail": fail_count})
                    pbar.update(1)
                    
                    # Checkpoint save
                    if len(results) % self.save_interval == 0:
                        with open(checkpoint_path, 'w') as f:
                            json.dump({"completed": len(results), "results": results}, f)
                
                pbar.close()
        else:
            for qa in tqdm(qa_corpus, desc="Azure RAG"):
                result = self._process_question(retriever, qa, top_k)
                results.append(result)
        
        # Calculate metrics
        valid = [r for r in results if r.get("success") and "judgment" in r]
        
        metrics = {
            "total": len(qa_corpus),
            "successful": len(valid),
            "failed": len(qa_corpus) - len(valid),
            "overall_score": round(sum(r["judgment"]["overall_score"] for r in valid) / len(valid), 2) if valid else 0,
            "pass_rate": round(sum(1 for r in valid if r["judgment"]["verdict"] == "pass") / len(valid), 2) if valid else 0,
            "correctness": round(sum(r["judgment"].get("correctness", 0) for r in valid) / len(valid), 2) if valid else 0,
            "completeness": round(sum(r["judgment"].get("completeness", 0) for r in valid) / len(valid), 2) if valid else 0,
            "faithfulness": round(sum(r["judgment"].get("faithfulness", 0) for r in valid) / len(valid), 2) if valid else 0,
            "relevance": round(sum(r["judgment"].get("relevance", 0) for r in valid) / len(valid), 2) if valid else 0,
            "clarity": round(sum(r["judgment"].get("clarity", 0) for r in valid) / len(valid), 2) if valid else 0,
            "avg_time_per_query": round(sum(r["timing"]["total_seconds"] for r in valid) / len(valid), 2) if valid else 0,
            "avg_answer_chars": round(sum(r.get("answer_length", 0) for r in valid) / len(valid), 0) if valid else 0,
        }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"azure_evaluation_{timestamp}.json"
        
        output_data = {
            "timestamp": timestamp,
            "config": {
                "search_index": AZURE_SEARCH_INDEX,
                "llm_model": AZURE_OPENAI_DEPLOYMENT,
                "embedding_model": AZURE_EMBEDDING_DEPLOYMENT,
                "top_k": top_k,
                "workers": parallel_workers,
            },
            "metrics": metrics,
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Print report
        print("\n" + "=" * 70)
        print("AZURE RAG EVALUATION RESULTS")
        print("=" * 70)
        print(f"Score: {metrics['overall_score']:.2f}  Pass%: {metrics['pass_rate']*100:.0f}%")
        print(f"Correct: {metrics['correctness']:.2f}  Complete: {metrics['completeness']:.2f}  Faithful: {metrics['faithfulness']:.2f}")
        print(f"Relevance: {metrics['relevance']:.2f}  Clarity: {metrics['clarity']:.2f}")
        print(f"Avg Time: {metrics['avg_time_per_query']:.2f}s  Avg Answer: {metrics['avg_answer_chars']:.0f} chars")
        print("=" * 70)
        
        return metrics


def main():
    """Run the Azure RAG evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Azure RAG Evaluation")
    parser.add_argument("--workers", "-w", type=int, default=8, help="Parallel workers (default: 8)")
    parser.add_argument("--max-questions", "-n", type=int, default=None, help="Limit questions")
    parser.add_argument("--top-k", "-k", type=int, default=12, help="Top-K chunks (default: 12)")
    parser.add_argument("--save-interval", "-s", type=int, default=10, help="Checkpoint interval (default: 10)")
    parser.add_argument("--filter-missing", action="store_true", help="Exclude questions from docs missing in Azure")
    args = parser.parse_args()
    
    tester = AzureRAGTester()
    tester.save_interval = args.save_interval
    
    # Filter out questions from documents missing in Azure
    if args.filter_missing:
        missing_docs = {
            '1-SO50016-BOYNE-MOUNTAIN-RESORT_SLD.0.12C.ADD-ON.SO87932.IFC',
            'AcquiSuite-Basics---External',
            'Acquisuite Backup and Restore',
            'DiGiWR21WR31ModemBasicsv1',
            'EPEC_1200-6000A UL 891 SWBD (Crit Power) Flyer',
            'Manual-PVI-Central-250-300',
            'PFMG - OM Agreement (WSD)_Executed',
            'PVI-6000-OUTD-US Service Manual',
        }
        original_count = len(tester.qa_corpus)
        tester.qa_corpus = [q for q in tester.qa_corpus if q.get('source_document') not in missing_docs]
        print(f"Filtered: {original_count} -> {len(tester.qa_corpus)} questions (excluded {original_count - len(tester.qa_corpus)} from missing docs)")
    
    tester.run_evaluation(
        max_questions=args.max_questions,
        parallel_workers=args.workers,
        top_k=args.top_k
    )


if __name__ == "__main__":
    main()
