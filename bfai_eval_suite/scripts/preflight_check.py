#!/usr/bin/env python3
"""
Pre-flight check for Gold Standard Evaluation.
Verifies all dependencies and access before running full eval.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, "/Users/scottmacon/Documents/GitHub/sm-dev-01")

def check_mark(success: bool) -> str:
    return "✅" if success else "❌"

def preflight():
    print("=" * 60)
    print("PRE-FLIGHT CHECK - Gold Standard Evaluation")
    print("=" * 60)
    
    all_passed = True
    
    # 1. Check ADC (Application Default Credentials)
    print("\n1. Checking GCP ADC...")
    try:
        import google.auth
        credentials, project = google.auth.default()
        print(f"   {check_mark(True)} ADC configured")
        print(f"   Project: {project}")
    except Exception as e:
        print(f"   {check_mark(False)} ADC FAILED: {e}")
        print("   Run: gcloud auth application-default login")
        all_passed = False
    
    # 2. Check orchestrator imports
    print("\n2. Checking orchestrator imports...")
    try:
        from libs.core.gcp_config import get_jobs_config, PROJECT_ID, LOCATION
        print(f"   {check_mark(True)} gcp_config loaded")
        print(f"   PROJECT_ID: {PROJECT_ID}")
        print(f"   LOCATION: {LOCATION}")
    except Exception as e:
        print(f"   {check_mark(False)} gcp_config FAILED: {e}")
        all_passed = False
        return False
    
    # 3. Check job config
    print("\n3. Checking job config...")
    JOB_ID = "bfai__eval66a_g1_1536_tt"
    try:
        jobs = get_jobs_config()
        job_config = jobs.get(JOB_ID, {})
        job_config["job_id"] = JOB_ID
        print(f"   {check_mark(True)} Job config loaded: {JOB_ID}")
    except Exception as e:
        print(f"   {check_mark(False)} Job config FAILED: {e}")
        all_passed = False
        return False
    
    # 4. Check retriever
    print("\n4. Checking VectorSearchRetriever...")
    try:
        from services.api.retrieval.vector_search import VectorSearchRetriever
        from services.api.core.config import QueryConfig
        retriever = VectorSearchRetriever(job_config)
        
        config = QueryConfig(recall_top_k=5, precision_top_n=3, enable_hybrid=True, job_id=JOB_ID)
        result = retriever.retrieve("test query voltage rating", config)
        chunks = list(result.chunks)
        print(f"   {check_mark(True)} Retriever works - got {len(chunks)} chunks")
    except Exception as e:
        print(f"   {check_mark(False)} Retriever FAILED: {e}")
        all_passed = False
    
    # 5. Check ranker
    print("\n5. Checking GoogleRanker...")
    try:
        from services.api.ranking.google_ranker import GoogleRanker
        ranker = GoogleRanker(project_id=PROJECT_ID)
        if chunks:
            reranked = ranker.rank("test query", chunks, 3)
            print(f"   {check_mark(True)} Ranker works - reranked to {len(reranked)} chunks")
        else:
            print(f"   {check_mark(False)} Ranker skipped - no chunks to rerank")
            all_passed = False
    except Exception as e:
        print(f"   {check_mark(False)} Ranker FAILED: {e}")
        all_passed = False
    
    # 6. Check generator
    print("\n6. Checking GeminiAnswerGenerator...")
    try:
        from services.api.generation.gemini import GeminiAnswerGenerator
        generator = GeminiAnswerGenerator()
        context = "The voltage rating is 480V AC."
        gen_result = generator.generate(
            query="What is the voltage rating?",
            context=context,
            config=config,
        )
        print(f"   {check_mark(True)} Generator works")
        print(f"   Model: gemini-2.5-flash (default)")
        print(f"   Answer preview: {gen_result.answer_text[:80]}...")
    except Exception as e:
        print(f"   {check_mark(False)} Generator FAILED: {e}")
        all_passed = False
    
    # 7. Check LLM Judge
    print("\n7. Checking LLM Judge (ChatVertexAI)...")
    try:
        from langchain_google_vertexai import ChatVertexAI
        judge = ChatVertexAI(
            model_name="gemini-2.0-flash",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0.0,
        )
        response = judge.invoke("Say 'OK' if you can hear me.")
        print(f"   {check_mark(True)} Judge works")
        print(f"   Model: gemini-2.0-flash")
        print(f"   Response: {response.content[:50]}...")
    except Exception as e:
        print(f"   {check_mark(False)} Judge FAILED: {e}")
        all_passed = False
    
    # 8. Check corpus
    print("\n8. Checking corpus...")
    corpus_path = Path(__file__).parent.parent / "corpus" / "qa_corpus_gold_500.json"
    try:
        with open(corpus_path) as f:
            data = json.load(f)
        questions = data.get("questions", data)
        print(f"   {check_mark(True)} Corpus loaded: {len(questions)} questions")
    except Exception as e:
        print(f"   {check_mark(False)} Corpus FAILED: {e}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to run evaluation")
    else:
        print("❌ SOME CHECKS FAILED - Fix issues before running")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = preflight()
    sys.exit(0 if success else 1)
