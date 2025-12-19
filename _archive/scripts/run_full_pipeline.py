#!/usr/bin/env python3
"""
BrightFox RAG Evaluation Pipeline
==================================
Complete end-to-end evaluation pipeline:
1. Download chunks from GCS
2. Generate 25 single-hop + 25 multi-hop questions
3. Rate questions with LLM (1-5 scale)
4. Filter to keep only 4s and 5s
5. Run Ragas evaluation on filtered questions
6. Output final results
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import config


def run_pipeline():
    """Run the complete evaluation pipeline"""
    print("="*60)
    print("BRIGHTFOX RAG EVALUATION PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now().isoformat()}")
    print(f"Project: {config.GCP_PROJECT_ID}")
    print(f"Location: {config.GCP_LOCATION}")
    print(f"Vector Search Endpoint: {config.VECTOR_SEARCH_ENDPOINT_ID}")
    print("="*60)
    
    # Step 1: Download chunks
    print("\n[STEP 1/5] Downloading chunks from GCS...")
    print("-"*40)
    from scripts.download_chunks import download_chunks
    chunks = download_chunks()
    print(f"✓ Downloaded {len(chunks)} chunks")
    
    # Step 2: Generate questions
    print("\n[STEP 2/5] Generating evaluation questions...")
    print("-"*40)
    from question_generator import QuestionGenerator
    generator = QuestionGenerator()
    questions = generator.generate_all_questions()
    print(f"✓ Generated {len(questions['single_hop'])} single-hop questions")
    print(f"✓ Generated {len(questions['multi_hop'])} multi-hop questions")
    
    # Step 3: Rate questions
    print("\n[STEP 3/5] Rating questions for quality...")
    print("-"*40)
    from question_rater import QuestionRater
    rater = QuestionRater()
    rated = rater.rate_all_questions()
    
    # Step 4: Filter high-quality questions
    print("\n[STEP 4/5] Filtering high-quality questions (score >= 4)...")
    print("-"*40)
    filtered = rater.filter_high_quality(min_score=4)
    total_filtered = len(filtered['single_hop']) + len(filtered['multi_hop'])
    print(f"✓ Kept {total_filtered} high-quality questions")
    
    # Step 5: Run Ragas evaluation
    print("\n[STEP 5/5] Running Ragas evaluation...")
    print("-"*40)
    from ragas_evaluator import RagasEvaluator
    evaluator = RagasEvaluator()
    results = evaluator.run_evaluation()
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Completed at: {datetime.now().isoformat()}")
    print(f"\nOutput files in: {Path(__file__).parent / config.OUTPUT_DIR}")
    print("  - generated_questions.json")
    print("  - rated_questions.json")
    print("  - filtered_questions.json")
    print("  - discarded_questions.json")
    print("  - ragas_evaluation_results.json")
    print("="*60)
    
    return results


if __name__ == "__main__":
    run_pipeline()
