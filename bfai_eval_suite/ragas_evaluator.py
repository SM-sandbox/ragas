"""Run Ragas evaluation on filtered questions"""
import json
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from config import config
from rag_retriever import RAGRetriever
from llm_client import LLMClient

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from datasets import Dataset
import litellm
import os


class RagasEvaluator:
    """Run Ragas evaluation on the filtered question set"""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / config.OUTPUT_DIR
        self.retriever = RAGRetriever()
        self.llm_client = LLMClient()
        
        # Set up Vertex AI for Ragas
        os.environ["VERTEXAI_PROJECT"] = config.GCP_PROJECT_ID
        os.environ["VERTEXAI_LOCATION"] = config.GCP_LOCATION
        
        # Create Ragas LLM using litellm
        self.ragas_llm = llm_factory(
            f"vertex_ai/{config.LLM_MODEL}",
            provider="litellm",
            client=litellm.completion,
        )
        
        # Create Ragas embeddings
        self.ragas_embeddings = embedding_factory(
            "litellm",
            model=f"vertex_ai/{config.EMBEDDING_MODEL}",
        )
    
    def load_filtered_questions(self) -> Dict[str, List[Dict]]:
        """Load filtered high-quality questions"""
        filtered_file = self.output_dir / "filtered_questions.json"
        if not filtered_file.exists():
            raise FileNotFoundError(f"Filtered questions not found: {filtered_file}. Run question_rater.py first.")
        
        with open(filtered_file, 'r') as f:
            return json.load(f)
    
    def generate_rag_response(self, question: str, context: str) -> str:
        """Generate a RAG response using the LLM"""
        prompt = f"""You are a helpful technical assistant for SCADA/Solar/Electrical equipment.
Answer the following question based ONLY on the provided context. If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm_client.generate(prompt)
    
    def prepare_evaluation_dataset(self, questions: Dict[str, List[Dict]]) -> Dataset:
        """Prepare dataset for Ragas evaluation"""
        eval_data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': [],
        }
        
        all_questions = questions.get('single_hop', []) + questions.get('multi_hop', [])
        
        print(f"Preparing evaluation dataset with {len(all_questions)} questions...")
        
        for i, q in enumerate(all_questions):
            question_text = q.get('question', '')
            ground_truth = q.get('ground_truth_answer', '')
            
            # Retrieve context
            context = self.retriever.retrieve_context(question_text, top_k=5)
            contexts = [context]  # Ragas expects list of contexts
            
            # Generate RAG response
            answer = self.generate_rag_response(question_text, context)
            
            eval_data['question'].append(question_text)
            eval_data['answer'].append(answer)
            eval_data['contexts'].append(contexts)
            eval_data['ground_truth'].append(ground_truth)
            
            print(f"  Processed question {i+1}/{len(all_questions)}")
        
        return Dataset.from_dict(eval_data)
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run the full Ragas evaluation"""
        # Load filtered questions
        questions = self.load_filtered_questions()
        total_questions = len(questions.get('single_hop', [])) + len(questions.get('multi_hop', []))
        
        if total_questions == 0:
            print("No questions to evaluate!")
            return {}
        
        print(f"\n=== Starting Ragas Evaluation ===")
        print(f"Total questions: {total_questions}")
        print(f"  Single-hop: {len(questions.get('single_hop', []))}")
        print(f"  Multi-hop: {len(questions.get('multi_hop', []))}")
        
        # Prepare dataset
        dataset = self.prepare_evaluation_dataset(questions)
        
        # Define metrics
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
        # Run evaluation
        print("\nRunning Ragas evaluation...")
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
        )
        
        # Convert to dict
        results_dict = {
            'timestamp': datetime.now().isoformat(),
            'num_questions': total_questions,
            'num_single_hop': len(questions.get('single_hop', [])),
            'num_multi_hop': len(questions.get('multi_hop', [])),
            'metrics': {
                'faithfulness': float(results['faithfulness']),
                'answer_relevancy': float(results['answer_relevancy']),
                'context_precision': float(results['context_precision']),
                'context_recall': float(results['context_recall']),
            },
            'per_question_results': results.to_pandas().to_dict('records') if hasattr(results, 'to_pandas') else []
        }
        
        # Save results
        results_file = self.output_dir / "ragas_evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        
        # Print summary
        print("\n" + "="*50)
        print("RAGAS EVALUATION RESULTS")
        print("="*50)
        print(f"Questions Evaluated: {total_questions}")
        print(f"  Single-hop: {results_dict['num_single_hop']}")
        print(f"  Multi-hop: {results_dict['num_multi_hop']}")
        print("\nMetrics:")
        print(f"  Faithfulness:       {results_dict['metrics']['faithfulness']:.4f}")
        print(f"  Answer Relevancy:   {results_dict['metrics']['answer_relevancy']:.4f}")
        print(f"  Context Precision:  {results_dict['metrics']['context_precision']:.4f}")
        print(f"  Context Recall:     {results_dict['metrics']['context_recall']:.4f}")
        print("="*50)
        print(f"\n✓ Full results saved to: {results_file}")
        
        return results_dict


if __name__ == "__main__":
    evaluator = RagasEvaluator()
    results = evaluator.run_evaluation()
