#!/usr/bin/env python3
"""
LLM-as-Judge evaluation for the Q&A corpus.
Evaluates answer quality, faithfulness, and relevance using Gemini 3 Flash.

Updated: Dec 2025 - Migrated to google-genai SDK
"""
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from config import config
from rag_retriever import RAGRetriever
from gemini_client import generate_for_judge, generate_for_rag, get_model_info


class LLMJudge:
    """LLM-as-Judge evaluator for Q&A corpus"""
    
    def __init__(self):
        self.output_dir = Path(__file__).parent / config.OUTPUT_DIR
        self.retriever = RAGRetriever()
        
        # Model info for logging
        model_info = get_model_info()
        print(f"Using model: {model_info['model_id']} ({model_info['status']})")
    
    def load_qa_corpus(self, filename: str = "qa_corpus_200.json") -> list[dict]:
        """Load the Q&A corpus"""
        qa_path = self.output_dir / filename
        if not qa_path.exists():
            raise FileNotFoundError(f"Q&A corpus not found: {qa_path}")
        
        with open(qa_path, 'r') as f:
            return json.load(f)
    
    def generate_rag_answer(self, question: str) -> tuple[str, str]:
        """Generate RAG answer for a question"""
        # Retrieve context
        context = self.retriever.retrieve_context(question, top_k=5)
        
        # Generate answer
        prompt = f"""You are a technical assistant for SCADA/Solar/Electrical equipment.
Answer the question based ONLY on the provided context. Be specific and accurate.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        response = generate_for_rag(prompt)
        return response, context
    
    def judge_answer(self, question: str, ground_truth: str, rag_answer: str, context: str) -> dict:
        """Use LLM to judge the RAG answer quality"""
        
        judge_prompt = f"""You are an expert evaluator for a RAG (Retrieval Augmented Generation) system.
Evaluate the RAG system's answer against the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth}

RAG System Answer: {rag_answer}

Retrieved Context (for reference):
{context[:2000]}

Evaluate on these criteria (score 1-5, where 5 is best):

1. **Correctness**: Is the RAG answer factually correct compared to ground truth?
2. **Completeness**: Does the RAG answer cover all key points from ground truth?
3. **Faithfulness**: Is the RAG answer faithful to the retrieved context (no hallucinations)?
4. **Relevance**: Is the RAG answer relevant to the question asked?
5. **Clarity**: Is the RAG answer clear and well-structured?

Respond with JSON in this exact format:
{{
    "correctness": <1-5>,
    "completeness": <1-5>,
    "faithfulness": <1-5>,
    "relevance": <1-5>,
    "clarity": <1-5>,
    "overall_score": <1-5>,
    "verdict": "pass|partial|fail",
    "explanation": "Brief explanation of the evaluation"
}}
"""
        
        try:
            # Use gemini_client for structured JSON output
            result = generate_for_judge(judge_prompt)
            return result
        except Exception as e:
            return {
                "correctness": 0,
                "completeness": 0,
                "faithfulness": 0,
                "relevance": 0,
                "clarity": 0,
                "overall_score": 0,
                "verdict": "error",
                "explanation": f"Evaluation error: {str(e)}"
            }
    
    def run_evaluation(self, max_questions: Optional[int] = None) -> dict:
        """Run full LLM-as-judge evaluation"""
        print("="*60)
        print("LLM-AS-JUDGE EVALUATION")
        print("="*60)
        
        # Load Q&A corpus
        print("\n[1/4] Loading Q&A corpus...")
        qa_corpus = self.load_qa_corpus()
        
        if max_questions:
            qa_corpus = qa_corpus[:max_questions]
        
        print(f"✓ Loaded {len(qa_corpus)} Q&A pairs")
        
        # Separate by type
        single_hop = [q for q in qa_corpus if q.get("question_type") == "single_hop"]
        multi_hop = [q for q in qa_corpus if q.get("question_type") == "multi_hop"]
        print(f"  Single-hop: {len(single_hop)}")
        print(f"  Multi-hop: {len(multi_hop)}")
        
        # Run evaluation
        print("\n[2/4] Generating RAG answers...")
        results = []
        
        for i, qa in enumerate(qa_corpus):
            question = qa.get("question", "")
            ground_truth = qa.get("answer", qa.get("ground_truth", ""))
            
            print(f"  Processing {i+1}/{len(qa_corpus)}: {question[:50]}...")
            
            try:
                # Generate RAG answer
                rag_answer, context = self.generate_rag_answer(question)
                
                # Judge the answer
                judgment = self.judge_answer(question, ground_truth, rag_answer, context)
                
                result = {
                    "id": qa.get("id", i+1),
                    "question": question,
                    "question_type": qa.get("question_type", "unknown"),
                    "ground_truth": ground_truth,
                    "rag_answer": rag_answer,
                    "judgment": judgment,
                }
                results.append(result)
                
                # Print progress
                verdict = judgment.get("verdict", "unknown")
                score = judgment.get("overall_score", 0)
                print(f"    → Score: {score}/5, Verdict: {verdict}")
                
            except Exception as e:
                print(f"    → Error: {e}")
                results.append({
                    "id": qa.get("id", i+1),
                    "question": question,
                    "question_type": qa.get("question_type", "unknown"),
                    "error": str(e),
                })
        
        # Calculate aggregate metrics
        print("\n[3/4] Calculating metrics...")
        metrics = self._calculate_metrics(results)
        
        # Save results
        print("\n[4/4] Saving results...")
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(qa_corpus),
            "single_hop_count": len(single_hop),
            "multi_hop_count": len(multi_hop),
            "metrics": metrics,
            "detailed_results": results,
        }
        
        output_path = self.output_dir / "llm_judge_results.json"
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Print summary
        self._print_summary(metrics, len(qa_corpus), len(single_hop), len(multi_hop))
        
        print(f"\n✓ Full results saved to: {output_path}")
        
        return evaluation_results
    
    def _calculate_metrics(self, results: list[dict]) -> dict:
        """Calculate aggregate metrics from results"""
        valid_results = [r for r in results if "judgment" in r]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        # Overall metrics
        metrics = {
            "correctness": {"mean": 0, "single_hop": 0, "multi_hop": 0},
            "completeness": {"mean": 0, "single_hop": 0, "multi_hop": 0},
            "faithfulness": {"mean": 0, "single_hop": 0, "multi_hop": 0},
            "relevance": {"mean": 0, "single_hop": 0, "multi_hop": 0},
            "clarity": {"mean": 0, "single_hop": 0, "multi_hop": 0},
            "overall_score": {"mean": 0, "single_hop": 0, "multi_hop": 0},
            "verdict_counts": {"pass": 0, "partial": 0, "fail": 0, "error": 0},
            "pass_rate": 0,
        }
        
        single_hop_results = [r for r in valid_results if r.get("question_type") == "single_hop"]
        multi_hop_results = [r for r in valid_results if r.get("question_type") == "multi_hop"]
        
        for metric in ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]:
            # Overall mean
            values = [r["judgment"].get(metric, 0) for r in valid_results]
            metrics[metric]["mean"] = sum(values) / len(values) if values else 0
            
            # Single-hop mean
            sh_values = [r["judgment"].get(metric, 0) for r in single_hop_results]
            metrics[metric]["single_hop"] = sum(sh_values) / len(sh_values) if sh_values else 0
            
            # Multi-hop mean
            mh_values = [r["judgment"].get(metric, 0) for r in multi_hop_results]
            metrics[metric]["multi_hop"] = sum(mh_values) / len(mh_values) if mh_values else 0
        
        # Verdict counts
        for r in valid_results:
            verdict = r["judgment"].get("verdict", "error")
            if verdict in metrics["verdict_counts"]:
                metrics["verdict_counts"][verdict] += 1
            else:
                metrics["verdict_counts"]["error"] += 1
        
        # Pass rate
        total = len(valid_results)
        passed = metrics["verdict_counts"]["pass"] + metrics["verdict_counts"]["partial"]
        metrics["pass_rate"] = passed / total if total > 0 else 0
        
        return metrics
    
    def _print_summary(self, metrics: dict, total: int, single_hop: int, multi_hop: int):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("LLM-AS-JUDGE EVALUATION RESULTS")
        print("="*60)
        
        print(f"\nQuestions Evaluated: {total}")
        print(f"  Single-hop: {single_hop}")
        print(f"  Multi-hop: {multi_hop}")
        
        print("\n--- Overall Metrics (1-5 scale) ---")
        for metric in ["correctness", "completeness", "faithfulness", "relevance", "clarity", "overall_score"]:
            m = metrics.get(metric, {})
            print(f"  {metric.capitalize():15} Mean: {m.get('mean', 0):.2f}  |  Single: {m.get('single_hop', 0):.2f}  |  Multi: {m.get('multi_hop', 0):.2f}")
        
        print("\n--- Verdict Distribution ---")
        verdicts = metrics.get("verdict_counts", {})
        print(f"  Pass:    {verdicts.get('pass', 0)}")
        print(f"  Partial: {verdicts.get('partial', 0)}")
        print(f"  Fail:    {verdicts.get('fail', 0)}")
        print(f"  Error:   {verdicts.get('error', 0)}")
        
        print(f"\n  Pass Rate: {metrics.get('pass_rate', 0)*100:.1f}%")
        print("="*60)


def main():
    judge = LLMJudge()
    # Run on all questions (or set max_questions for testing)
    results = judge.run_evaluation(max_questions=None)


if __name__ == "__main__":
    main()
