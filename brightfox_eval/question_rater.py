"""Rate generated questions for quality using LLM"""
import json
from typing import List, Dict, Any
from pathlib import Path

from config import config
from llm_client import LLMClient


class QuestionRater:
    """Rate questions for quality on a 1-5 scale"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.output_dir = Path(__file__).parent / config.OUTPUT_DIR
        
    def load_questions(self) -> Dict[str, List[Dict]]:
        """Load generated questions"""
        questions_file = self.output_dir / "generated_questions.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}. Run question_generator.py first.")
        
        with open(questions_file, 'r') as f:
            return json.load(f)
    
    def rate_question(self, question: Dict) -> Dict:
        """Rate a single question on quality (1-5 scale)"""
        q_type = question.get('question_type', 'unknown')
        q_text = question.get('question', '')
        answer = question.get('ground_truth_answer', '')
        reasoning = question.get('reasoning', '')
        
        prompt = f"""You are an expert evaluator for RAG (Retrieval Augmented Generation) evaluation datasets.
This is a SCADA/Solar/Electrical equipment technical corpus.

Rate the following {q_type} question on a scale of 1-5 based on these criteria:
- Clarity: Is the question clear and unambiguous?
- Specificity: Is it specific enough to have a definite answer?
- Technical Quality: Is it appropriate for a technical SCADA/Solar domain?
- Answerability: Can it be answered from the provided context?
- Evaluation Value: Would it be useful for evaluating a RAG system?

Question: {q_text}

Ground Truth Answer: {answer}

Original Reasoning: {reasoning}

Rate this question and respond with JSON in this exact format:
{{
    "score": <1-5 integer>,
    "clarity_score": <1-5>,
    "specificity_score": <1-5>,
    "technical_quality_score": <1-5>,
    "evaluation_value_score": <1-5>,
    "feedback": "Brief explanation of the rating",
    "issues": ["list", "of", "any", "issues"] or []
}}

Scoring guide:
5 = Excellent - High quality, clear, specific, valuable for evaluation
4 = Good - Minor issues but still useful
3 = Acceptable - Some issues, borderline useful
2 = Poor - Significant issues, not recommended
1 = Bad - Major problems, should not be used"""

        try:
            result = self.llm.generate_json(prompt)
            return result
        except Exception as e:
            print(f"Failed to rate question: {e}")
            return {"score": 0, "feedback": f"Rating failed: {str(e)}", "issues": ["rating_failed"]}
    
    def rate_all_questions(self) -> Dict[str, List[Dict]]:
        """Rate all generated questions"""
        questions = self.load_questions()
        rated_questions = {'single_hop': [], 'multi_hop': []}
        
        for q_type in ['single_hop', 'multi_hop']:
            print(f"\n--- Rating {q_type.replace('_', '-')} questions ---")
            for i, question in enumerate(questions.get(q_type, [])):
                rating = self.rate_question(question)
                question['rating'] = rating
                rated_questions[q_type].append(question)
                score = rating.get('score', 0)
                print(f"  Question {i+1}: Score {score}/5 - {rating.get('feedback', '')[:50]}...")
        
        # Save rated questions
        output_file = self.output_dir / "rated_questions.json"
        with open(output_file, 'w') as f:
            json.dump(rated_questions, f, indent=2)
        
        print(f"\n✓ Saved rated questions to {output_file}")
        
        return rated_questions
    
    def filter_high_quality(self, min_score: int = None) -> Dict[str, List[Dict]]:
        """Filter to keep only high-quality questions (score >= min_score)"""
        if min_score is None:
            min_score = config.MIN_QUALITY_SCORE
        
        rated_file = self.output_dir / "rated_questions.json"
        if not rated_file.exists():
            raise FileNotFoundError(f"Rated questions not found: {rated_file}. Run rate_all_questions first.")
        
        with open(rated_file, 'r') as f:
            rated_questions = json.load(f)
        
        filtered = {'single_hop': [], 'multi_hop': []}
        discarded = {'single_hop': [], 'multi_hop': []}
        
        for q_type in ['single_hop', 'multi_hop']:
            for question in rated_questions.get(q_type, []):
                score = question.get('rating', {}).get('score', 0)
                if score >= min_score:
                    filtered[q_type].append(question)
                else:
                    discarded[q_type].append(question)
        
        # Save filtered questions
        filtered_file = self.output_dir / "filtered_questions.json"
        with open(filtered_file, 'w') as f:
            json.dump(filtered, f, indent=2)
        
        # Save discarded for reference
        discarded_file = self.output_dir / "discarded_questions.json"
        with open(discarded_file, 'w') as f:
            json.dump(discarded, f, indent=2)
        
        print(f"\n--- Filtering Results (min_score={min_score}) ---")
        print(f"Single-hop: {len(filtered['single_hop'])} kept, {len(discarded['single_hop'])} discarded")
        print(f"Multi-hop: {len(filtered['multi_hop'])} kept, {len(discarded['multi_hop'])} discarded")
        print(f"Total: {len(filtered['single_hop']) + len(filtered['multi_hop'])} questions for evaluation")
        print(f"\n✓ Saved filtered questions to {filtered_file}")
        
        return filtered


if __name__ == "__main__":
    rater = QuestionRater()
    
    # Rate all questions
    rated = rater.rate_all_questions()
    
    # Filter to high quality
    filtered = rater.filter_high_quality()
