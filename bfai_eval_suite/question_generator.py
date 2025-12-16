"""Generate single-hop and multi-hop questions from document chunks"""
import json
import random
from typing import List, Dict, Any
from pathlib import Path

from config import config
from llm_client import LLMClient


class QuestionGenerator:
    """Generate evaluation questions from document corpus"""
    
    def __init__(self):
        self.llm = LLMClient()
        self.data_dir = Path(__file__).parent / config.DATA_DIR
        
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load all chunks from the combined file"""
        chunks_file = self.data_dir / "all_chunks.json"
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}. Run download_chunks.py first.")
        
        with open(chunks_file, 'r') as f:
            return json.load(f)
    
    def _sample_chunks_for_single_hop(self, chunks: List[Dict], num_samples: int) -> List[Dict]:
        """Sample chunks that are good candidates for single-hop questions"""
        # Filter chunks with sufficient content
        valid_chunks = [c for c in chunks if len(c.get('text', c.get('content', ''))) > 200]
        
        # Sample from different documents for diversity
        docs = {}
        for chunk in valid_chunks:
            doc = chunk.get('source_document', 'unknown')
            if doc not in docs:
                docs[doc] = []
            docs[doc].append(chunk)
        
        # Sample evenly across documents
        sampled = []
        doc_list = list(docs.keys())
        random.shuffle(doc_list)
        
        idx = 0
        while len(sampled) < num_samples and idx < len(doc_list) * 3:
            doc = doc_list[idx % len(doc_list)]
            if docs[doc]:
                chunk = random.choice(docs[doc])
                docs[doc].remove(chunk)
                sampled.append(chunk)
            idx += 1
        
        return sampled[:num_samples]
    
    def _sample_chunks_for_multi_hop(self, chunks: List[Dict], num_samples: int) -> List[List[Dict]]:
        """Sample pairs of related chunks for multi-hop questions"""
        # Group by document
        docs = {}
        for chunk in chunks:
            doc = chunk.get('source_document', 'unknown')
            if doc not in docs:
                docs[doc] = []
            docs[doc].append(chunk)
        
        # Find documents with multiple chunks
        multi_chunk_docs = {k: v for k, v in docs.items() if len(v) >= 2}
        
        pairs = []
        doc_list = list(multi_chunk_docs.keys())
        random.shuffle(doc_list)
        
        for doc in doc_list:
            if len(pairs) >= num_samples:
                break
            doc_chunks = multi_chunk_docs[doc]
            if len(doc_chunks) >= 2:
                # Sample 2 chunks from same document
                pair = random.sample(doc_chunks, 2)
                pairs.append(pair)
        
        # If not enough same-doc pairs, create cross-doc pairs
        while len(pairs) < num_samples:
            all_chunks = [c for c in chunks if len(c.get('text', c.get('content', ''))) > 200]
            if len(all_chunks) >= 2:
                pair = random.sample(all_chunks, 2)
                pairs.append(pair)
            else:
                break
        
        return pairs[:num_samples]
    
    def generate_single_hop_questions(self, chunks: List[Dict], num_questions: int = 25) -> List[Dict]:
        """Generate single-hop questions that can be answered from a single chunk"""
        sampled_chunks = self._sample_chunks_for_single_hop(chunks, num_questions)
        questions = []
        
        for i, chunk in enumerate(sampled_chunks):
            text = chunk.get('text', chunk.get('content', ''))
            doc_name = chunk.get('source_document', 'unknown')
            
            prompt = f"""You are an expert at creating evaluation questions for a RAG (Retrieval Augmented Generation) system.
This is a SCADA/Solar/Electrical equipment technical corpus.

Given the following document chunk, create ONE high-quality single-hop question that:
1. Can be answered directly from the information in this chunk
2. Is specific and technical (not vague)
3. Has a clear, factual answer
4. Would be useful for testing a RAG system's retrieval and answer quality

Document: {doc_name}
Chunk content:
{text[:3000]}

Respond with JSON in this exact format:
{{
    "question": "Your question here",
    "ground_truth_answer": "The correct answer based on the chunk",
    "reasoning": "Why this is a good evaluation question"
}}"""

            try:
                result = self.llm.generate_json(prompt)
                result['chunk_id'] = chunk.get('id', chunk.get('chunk_id', f'chunk_{i}'))
                result['source_document'] = doc_name
                result['question_type'] = 'single_hop'
                result['source_text'] = text[:2000]
                questions.append(result)
                print(f"✓ Generated single-hop question {len(questions)}/{num_questions}")
            except Exception as e:
                print(f"✗ Failed to generate question for chunk {i}: {e}")
                continue
        
        return questions
    
    def generate_multi_hop_questions(self, chunks: List[Dict], num_questions: int = 25) -> List[Dict]:
        """Generate multi-hop questions that require information from multiple chunks"""
        chunk_pairs = self._sample_chunks_for_multi_hop(chunks, num_questions)
        questions = []
        
        for i, pair in enumerate(chunk_pairs):
            text1 = pair[0].get('text', pair[0].get('content', ''))
            text2 = pair[1].get('text', pair[1].get('content', ''))
            doc1 = pair[0].get('source_document', 'unknown')
            doc2 = pair[1].get('source_document', 'unknown')
            
            prompt = f"""You are an expert at creating evaluation questions for a RAG (Retrieval Augmented Generation) system.
This is a SCADA/Solar/Electrical equipment technical corpus.

Given the following TWO document chunks, create ONE high-quality multi-hop question that:
1. REQUIRES information from BOTH chunks to answer completely
2. Tests the ability to synthesize information across documents
3. Is specific and technical (not vague)
4. Has a clear, factual answer that combines information from both sources

Document 1: {doc1}
Chunk 1:
{text1[:2000]}

Document 2: {doc2}
Chunk 2:
{text2[:2000]}

Respond with JSON in this exact format:
{{
    "question": "Your multi-hop question here",
    "ground_truth_answer": "The correct answer combining information from both chunks",
    "reasoning": "Why this requires both chunks and is a good evaluation question"
}}"""

            try:
                result = self.llm.generate_json(prompt)
                result['chunk_ids'] = [
                    pair[0].get('id', pair[0].get('chunk_id', f'chunk_a_{i}')),
                    pair[1].get('id', pair[1].get('chunk_id', f'chunk_b_{i}'))
                ]
                result['source_documents'] = [doc1, doc2]
                result['question_type'] = 'multi_hop'
                result['source_texts'] = [text1[:1500], text2[:1500]]
                questions.append(result)
                print(f"✓ Generated multi-hop question {len(questions)}/{num_questions}")
            except Exception as e:
                print(f"✗ Failed to generate multi-hop question {i}: {e}")
                continue
        
        return questions
    
    def generate_all_questions(self) -> Dict[str, List[Dict]]:
        """Generate both single-hop and multi-hop questions"""
        chunks = self.load_chunks()
        print(f"Loaded {len(chunks)} chunks")
        
        print("\n--- Generating Single-Hop Questions ---")
        single_hop = self.generate_single_hop_questions(chunks, config.NUM_SINGLE_HOP_QUESTIONS)
        
        print("\n--- Generating Multi-Hop Questions ---")
        multi_hop = self.generate_multi_hop_questions(chunks, config.NUM_MULTI_HOP_QUESTIONS)
        
        all_questions = {
            'single_hop': single_hop,
            'multi_hop': multi_hop
        }
        
        # Save to file
        output_file = self.data_dir.parent / config.OUTPUT_DIR / "generated_questions.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(all_questions, f, indent=2)
        
        print(f"\n✓ Saved {len(single_hop)} single-hop and {len(multi_hop)} multi-hop questions to {output_file}")
        
        return all_questions


if __name__ == "__main__":
    generator = QuestionGenerator()
    questions = generator.generate_all_questions()
