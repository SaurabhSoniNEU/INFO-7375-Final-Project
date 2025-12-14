"""Comprehensive evaluation script for the RAG system"""
import time
import json
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict

from rag_engine import RAGEngine
from document_processor import DocumentProcessor

class RAGEvaluator:
    """Evaluate RAG system performance"""
    
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.results = {
            'retrieval': {},
            'response': {},
            'latency': {},
            'overall': {}
        }
    
    def create_test_questions(self) -> List[Dict]:
        """Create test questions with expected answers"""
        return [
            {
                'question': 'What are the main conclusions of the study?',
                'expected_section': 'conclusion',
                'task_type': 'qa',
                'difficulty': 'easy'
            },
            {
                'question': 'What methodology was used in the research?',
                'expected_section': 'method',
                'task_type': 'qa',
                'difficulty': 'medium'
            },
            {
                'question': 'Summarize the key findings',
                'expected_section': 'results',
                'task_type': 'summary',
                'difficulty': 'medium'
            },
            {
                'question': 'What are the limitations mentioned in the study?',
                'expected_section': 'discussion',
                'task_type': 'extract',
                'difficulty': 'hard'
            },
            {
                'question': 'Compare the approaches used',
                'expected_section': 'method',
                'task_type': 'compare',
                'difficulty': 'hard'
            },
            {
                'question': 'What were the study participants?',
                'expected_section': 'method',
                'task_type': 'extract',
                'difficulty': 'easy'
            },
            {
                'question': 'Provide a critical analysis of the methodology',
                'expected_section': 'method',
                'task_type': 'critique',
                'difficulty': 'hard'
            },
            {
                'question': 'What future work is suggested?',
                'expected_section': 'conclusion',
                'task_type': 'extract',
                'difficulty': 'medium'
            }
        ]
    
    def evaluate_retrieval_accuracy(
        self,
        test_questions: List[Dict],
        top_k: int = 5
    ) -> Dict:
        """Evaluate retrieval quality"""
        print("\n" + "="*60)
        print("EVALUATING RETRIEVAL ACCURACY")
        print("="*60)
        
        results = {
            'section_accuracy': [],
            'avg_rank': [],
            'mrr': [],  # Mean Reciprocal Rank
            'ndcg': []  # Normalized Discounted Cumulative Gain
        }
        
        for i, test in enumerate(test_questions, 1):
            print(f"\nTest {i}/{len(test_questions)}: {test['question'][:50]}...")
            
            # Retrieve chunks
            chunks = self.rag_engine.retrieve_relevant_chunks(
                test['question'],
                top_k=top_k
            )
            
            if not chunks:
                results['section_accuracy'].append(0)
                results['avg_rank'].append(top_k + 1)
                results['mrr'].append(0)
                results['ndcg'].append(0)
                continue
            
            # Check if expected section is in top results
            expected_section = test['expected_section']
            sections_found = [c['metadata'].get('section') for c in chunks]
            
            # Section accuracy
            section_correct = expected_section in sections_found
            results['section_accuracy'].append(1 if section_correct else 0)
            
            # Rank of first correct result
            try:
                rank = sections_found.index(expected_section) + 1
                results['avg_rank'].append(rank)
                results['mrr'].append(1.0 / rank)
                
                print(f"  ✅ Found in position {rank}")
            except ValueError:
                results['avg_rank'].append(top_k + 1)
                results['mrr'].append(0)
                print(f"  ❌ Expected section '{expected_section}' not found")
            
            # NDCG calculation (simplified)
            relevance_scores = [
                2 if c['metadata'].get('section') == expected_section else 1
                for c in chunks
            ]
            dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores))
            ideal_dcg = sum(2 / np.log2(i + 2) for i in range(min(top_k, len(relevance_scores))))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
            results['ndcg'].append(ndcg)
        
        # Calculate averages
        summary = {
            'section_accuracy': np.mean(results['section_accuracy']) * 100,
            'mean_rank': np.mean(results['avg_rank']),
            'mrr': np.mean(results['mrr']),
            'ndcg@5': np.mean(results['ndcg'])
        }
        
        print("\n" + "-"*60)
        print("RETRIEVAL RESULTS:")
        print(f"  Section Accuracy: {summary['section_accuracy']:.1f}%")
        print(f"  Mean Rank: {summary['mean_rank']:.2f}")
        print(f"  MRR: {summary['mrr']:.3f}")
        print(f"  NDCG@5: {summary['ndcg@5']:.3f}")
        print("-"*60)
        
        self.results['retrieval'] = summary
        return summary
    
    def evaluate_response_quality(
        self,
        test_questions: List[Dict]
    ) -> Dict:
        """Evaluate generated response quality"""
        print("\n" + "="*60)
        print("EVALUATING RESPONSE QUALITY")
        print("="*60)
        
        results = {
            'response_length': [],
            'has_citations': [],
            'coherence_score': [],
            'by_task_type': defaultdict(list)
        }
        
        for i, test in enumerate(test_questions, 1):
            print(f"\nTest {i}/{len(test_questions)}: {test['question'][:50]}...")
            
            start_time = time.time()
            
            # Generate response
            result = self.rag_engine.generate_response(
                query=test['question'],
                task_type=test['task_type'],
                top_k=5
            )
            
            latency = time.time() - start_time
            
            if result['status'] != 'success':
                print(f"  ❌ Failed: {result['response']}")
                continue
            
            response = result['response']
            
            # Metrics
            response_length = len(response.split())
            has_citations = any(word in response.lower() for word in ['source', 'according', 'based on'])
            
            # Simple coherence check (heuristic)
            coherence = self._calculate_coherence(response)
            
            results['response_length'].append(response_length)
            results['has_citations'].append(1 if has_citations else 0)
            results['coherence_score'].append(coherence)
            results['by_task_type'][test['task_type']].append({
                'length': response_length,
                'latency': latency,
                'coherence': coherence
            })
            
            print(f"  ✅ Response: {response_length} words, {latency:.2f}s")
            print(f"     Citations: {'Yes' if has_citations else 'No'}")
            print(f"     Coherence: {coherence:.2f}")
        
        # Calculate summary
        summary = {
            'avg_length': np.mean(results['response_length']),
            'citation_rate': np.mean(results['has_citations']) * 100,
            'avg_coherence': np.mean(results['coherence_score']),
            'by_task_type': {}
        }
        
        for task_type, metrics in results['by_task_type'].items():
            summary['by_task_type'][task_type] = {
                'avg_length': np.mean([m['length'] for m in metrics]),
                'avg_latency': np.mean([m['latency'] for m in metrics]),
                'avg_coherence': np.mean([m['coherence'] for m in metrics])
            }
        
        print("\n" + "-"*60)
        print("RESPONSE QUALITY RESULTS:")
        print(f"  Avg Length: {summary['avg_length']:.0f} words")
        print(f"  Citation Rate: {summary['citation_rate']:.1f}%")
        print(f"  Avg Coherence: {summary['avg_coherence']:.2f}")
        print("\n  By Task Type:")
        for task, metrics in summary['by_task_type'].items():
            print(f"    {task}: {metrics['avg_length']:.0f} words, {metrics['avg_latency']:.2f}s")
        print("-"*60)
        
        self.results['response'] = summary
        return summary
    
    def evaluate_latency(
        self,
        test_questions: List[Dict],
        runs: int = 3
    ) -> Dict:
        """Evaluate system latency"""
        print("\n" + "="*60)
        print("EVALUATING SYSTEM LATENCY")
        print("="*60)
        
        latencies = {
            'retrieval': [],
            'generation': [],
            'total': []
        }
        
        for test in test_questions[:5]:  # Use subset for speed
            for run in range(runs):
                # Measure retrieval
                start = time.time()
                chunks = self.rag_engine.retrieve_relevant_chunks(test['question'])
                retrieval_time = time.time() - start
                
                # Measure generation
                start = time.time()
                result = self.rag_engine.generate_response(
                    query=test['question'],
                    task_type=test['task_type']
                )
                generation_time = time.time() - start
                
                total_time = retrieval_time + generation_time
                
                latencies['retrieval'].append(retrieval_time)
                latencies['generation'].append(generation_time)
                latencies['total'].append(total_time)
        
        summary = {
            'avg_retrieval': np.mean(latencies['retrieval']),
            'avg_generation': np.mean(latencies['generation']),
            'avg_total': np.mean(latencies['total']),
            'p95_total': np.percentile(latencies['total'], 95),
            'p99_total': np.percentile(latencies['total'], 99)
        }
        
        print(f"\n  Avg Retrieval Time: {summary['avg_retrieval']:.3f}s")
        print(f"  Avg Generation Time: {summary['avg_generation']:.3f}s")
        print(f"  Avg Total Time: {summary['avg_total']:.3f}s")
        print(f"  P95 Total Time: {summary['p95_total']:.3f}s")
        print(f"  P99 Total Time: {summary['p99_total']:.3f}s")
        print("-"*60)
        
        self.results['latency'] = summary
        return summary
    
    def _calculate_coherence(self, text: str) -> float:
        """Simple heuristic for text coherence"""
        # Check for basic quality indicators
        score = 0.5  # Base score
        
        # Has proper sentences
        sentences = text.split('.')
        if len(sentences) > 2:
            score += 0.1
        
        # Has paragraphs
        if '\n' in text:
            score += 0.1
        
        # Reasonable length
        words = text.split()
        if 50 < len(words) < 500:
            score += 0.1
        
        # No excessive repetition (simple check)
        unique_ratio = len(set(words)) / max(len(words), 1)
        if unique_ratio > 0.4:
            score += 0.2
        
        return min(score, 1.0)
    
    def run_full_evaluation(self) -> Dict:
        """Run complete evaluation suite"""
        print("\n" + "="*60)
        print("STARTING FULL EVALUATION")
        print("="*60)
        
        # Check if documents are loaded
        stats = self.rag_engine.get_collection_stats()
        print(f"\nDocuments in knowledge base: {stats['total_chunks']} chunks")
        
        if stats['total_chunks'] == 0:
            print("❌ No documents loaded. Please upload documents first.")
            return {}
        
        # Create test questions
        test_questions = self.create_test_questions()
        print(f"Created {len(test_questions)} test questions")
        
        # Run evaluations
        retrieval_results = self.evaluate_retrieval_accuracy(test_questions)
        response_results = self.evaluate_response_quality(test_questions)
        latency_results = self.evaluate_latency(test_questions)
        
        # Calculate overall score
        overall_score = (
            retrieval_results['section_accuracy'] * 0.4 +
            response_results['citation_rate'] * 0.3 +
            (1 / max(latency_results['avg_total'], 0.1)) * 10 * 0.3
        )
        
        self.results['overall'] = {
            'score': overall_score,
            'retrieval': retrieval_results,
            'response': response_results,
            'latency': latency_results
        }
        
        # Save results
        self.save_results()
        
        # Print final summary
        self.print_final_summary()
        
        return self.results
    
    def save_results(self, filename: str = 'evaluation_results.json'):
        """Save evaluation results to file"""
        output_dir = Path('evaluation_results')
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to {filepath}")
    
    def print_final_summary(self):
        """Print formatted final summary"""
        print("\n" + "="*60)
        print("FINAL EVALUATION SUMMARY")
        print("="*60)
        
        print("\n📊 OVERALL METRICS:")
        print(f"  Overall Score: {self.results['overall']['score']:.1f}/100")
        print(f"  Retrieval Accuracy: {self.results['retrieval']['section_accuracy']:.1f}%")
        print(f"  Avg Response Time: {self.results['latency']['avg_total']:.2f}s")
        print(f"  Citation Rate: {self.results['response']['citation_rate']:.1f}%")
        
        print("\n🎯 DETAILED BREAKDOWN:")
        print(f"  • Retrieval MRR: {self.results['retrieval']['mrr']:.3f}")
        print(f"  • Retrieval NDCG@5: {self.results['retrieval']['ndcg@5']:.3f}")
        print(f"  • Avg Response Length: {self.results['response']['avg_length']:.0f} words")
        print(f"  • P95 Latency: {self.results['latency']['p95_total']:.3f}s")
        
        print("\n" + "="*60)

def main():
    """Main evaluation function"""
    print("🔬 RAG System Evaluation Tool")
    print("="*60)
    
    # Initialize RAG engine
    rag_engine = RAGEngine()
    
    # Check if documents are loaded
    stats = rag_engine.get_collection_stats()
    if stats['total_chunks'] == 0:
        print("\n⚠️  No documents in knowledge base!")
        print("Please upload documents through the web interface first.")
        print("Run: python app.py")
        return
    
    # Create evaluator
    evaluator = RAGEvaluator(rag_engine)
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    print("\n✅ Evaluation complete!")
    print("Check evaluation_results/ directory for detailed results.")

if __name__ == "__main__":
    main()