"""Enhanced RAG Engine with hybrid search and adjustable parameters"""
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from collections import Counter

from config import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, EMBEDDING_MODEL, 
    TOP_K_RESULTS, CHROMA_DIR, TEMPERATURE, MAX_TOKENS
)
from document_processor import DocumentChunk
from prompt_templates import PromptManager

class RAGEngine:
    """Enhanced RAG engine with hybrid search and adjustable parameters"""
    
    def __init__(self):
        """Initialize RAG engine"""
        print("Initializing Enhanced RAG Engine...")
        
        # Initialize embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        print(f"Initializing ChromaDB at {CHROMA_DIR}")
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="research_papers",
            metadata={"description": "Research paper chunks with embeddings"}
        )
        
        # Initialize BM25 index
        self.bm25_index = None
        self.bm25_documents = []
        self.bm25_metadata = []
        
        self.prompt_manager = PromptManager()
        
        print("Enhanced RAG Engine initialized!")
    
    def check_ollama_connection(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def add_documents(self, chunks: List[DocumentChunk]) -> Dict:
        """Add documents to both vector store and BM25 index"""
        if not chunks:
            return {"status": "error", "message": "No chunks provided"}
        
        print(f"Adding {len(chunks)} chunks to vector store...")
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Prepare metadata
        metadatas = [
            {
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "page": chunk.page_number,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        # Generate unique IDs
        ids = [f"{chunk.source}_{chunk.chunk_id}" for chunk in chunks]
        
        # Add to ChromaDB
        try:
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            # Build BM25 index
            print("Building BM25 index...")
            tokenized_docs = [doc.lower().split() for doc in texts]
            self.bm25_index = BM25Okapi(tokenized_docs)
            self.bm25_documents = texts
            self.bm25_metadata = metadatas
            
            print(f"Successfully added {len(chunks)} chunks!")
            return {
                "status": "success",
                "chunks_added": len(chunks),
                "source": chunks[0].source
            }
        except Exception as e:
            print(f"Error adding documents: {e}")
            return {"status": "error", "message": str(e)}
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
        source_filter: Optional[str] = None,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        section_boost: float = 2.0
    ) -> List[Dict]:
        """Hybrid search with adjustable parameters"""
        
        # 1. Semantic search (vector similarity)
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        where_clause = {"source": source_filter} if source_filter else None
        
        vector_results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k * 2,
            where=where_clause
        )
        
        # 2. BM25 keyword search
        bm25_scores = []
        if self.bm25_index:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 3. Combine scores with custom weights
        combined_results = {}
        
        # Add vector results
        if vector_results['documents'][0]:
            for i, doc in enumerate(vector_results['documents'][0]):
                doc_id = vector_results['ids'][0][i]
                metadata = vector_results['metadatas'][0][i]
                distance = vector_results['distances'][0][i]
                
                # Apply section boosting
                boost = section_boost if metadata.get('section') in ['conclusion', 'discussion'] else 1.0
                has_conclusion_boost = 1.5 if metadata.get('has_conclusion') else 1.0
                
                # Normalize distance to similarity score (0-1)
                semantic_score = 1 / (1 + distance)
                
                combined_results[doc_id] = {
                    'content': doc,
                    'metadata': metadata,
                    'semantic_score': semantic_score * boost * has_conclusion_boost,
                    'bm25_score': 0.0
                }
        
        # Add BM25 scores (check if it's a numpy array or list)
        if len(bm25_scores) > 0:
            for idx, score in enumerate(bm25_scores):
                if idx < len(self.bm25_documents):
                    doc_id = f"{self.bm25_metadata[idx]['source']}_{self.bm25_metadata[idx]['chunk_id']}"
                    
                    if doc_id in combined_results:
                        combined_results[doc_id]['bm25_score'] = float(score)
                    else:
                        metadata = self.bm25_metadata[idx]
                        boost = section_boost if metadata.get('section') in ['conclusion', 'discussion'] else 1.0
                        
                        combined_results[doc_id] = {
                            'content': self.bm25_documents[idx],
                            'metadata': metadata,
                            'semantic_score': 0.0,
                            'bm25_score': float(score) * boost
                        }
        
        # 4. Calculate final scores with custom weights
        for doc_id, result in combined_results.items():
            result['final_score'] = (
                semantic_weight * result['semantic_score'] + 
                keyword_weight * result['bm25_score']
            )
        
        # 5. Sort and return
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x['final_score'],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Wrapper method for evaluation compatibility - calls hybrid_search"""
        return self.hybrid_search(query, top_k=top_k)
    
    def generate_query_suggestions(self, query: str, sources: List[Dict]) -> List[str]:
        """Generate smart query suggestions based on current query and retrieved content"""
        suggestions = []
        
        # Analyze sections found
        sections = [s['metadata'].get('section', '') for s in sources]
        section_counter = Counter(sections)
        
        # Generic suggestions based on query type
        query_lower = query.lower()
        
        # If asking about methods
        if any(word in query_lower for word in ['method', 'methodology', 'approach', 'how']):
            suggestions.extend([
                "What were the key findings of this study?",
                "What limitations were mentioned?",
                "How large was the sample size?"
            ])
        
        # If asking about results/findings
        elif any(word in query_lower for word in ['result', 'finding', 'conclusion', 'outcome']):
            suggestions.extend([
                "What methodology was used?",
                "How do these findings compare to previous research?",
                "What future work is suggested?"
            ])
        
        # If asking about introduction/background
        elif any(word in query_lower for word in ['introduction', 'background', 'overview']):
            suggestions.extend([
                "What is the main research question?",
                "What methodology did they use?",
                "What are the key findings?"
            ])
        
        # Default suggestions
        else:
            suggestions.extend([
                "Summarize the main conclusions",
                "What were the study limitations?",
                "Compare the methodologies across papers"
            ])
        
        # Add section-specific suggestions
        if 'method' in section_counter:
            suggestions.append("What was the participant demographic?")
        
        if 'results' in section_counter:
            suggestions.append("Were the results statistically significant?")
        
        if 'discussion' in section_counter:
            suggestions.append("What are the practical implications?")
        
        # Return unique suggestions (max 5)
        return list(dict.fromkeys(suggestions))[:5]
    
    def extract_paper_metrics(self) -> Dict:
        """Extract key metrics from all papers for comparison"""
        stats = self.get_collection_stats()
        
        if stats['total_chunks'] == 0:
            return {}
        
        results = self.collection.get(limit=stats['total_chunks'])
        
        metrics_by_paper = {}
        
        # Common stop words to exclude
        stop_words = {
            'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'has',
            'been', 'were', 'are', 'was', 'will', 'can', 'could', 'would', 'should',
            'may', 'might', 'must', 'also', 'such', 'than', 'then', 'their', 'they',
            'these', 'those', 'there', 'where', 'when', 'which', 'while', 'who',
            'what', 'how', 'why', 'does', 'did', 'done', 'more', 'most', 'other',
            'some', 'into', 'through', 'about', 'after', 'before', 'between',
            'each', 'both', 'all', 'any'
        }
        
        # Common PDF extraction artifacts (reversed/garbage words)
        garbage_words = {
            'foorp', 'rohtua', 'tnetnoc', 'noitamrofni', 'sisylana',
            'hcraeser', 'rebmun', 'tcejbo', 'egami', 'noitceted'
        }
        
        # Common academic filler words
        academic_filler = {
            'paper', 'study', 'research', 'figure', 'table', 
            'results', 'method', 'using', 'based', 'shown',
            'section', 'approach', 'system', 'model', 'analysis',
            'information', 'number', 'value', 'process', 'provide'
        }
        
        for idx, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
            source = meta.get('source', 'Unknown')
            
            if source not in metrics_by_paper:
                metrics_by_paper[source] = {
                    'sections': Counter(),
                    'keywords': Counter(),
                    'total_chunks': 0,
                    'avg_chunk_length': [],
                    'has_methodology': False,
                    'has_results': False,
                    'has_conclusion': False,
                    'all_sections_seen': set()  # Track ALL sections seen
                }
            
            metrics = metrics_by_paper[source]
            metrics['total_chunks'] += 1
            metrics['avg_chunk_length'].append(len(doc))
            
            # Track sections
            section = meta.get('section', 'unknown')
            metrics['sections'][section] += 1
            metrics['all_sections_seen'].add(section)
            
            # IMPORTANT: Also check split_page_sections (stored as comma-separated string)
            split_sections_str = meta.get('split_page_sections', '')
            if split_sections_str:
                split_sections = split_sections_str.split(',')  # Convert string back to list
                for split_sec in split_sections:
                    if split_sec:  # Make sure it's not empty
                        metrics['all_sections_seen'].add(split_sec)
                        print(f"  🎯 DEBUG: Found split page section '{split_sec}' in {source}")
            
            # Check for key sections - MORE FLEXIBLE
            if section in ['method', 'methodology']:
                metrics['has_methodology'] = True
            if section == 'results':
                metrics['has_results'] = True
            if section in ['conclusion', 'discussion']:
                metrics['has_conclusion'] = True
            
            # Extract keywords with BETTER FILTERING
            words = doc.lower().split()
            clean_words = []
            
            for word in words:
                # Remove punctuation from start and end
                word = word.strip('.,;:!?()[]{}"\'-')
                
                # Filter criteria:
                # 1. Must be 5+ characters
                # 2. Must be all alphabetic
                # 3. Not in stop words
                # 4. Not in garbage words (reversed/artifacts)
                # 5. Not in academic filler
                # 6. Not too many consonants (likely garbage)
                
                if len(word) >= 5 and word.isalpha():
                    # Skip if in any blacklist
                    if word in stop_words or word in garbage_words or word in academic_filler:
                        continue
                    
                    # Check for suspicious consonant patterns (likely reversed words)
                    vowels = sum(1 for c in word if c in 'aeiou')
                    consonants = len(word) - vowels
                    
                    # Skip words with too few vowels (likely garbage)
                    if vowels < 2 or consonants / len(word) > 0.7:
                        continue
                    
                    clean_words.append(word)
            
            metrics['keywords'].update(clean_words)
        
        # Calculate averages and ALSO check all_sections_seen
        for source, metrics in metrics_by_paper.items():
            metrics['avg_chunk_length'] = int(np.mean(metrics['avg_chunk_length']))
            
            # ADDITIONAL CHECK: Look at all sections that appeared
            # This catches sections on split pages
            if 'results' in metrics['all_sections_seen'] or any('result' in s for s in metrics['all_sections_seen']):
                metrics['has_results'] = True
            if any(s in metrics['all_sections_seen'] for s in ['conclusion', 'discussion']):
                metrics['has_conclusion'] = True
            if any(s in metrics['all_sections_seen'] for s in ['method', 'methodology']):
                metrics['has_methodology'] = True
            
            print(f"\n📊 DEBUG - Final sections for {source}:")
            print(f"   All sections seen: {metrics['all_sections_seen']}")
            print(f"   Has methodology: {metrics['has_methodology']}")
            print(f"   Has results: {metrics['has_results']}")
            print(f"   Has conclusion: {metrics['has_conclusion']}")
            
            # Get top keywords
            top_keywords = []
            for word, count in metrics['keywords'].most_common(20):
                if (word.isalpha() and 
                    len(word) >= 5 and 
                    not any(char in word for char in ['(', ')', '[', ']', '{', '}', '<', '>'])):
                    top_keywords.append(word)
                
                if len(top_keywords) >= 10:
                    break
            
            metrics['top_keywords'] = top_keywords
            del metrics['keywords']
            del metrics['all_sections_seen']  # Don't need this in final output
        
        return metrics_by_paper
    
    def generate_response(
        self,
        query: str,
        task_type: str = "qa",
        top_k: int = TOP_K_RESULTS,
        source_filter: Optional[str] = None,
        semantic_weight: float = 0.6,
        keyword_weight: float = 0.4,
        section_boost: float = 2.0,
        temperature: float = TEMPERATURE
    ) -> Dict:
        """Generate response with adjustable parameters"""
        
        if not self.check_ollama_connection():
            return {
                "status": "error",
                "response": "Cannot connect to Ollama. Please ensure Ollama is running.",
                "sources": [],
                "suggestions": []
            }
        
        # Use hybrid search with custom parameters
        print(f"Hybrid search: semantic={semantic_weight}, keyword={keyword_weight}, boost={section_boost}")
        relevant_chunks = self.hybrid_search(
            query, top_k, source_filter, 
            semantic_weight, keyword_weight, section_boost
        )
        
        if not relevant_chunks:
            return {
                "status": "error",
                "response": "No relevant information found in the knowledge base.",
                "sources": [],
                "suggestions": []
            }
        
        # Generate query suggestions
        suggestions = self.generate_query_suggestions(query, relevant_chunks)
        
        # Build context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown')
            section = chunk['metadata'].get('section', 'unknown')
            page = chunk['metadata'].get('page', '?')
            content = chunk['content']
            
            context_parts.append(
                f"[Source {i}: {source} - Page {page} - Section: {section}]\n{content}\n"
            )
        
        context = "\n---\n".join(context_parts)
        
        # Get prompt
        system_prompt, user_prompt = self.prompt_manager.get_prompt(
            task_type, context, query
        )
        
        # Generate response
        print("Generating response...")
        try:
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": f"{system_prompt}\n\n{user_prompt}",
                    "stream": True,
                    "options": {
                        "temperature": temperature,
                        "num_predict": MAX_TOKENS,
                        "num_ctx": 4096
                    }
                },
                timeout=300,
                stream=True
            )
            
            if response.status_code == 200:
                generated_text = ""
                
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = requests.compat.json.loads(line)
                            if 'response' in json_response:
                                generated_text += json_response['response']
                            if json_response.get('done', False):
                                break
                        except:
                            continue
                
                sources = [
                    {
                        'source': chunk['metadata'].get('source'),
                        'section': chunk['metadata'].get('section'),
                        'page': chunk['metadata'].get('page'),
                        'chunk_id': chunk['metadata'].get('chunk_id'),
                        'score': round(chunk.get('final_score', 0), 3),
                        'preview': chunk['content'][:200] + "..."
                    }
                    for chunk in relevant_chunks
                ]
                
                return {
                    "status": "success",
                    "response": generated_text if generated_text else "No response generated.",
                    "sources": sources,
                    "num_sources": len(sources),
                    "suggestions": suggestions
                }
            else:
                return {
                    "status": "error",
                    "response": f"Ollama API error: {response.status_code}",
                    "sources": [],
                    "suggestions": []
                }
                
        except Exception as e:
            return {
                "status": "error",
                "response": f"Error: {str(e)}",
                "sources": [],
                "suggestions": []
            }
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        count = self.collection.count()
        
        if count > 0:
            results = self.collection.get(limit=count)
            sources = set(meta['source'] for meta in results['metadatas'])
            
            # Count sections
            sections = {}
            for meta in results['metadatas']:
                section = meta.get('section', 'unknown')
                sections[section] = sections.get(section, 0) + 1
        else:
            sources = set()
            sections = {}
        
        return {
            "total_chunks": count,
            "unique_documents": len(sources),
            "documents": list(sources),
            "sections": sections
        }
    
    def clear_collection(self):
        """Clear all documents"""
        try:
            self.chroma_client.delete_collection("research_papers")
            self.collection = self.chroma_client.create_collection(
                name="research_papers",
                metadata={"description": "Research paper chunks with embeddings"}
            )
            self.bm25_index = None
            self.bm25_documents = []
            self.bm25_metadata = []
            return {"status": "success", "message": "Collection cleared"}
        except Exception as e:
            return {"status": "error", "message": str(e)}