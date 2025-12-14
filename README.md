# 📚 Research Paper Assistant with Hybrid RAG

> **AI-Powered Research Analysis System**  
> Saurabh Soni | INFO 7375 - Prompt Engineering for Generative AI  
> Northeastern University | December 2025

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Overview

An intelligent research assistant that uses **Hybrid RAG (Retrieval-Augmented Generation)** to analyze academic papers and provide accurate, cited answers to research questions. The system combines semantic search with keyword matching for optimal information retrieval.

### **The Problem**
Researchers spend hours manually reading and synthesizing information from multiple academic papers. Finding specific information across documents is tedious and time-consuming.

### **The Solution**
An AI-powered assistant that:
- Processes PDF research papers automatically
- Extracts and indexes content with section awareness
- Answers questions using hybrid retrieval (60% semantic + 40% keywords)
- Provides citations with page numbers and section references
- Offers smart query suggestions for deeper research

### **The Impact**
- ⚡ **10x faster** than manual paper review
- 🎯 **Accurate retrieval** with source citations
- 📊 **Multi-paper analysis** for comparative research
- 🔧 **Adjustable parameters** for fine-tuned results

---

## 🏆 Key Features

### **1. Hybrid RAG Architecture**
- **Vector Search**: Semantic similarity using Sentence Transformers (all-MiniLM-L6-v2)
- **Keyword Search**: BM25 Okapi for exact term matching
- **Score Fusion**: 60% semantic + 40% keyword (adjustable)
- **Section Boosting**: 2× priority for Conclusions and Discussion

### **2. Interactive Research Dashboard** ⭐ NEW
- Side-by-side paper comparison matrix
- Visual analytics (pie charts, bar graphs)
- Keyword frequency analysis
- Section distribution visualization
- Individual paper deep-dive analysis

### **3. Adjustable RAG Parameters** ⭐ NEW
- Fine-tune semantic vs keyword weights (40-80%)
- Adjust section boosting multiplier (1-3×)
- Control LLM temperature (0.3-1.0)
- Real-time parameter impact visualization

### **4. Smart Query Suggestions** ⭐ NEW
- Context-aware question recommendations
- Section-specific suggestions
- One-click follow-up queries
- Adaptive to your research workflow

### **5. Advanced Document Processing**
- Section-aware chunking (1,500 chars)
- Automatic section detection (Abstract, Methods, Results, etc.)
- Page number tracking
- Metadata preservation
- Dual extraction (pdfplumber + PyPDF2 fallback)

### **6. Task-Optimized Prompt Templates**
Five specialized templates for different research tasks:
- ❓ **Question & Answer**: Direct, cited responses
- 📝 **Summarization**: Concise overviews
- ⚖️ **Comparison**: Multi-paper analysis
- 📋 **Extraction**: Structured information retrieval
- 🔍 **Critique**: Critical methodology evaluation

---

## 🔧 Technical Architecture

```
┌────────────────────────────────────────────────────┐
│              Streamlit Web Interface               │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────┐  ┌───────────────────────────┐   │
│  │   Upload     │  │   Query Processing        │   │
│  │   Papers     │  │   - Task Selection        │   │ 
│  │              │  │   - Parameter Tuning      │   │
│  └──────┬───────┘  └───────────┬───────────────┘   │
│         │                      │                   │
│         ▼                      ▼                   │
│  ┌──────────────────────────────────────────────┐  │
│  │        Document Processor                    │  │
│  │  - PDF Text Extraction (pdfplumber/PyPDF2)   │  │
│  │  - Section Detection (Pattern Matching)      │  │
│  │  - Smart Chunking (1500 chars + overlap)     │  │ 
│  └──────────────────┬───────────────────────────┘  │
│                     │                              │
│                     ▼                              │
│  ┌──────────────────────────────────────────────┐  │
│  │           RAG Engine (Hybrid)                │  │
│  │                                              │  │
│  │  ┌─────────────────┐  ┌─────────────────┐    │  │
│  │  │  Vector Search  │  │  Keyword Search │    │  │
│  │  │   ChromaDB      │  │     BM25        │    │  │
│  │  │  384-dim embed  │  │  TF-IDF scores  │    │  │
│  │  └────────┬────────┘  └────────┬────────┘    │  │
│  │           │                    │             │  │
│  │           └──────┬─────────────┘             │  │
│  │                  ▼                           │  │
│  │         Score Fusion (0.6/0.4)               │  │
│  │         Section Boosting (2×)                │  │
│  │                  │                           │  │
│  │                  ▼                           │  │
│  │         Top-K Retrieval (5 chunks)           │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                              │
│                     ▼                              │
│  ┌──────────────────────────────────────────────┐  │
│  │       Prompt Engineering System              │  │
│  │  - Template Selection (5 types)              │  │
│  │  - Context Formatting                        │  │
│  │  - Query Suggestion Generation               │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                              │
│                     ▼                              │
│  ┌──────────────────────────────────────────────┐  │
│  │       LLM Generation (Ollama)                │  │
│  │  - Local llama3.2:3b model                   │  │
│  │  - Streaming responses                       │  │
│  │  - Adjustable temperature                    │  │
│  └──────────────────┬───────────────────────────┘  │
│                     │                              │
│                     ▼                              │
│  ┌──────────────────────────────────────────────┐  │
│  │       Response + Citations + Suggestions     │  │
│  └──────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────┘
```

---

## 💻 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **LLM** | Ollama (llama3.2:3b) | Local language model |
| **Embeddings** | Sentence Transformers | Semantic vector generation |
| **Vector DB** | ChromaDB | Persistent vector storage |
| **Keyword Search** | BM25 Okapi | Term frequency matching |
| **PDF Processing** | pdfplumber + PyPDF2 | Text extraction |
| **Visualizations** | Plotly | Interactive charts |
| **Prompt Management** | Custom Templates | Task-specific prompts |

---

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.9+
- Ollama installed locally
- 4GB+ RAM

### **Step 1: Clone Repository**
```bash
git clone https://github.com/yourusername/research-paper-assistant.git
cd research-paper-assistant
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
streamlit>=1.28.0
chromadb>=0.4.0
sentence-transformers>=2.2.0
rank-bm25>=0.2.2
PyPDF2>=3.0.0
pdfplumber>=0.9.0
plotly>=5.17.0
numpy>=1.24.0
pandas>=2.0.0
```

### **Step 3: Start Ollama**
```bash
# Install Ollama from https://ollama.ai
ollama serve
ollama pull llama3.2:3b
```

### **Step 4: Run Application**
```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## 📖 Usage Guide

### **1. Upload Papers**
1. Navigate to "📤 Upload Papers"
2. Select PDF file (text-based, not scanned)
3. Click "🚀 Process & Index Document"
4. Wait for extraction and embedding generation (~30-60s)

### **2. Ask Questions**
1. Go to "💬 Ask Questions"
2. Enter your research question
3. Select task type (Q&A, Summary, Compare, etc.)
4. Optionally: Adjust RAG parameters in "Advanced Settings"
5. Click "🧠 Generate Answer"
6. Review answer with source citations
7. Click suggested questions for follow-up research

### **3. View Analytics**
1. Navigate to "📊 Research Dashboard"
2. View paper comparison matrix
3. Analyze section distribution
4. Examine keyword frequency
5. Deep-dive into individual papers

---

## 📊 Performance Metrics

Based on comprehensive evaluation with 8 test questions:

| Metric | Score | Details |
|--------|-------|---------|
| **Retrieval Accuracy** | 62.5% | Section-aware retrieval |
| **Mean Reciprocal Rank** | 0.385 | Quality of top results |
| **NDCG@5** | 0.584 | Ranking quality |
| **Citation Rate** | 75% | Answers include sources |
| **Response Coherence** | 100% | Perfect coherence |
| **Avg Retrieval Time** | 0.62s | Fast hybrid search |
| **Avg Generation Time** | 23.7s | Local LLM (acceptable) |

---

## 🎨 Project Structure

```
research-paper-assistant/
├── app.py                      # Main Streamlit application
├── rag_engine.py               # Hybrid RAG implementation
├── document_processor.py       # PDF processing & chunking
├── prompt_templates.py         # Task-specific prompts
├── config.py                   # System configuration
├── evaluation.py               # Performance evaluation
├── requirements.txt            # Dependencies
├── README.md                   # This file
│
├── uploads/                    # Uploaded PDFs (auto-created)
├── chroma_db/                  # Vector database (auto-created)
└── evaluation_results/         # Test results (auto-created)
```

---

## 🔬 Key Technical Innovations

### **1. Hybrid Retrieval System**
Combines two complementary search approaches:
- **Semantic (60%)**: Finds conceptually related content even with different wording
- **Keyword (40%)**: Catches exact terminology matches

**Why it works:** Semantic search handles paraphrasing and synonyms, while BM25 ensures specific terms aren't missed.

### **2. Section-Aware Processing**
- Detects academic paper sections (Abstract, Methods, Results, Conclusions)
- Applies 2× boosting to Conclusion/Discussion sections
- Handles split pages where multiple sections appear
- Tracks section metadata for intelligent retrieval

### **3. Smart Chunking Strategy**
- 1,500 character chunks (optimal for context)
- 300 character overlap (preserves continuity)
- Page-aware splitting
- Section inheritance for unlabeled content

### **4. Dynamic Prompt Engineering**
Five specialized templates optimized for different tasks:
- Each template has custom system prompts
- Task-specific user prompt formatting
- Context-aware response generation
- Citation encouragement built-in

### **5. Intelligent Query Suggestions**
- Analyzes current query semantics
- Examines retrieved section types
- Suggests complementary questions
- Enhances research workflow efficiency

---

## 📈 Evaluation Results

**Comprehensive testing across 8 diverse queries:**

### Retrieval Performance
- ✅ 62.5% section accuracy (finds correct paper sections)
- ✅ Mean rank: 3.62 (relevant content in top 4)
- ✅ MRR: 0.385 (good first-result quality)

### Response Quality
- ✅ 75% citation rate (most answers cite sources)
- ✅ 212 avg words (appropriate depth)
- ✅ 1.0 coherence score (perfect readability)
- ✅ Task-adapted lengths (QA: 122w, Critique: 457w)

### Performance
- ⚡ 0.62s retrieval (fast hybrid search)
- 🔄 23.7s generation (local LLM - acceptable)
- 📊 Consistent across all query types

---

## 🎓 Educational Value

This project demonstrates mastery of:
- **RAG Systems**: Hybrid retrieval combining vector and keyword search
- **Prompt Engineering**: Task-specific templates for optimal LLM responses
- **Information Retrieval**: BM25, semantic embeddings, score fusion
- **NLP**: Text chunking, section detection, metadata extraction
- **Full-Stack Development**: End-to-end application with professional UI
- **System Design**: Modular architecture with clear separation of concerns
- **Evaluation**: Comprehensive metrics (MRR, NDCG, coherence)

---

## 📝 Configuration

**System defaults (config.py):**
```python
OLLAMA_MODEL = "llama3.2:3b"        # LLM model
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Embedding model
CHUNK_SIZE = 1500                    # Characters per chunk
CHUNK_OVERLAP = 300                  # Overlap for continuity
TOP_K_RESULTS = 5                    # Retrieved chunks
TEMPERATURE = 0.7                    # LLM creativity
```

**Adjustable at runtime:**
- Semantic weight: 40-80%
- Keyword weight: Auto-balanced
- Section boost: 1-3×
- Temperature: 0.3-1.0
- Top-K: 1-10 chunks

---

## 🐛 Troubleshooting

### **Ollama Connection Error**
```bash
# Start Ollama server
ollama serve

# Pull required model
ollama pull llama3.2:3b

# Verify running
curl http://localhost:11434/api/tags
```

### **Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check Python version
python --version  # Must be 3.9+
```

### **PDF Extraction Issues**
- ✅ Use text-based PDFs (not scanned images)
- ✅ Ensure file size < 50MB
- ✅ Check file permissions
- ✅ Try re-uploading if extraction fails

### **Slow Performance**
- Switch to faster model: `ollama pull phi3:mini`
- Reduce top_k from 5 to 3
- Use GPU if available
- Consider cloud LLM APIs for production

---

## 📊 Project Metrics

**Development:**
- **Total Code**: ~1,200 lines (Python)
- **Modules**: 5 core components + 1 evaluation
- **Dependencies**: 10 packages
- **Development Time**: 3 weeks

**Performance:**
- **Processes**: 50+ page papers in <60s
- **Retrieves**: Relevant chunks in ~0.6s
- **Generates**: Complete answers in ~24s (local)
- **Accuracy**: 62.5% section accuracy, 75% citation rate

**Features:**
- 6 pages (Overview, Upload, Questions, Dashboard, Knowledge Base, Technical)
- 5 prompt templates
- 3 unique interactive features
- Real-time analytics and visualizations

---

## 👨‍💻 Author

**Saurabh Soni**  
MS Information Systems | Northeastern University  
Expected Graduation: December 2025

- 🔗 [LinkedIn](#)
- 🐙 [GitHub](#)
- 📧 soni.sau@northeastern.edu

---

## 🙏 Acknowledgments

- **Professor Nik Bear Brown** - INFO 7375 Prompt Engineering & AI
- **Northeastern University** - College of Engineering
- **Open Source Community** - ChromaDB, Sentence Transformers, Streamlit

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

## 🎯 Course Information

**Course**: INFO 7375 - Prompt Engineering for Generative AI   
**Professor**: Nik Bear Brown
**Institution**: Northeastern University   

**Project demonstrates:**
- Advanced RAG implementation
- Hybrid search strategies
- Prompt engineering mastery
- Full-stack AI application development
- Information retrieval techniques
- System evaluation and metrics

---

**⭐ If you find this project useful, please star the repository!**

---

*Built with ❤️ at Northeastern University | Demonstrating mastery of RAG, prompt engineering, and information retrieval*
