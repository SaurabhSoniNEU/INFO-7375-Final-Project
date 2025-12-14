"""
Research Paper Assistant - Enhanced Portfolio Web Interface
With Interactive Dashboard, Adjustable Parameters, and Smart Suggestions
"""

import streamlit as st
import json
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Import your RAG components
try:
    from rag_engine import RAGEngine
    from document_processor import DocumentProcessor
    from config import OLLAMA_MODEL, TOP_K_RESULTS, UPLOAD_DIR
except ImportError:
    st.error("⚠️ Missing dependencies. Install: pip install chromadb sentence-transformers rank-bm25 pypdf2 pdfplumber plotly")
    st.stop()

# Page config
st.set_page_config(
    page_title="Research Paper Assistant - Saurabh Soni",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components (cached for performance)
@st.cache_resource
def get_rag_engine():
    """Initialize RAG engine (singleton)"""
    return RAGEngine()

@st.cache_resource
def get_doc_processor():
    """Initialize document processor (singleton)"""
    return DocumentProcessor()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #003366;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: #000000;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #003366;
        margin: 1rem 0;
    }
    .source-card {
        background: #000000;
        border: 1px solid #ddd;
        border-left: 4px solid #0066CC;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .answer-box {
        background: #000000;
        border: 2px solid #0066CC;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }
    .info-box {
        background: #f0f4f8;
        border-left: 4px solid #0066CC;
        padding: 1rem;
        border-radius: 5px;
    }
    .suggestion-btn {
        background: #f0f0f0;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        display: inline-block;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📚 Research Paper Assistant</h1>', unsafe_allow_html=True)
st.markdown("### *AI-Powered Research Analysis with Hybrid RAG Technology*")
st.markdown("**By Saurabh Soni** | Northeastern University | INFO 7375")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    page = st.radio(
        "Select Section:",
        ["🏠 Overview", "📤 Upload Papers", "💬 Ask Questions", "📊 Research Dashboard", "🗄️ Knowledge Base", "🔬 Technical Details"]
    )
    
    st.markdown("---")
    st.markdown("### 📈 System Stats")
    
    # Get real-time stats
    try:
        rag_engine = get_rag_engine()
        stats = rag_engine.get_collection_stats()
        
        st.metric("Documents", stats['unique_documents'])
        st.metric("Chunks", stats['total_chunks'])
        st.metric("Model", OLLAMA_MODEL.split(':')[0])
        
        # Connection status
        if rag_engine.check_ollama_connection():
            st.success("🟢 Ollama Running")
        else:
            st.error("🔴 Ollama Offline")
    except Exception as e:
        st.metric("Status", "Initializing...")
    
    st.markdown("---")
    st.markdown("### 🔗 Quick Links")
    st.markdown("- [📄 GitHub Repo](#)")
    st.markdown("- [💻 Source Code](#)")
    st.markdown("- [🎥 Demo Video](#)")

# OVERVIEW PAGE
if page == "🏠 Overview":
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 What This System Does")
        
        st.info("""
        **Problem:** Researchers spend hours manually reading and synthesizing information 
        from multiple academic papers. Finding specific information across documents is tedious.
        
        **Solution:** An AI-powered research assistant using **hybrid RAG** (Retrieval-Augmented 
        Generation) that combines semantic search with keyword matching for optimal retrieval.
        
        **Result:** Fast, accurate answers with source citations from your research collection.
        """)
        
        st.markdown("### 🎨 Core Technologies")
        
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown("#### 📚 RAG Implementation")
            st.markdown("""
            **Vector Database:**
            - ChromaDB for semantic search
            - Sentence Transformers embeddings
            - Hybrid search (vector + BM25)
            - Section-aware chunking
            
            **Improves accuracy and relevance**
            """)
        
        with tech_col2:
            st.markdown("#### 🎯 Prompt Engineering")
            st.markdown("""
            **5 Specialized Templates:**
            - Question & Answer using Chain of Thought
            - Summarization using Few Shots
            - Comparison Analysis using Structured Reasoning
            - Information Extraction using Role Based
            - Critical Critique using Persona Pattern
            
            **Task-optimized responses**
            """)
        
        st.markdown("### 🏆 Key Features")
        
        feature_cols = st.columns(4)
        
        with feature_cols[0]:
            st.success("**Hybrid Search**\n\n60% Semantic + 40% Keywords")
        
        with feature_cols[1]:
            st.success("**Section Detection**\n\nAuto-identifies paper parts")
        
        with feature_cols[2]:
            st.success("**Multi-Paper Analysis**\n\nCompare documents")
        
        with feature_cols[3]:
            st.success("**Source Citations**\n\nPage & section tracking")
    
    with col2:
        st.markdown("## 🔧 Tech Stack")
        
        st.markdown("""
        ### LLM & Embeddings
        - **Ollama**: Local LLM
        - **Sentence Transformers**: Embeddings
        - **ChromaDB**: Vector store
        
        ### Search & Retrieval
        - **Hybrid Search**: Vector + BM25
        - **Smart Chunking**: 1500 chars
        - **Section Boosting**: 2× priority
        
        ### Processing
        - **pdfplumber**: Primary extraction
        - **PyPDF2**: Backup extraction
        - **Metadata**: Page & section tracking
        """)
    
    st.markdown("---")
    
    # How it works
    st.markdown("## 🔄 How It Works")
    
    steps_cols = st.columns(4)
    
    with steps_cols[0]:
        st.markdown("### 1️⃣ Upload")
        st.info("Upload PDF → Extract text → Detect sections")
    
    with steps_cols[1]:
        st.markdown("### 2️⃣ Process")
        st.warning("Chunk text → Generate embeddings → Build indexes")
    
    with steps_cols[2]:
        st.markdown("### 3️⃣ Query")
        st.success("Ask question → Hybrid search → Retrieve top-K")
    
    with steps_cols[3]:
        st.markdown("### 4️⃣ Generate")
        st.error("Prompt LLM → Generate answer → Return with citations")

# UPLOAD PAGE
elif page == "📤 Upload Papers":
    
    st.markdown("## 📤 Upload Research Papers")
    
    st.info("""
    Upload PDF research papers to build your intelligent knowledge base.
    The system will extract text, detect sections, create embeddings, and enable AI-powered search.
    """)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📄 Select PDF File")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF research paper",
            type=['pdf'],
            help="Upload text-based PDFs (not scanned images)"
        )
        
        if uploaded_file is not None:
            st.success(f"✅ File selected: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
            
            if st.button("🚀 Process & Index Document", type="primary"):
                
                # Save uploaded file
                UPLOAD_DIR.mkdir(exist_ok=True)
                file_path = UPLOAD_DIR / uploaded_file.name
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Extract text
                    status_text.text("📄 Extracting text from PDF...")
                    progress_bar.progress(25)
                    
                    doc_processor = get_doc_processor()
                    chunks, metadata = doc_processor.process_document(str(file_path))
                    
                    st.info(f"""
                    **Extraction Complete:**
                    - Pages: {metadata['pages']}
                    - Characters: {metadata['total_characters']:,}
                    - Chunks: {len(chunks)}
                    - Method: {metadata['extraction_method']}
                    """)
                    
                    # Step 2: Generate embeddings
                    status_text.text("🧠 Generating embeddings and building index...")
                    progress_bar.progress(50)
                    
                    rag_engine = get_rag_engine()
                    result = rag_engine.add_documents(chunks)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")
                    
                    if result['status'] == 'success':
                        st.success(f"""
                        ### ✅ Successfully Processed!
                        
                        **Document:** {uploaded_file.name}
                        - **{result['chunks_added']} chunks** added to knowledge base
                        - **Embeddings** generated with Sentence Transformers
                        - **BM25 index** built for keyword search
                        - **Hybrid search** ready for queries
                        
                        **Next Step:** Go to "💬 Ask Questions" to query your documents!
                        """)
                        
                        time.sleep(2)
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {result.get('message', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    with st.expander("🐛 Debug Information"):
                        import traceback
                        st.code(traceback.format_exc())
    
    with col2:
        st.markdown("### 💡 Upload Tips")
        
        st.markdown("""
        <div class="metric-card">
            <strong>✅ Best Results:</strong><br><br>
            • Text-based PDFs<br>
            • Academic/research papers<br>
            • Clear, readable text<br>
            • Under 50 MB<br><br>
            <strong>❌ Avoid:</strong><br><br>
            • Scanned images only<br>
            • Handwritten documents<br>
            • Protected/encrypted PDFs<br>
            • Very poor quality scans
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("### 📊 Processing Pipeline")

    # Create 9 columns (5 for steps, 4 for arrows)
    cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])

    # Step 1: Text Extraction
    with cols[0]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.2rem; border-radius: 8px; text-align: center; color: white; height: 100px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-weight: bold; margin-bottom: 0.3rem;">📄 Text Extraction</div>
            <div style="font-size: 0.75rem; opacity: 0.9;">pdfplumber → PyPDF2</div>
        </div>
        """, unsafe_allow_html=True)

    # Arrow 1
    with cols[1]:
        st.markdown("<div style='text-align: center; font-size: 2rem; color: #666; line-height: 100px;'>→</div>", unsafe_allow_html=True)

    # Step 2: Section Detection
    with cols[2]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.2rem; border-radius: 8px; text-align: center; color: white; height: 100px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-weight: bold; margin-bottom: 0.3rem;">🎯 Section Detection</div>
            <div style="font-size: 0.75rem; opacity: 0.9;">Abstract, Methods</div>
        </div>
        """, unsafe_allow_html=True)

    # Arrow 2
    with cols[3]:
        st.markdown("<div style='text-align: center; font-size: 2rem; color: #666; line-height: 100px;'>→</div>", unsafe_allow_html=True)

    # Step 3: Smart Chunking
    with cols[4]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.2rem; border-radius: 8px; text-align: center; color: white; height: 100px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-weight: bold; margin-bottom: 0.3rem;">✂️ Smart Chunking</div>
            <div style="font-size: 0.75rem; opacity: 0.9;">1,500 chars</div>
        </div>
        """, unsafe_allow_html=True)

    # Arrow 3
    with cols[5]:
        st.markdown("<div style='text-align: center; font-size: 2rem; color: #666; line-height: 100px;'>→</div>", unsafe_allow_html=True)

    # Step 4: Embeddings
    with cols[6]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                    padding: 1.2rem; border-radius: 8px; text-align: center; color: white; height: 100px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-weight: bold; margin-bottom: 0.3rem;">🧠 Embeddings</div>
            <div style="font-size: 0.75rem; opacity: 0.9;">384-dim vectors</div>
        </div>
        """, unsafe_allow_html=True)

    # Arrow 4
    with cols[7]:
        st.markdown("<div style='text-align: center; font-size: 2rem; color: #666; line-height: 100px;'>→</div>", unsafe_allow_html=True)

    # Step 5: Indexing
    with cols[8]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%); 
                    padding: 1.2rem; border-radius: 8px; text-align: center; color: white; height: 100px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-weight: bold; margin-bottom: 0.3rem;">📇 Indexing</div>
            <div style="font-size: 0.75rem; opacity: 0.9;">ChromaDB + BM25</div>
        </div>
        """, unsafe_allow_html=True)

# ASK QUESTIONS PAGE - WITH ADJUSTABLE PARAMETERS
elif page == "💬 Ask Questions":
    
    st.markdown("## 💬 Ask Intelligent Questions")
    
    # Initialize RAG engine
    rag_engine = get_rag_engine()
    
    # Check connection
    if not rag_engine.check_ollama_connection():
        st.error("""
        ❌ **Cannot connect to Ollama!**
        
        Please start Ollama:
        ```bash
        ollama serve
        ollama run llama3.2:3b
        ```
        """)
        st.stop()
    
    # Check if documents loaded
    stats = rag_engine.get_collection_stats()
    
    if stats['total_chunks'] == 0:
        st.warning("""
        ⚠️ **No documents in knowledge base yet!**
        
        Please upload research papers first:
        1. Go to "📤 Upload Papers"
        2. Upload a PDF
        3. Wait for processing
        4. Return here to ask questions
        """)
        st.stop()
    
    st.success(f"✅ Knowledge base ready: **{stats['unique_documents']} documents**, **{stats['total_chunks']} chunks**")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🔍 Enter Your Question")
        
        question = st.text_area(
            "Question:",
            placeholder="What are the main conclusions of the study?",
            height=120,
            help="Ask anything about your uploaded research papers"
        )
        
        col_task, col_k = st.columns(2)
        
        with col_task:
            task_type = st.selectbox(
                "Task Type:",
                options=["qa", "summary", "compare", "extract", "critique"],
                format_func=lambda x: {
                    "qa": "❓ Question & Answer (Chain of Thought)",
                    "summary": "📝 Summarize (Few-Shot Learning)",
                    "compare": "⚖️ Compare Papers (Structured Reasoning)",
                    "extract": "📋 Extract Information (Role-Based)",
                    "critique": "🔍 Critical Analysis (Persona Pattern)"
                }[x],
                help="Each task uses a different prompting pattern"
            )
        
        with col_k:
            top_k = st.slider(
                "Number of Sources:",
                min_value=1,
                max_value=10,
                value=5,
                help="How many document chunks to retrieve and use"
            )
        
        # Source filter (if multiple documents)
        if stats['unique_documents'] > 1:
            source_filter = st.selectbox(
                "Filter by Document (optional):",
                options=["All Documents"] + stats['documents']
            )
            source_filter = None if source_filter == "All Documents" else source_filter
        else:
            source_filter = None
        
        # ⭐ NEW FEATURE: Adjustable RAG Parameters
        with st.expander("🎛️ Advanced RAG Parameters (Adjustable)", expanded=False):
            st.markdown("**Fine-tune retrieval behavior:**")
            
            col_sem, col_key = st.columns(2)
            
            with col_sem:
                semantic_weight = st.slider(
                    "Semantic Weight:",
                    min_value=0.4,
                    max_value=0.8,
                    value=0.6,
                    step=0.05,
                    help="How much to rely on semantic similarity (meaning-based search)"
                )
            
            with col_key:
                keyword_weight = 1.0 - semantic_weight
                st.metric("Keyword Weight", f"{keyword_weight:.2f}")
                st.caption("Automatically balanced (1 - semantic)")
            
            section_boost = st.slider(
                "Section Boosting:",
                min_value=1.0,
                max_value=3.0,
                value=2.0,
                step=0.5,
                help="Boost priority of Conclusion/Discussion sections"
            )
            
            temperature = st.slider(
                "Temperature (Creativity):",
                min_value=0.3,
                max_value=1.0,
                value=0.7,
                step=0.1,
                help="Higher = more creative, Lower = more factual"
            )
            
            st.info(f"""
            **Current Settings:**
            - Semantic: {semantic_weight*100:.0f}% | Keywords: {keyword_weight*100:.0f}%
            - Section Boost: {section_boost}×
            - Temperature: {temperature}
            """)
        
        # Generate button
        if st.button("🧠 Generate Answer", type="primary"):
            
            if not question.strip():
                st.warning("⚠️ Please enter a question first!")
            else:
                with st.spinner("🔍 Searching knowledge base and generating answer..."):
                    
                    start_time = time.time()
                    
                    # Generate response using RAG with custom parameters
                    result = rag_engine.generate_response(
                        query=question,
                        task_type=task_type,
                        top_k=top_k,
                        source_filter=source_filter,
                        semantic_weight=semantic_weight,
                        keyword_weight=keyword_weight,
                        section_boost=section_boost,
                        temperature=temperature
                    )
                    
                    total_time = time.time() - start_time
                    
                    if result['status'] == 'success':
                        # Show answer
                        st.markdown("### 💡 AI-Generated Answer")
                        
                        st.markdown(f"""
                        <div class="answer-box">
                            {result['response']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.caption(f"⏱️ Generated in {total_time:.2f}s using {result['num_sources']} sources | Model: {OLLAMA_MODEL}")
                        
                        # ⭐ NEW FEATURE: Smart Query Suggestions
                        if result.get('suggestions'):
                            st.markdown("---")
                            st.markdown("### 💡 Related Questions You Might Ask:")
                            
                            cols = st.columns(2)
                            for idx, suggestion in enumerate(result['suggestions']):
                                col_idx = idx % 2
                                with cols[col_idx]:
                                    if st.button(f"💬 {suggestion}", key=f"sugg_{idx}", use_container_width=True):
                                        st.session_state['suggested_question'] = suggestion
                                        st.rerun()
                        
                        st.markdown("---")
                        
                        # Show sources
                        st.markdown("### 📚 Sources Used")
                        
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"📄 Source {i}: {source['source']} - Page {source['page']} ({source['section']}) | Relevance: {source['score']:.3f}"):
                                st.markdown("**Chunk Preview:**")
                                st.text(source['preview'])
                                
                                st.markdown("**Metadata:**")
                                st.json({
                                    "Source": source['source'],
                                    "Page": source['page'],
                                    "Section": source['section'],
                                    "Chunk ID": source['chunk_id'],
                                    "Relevance Score": source['score']
                                })
                    
                    else:
                        st.error(f"❌ {result['response']}")
    
    with col2:
        st.markdown("### 💡 Example Questions")
        
        st.markdown("""
        <div class="metric-card">
            <strong>Try asking:</strong><br><br>
            📌 "What is the main research question?"<br>
            📌 "Summarize the methodology used"<br>
            📌 "What are the key findings?"<br>
            📌 "What are the study limitations?"<br>
            📌 "Compare approaches in diff. papers"<br>
            📌 "Extract participant demographics"<br>
            📌 "Critique the experimental design"
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🎯 Task Types Explained")
        
        st.markdown("""
        **❓ Q&A:** Direct answers with citations
        
        **📝 Summary:** Concise content overview
        
        **⚖️ Compare:** Multi-paper analysis
        
        **📋 Extract:** Specific information retrieval
        
        **🔍 Critique:** Critical evaluation of methodology
        """)
        
        # Quick examples
        st.markdown("### ⚡ Quick Examples")
        
        examples = [
            ("Main findings?", "qa"),
            ("Summarize", "summary"),
            ("Limitations?", "extract")
        ]
        
        for q, t in examples:
            if st.button(f"📝 {q}", key=f"ex_{q}"):
                st.session_state['example_q'] = q
                st.session_state['example_t'] = t

# ⭐ NEW PAGE: RESEARCH DASHBOARD
elif page == "📊 Research Dashboard":
    
    st.markdown("## 📊 Interactive Research Dashboard")
    
    rag_engine = get_rag_engine()
    stats = rag_engine.get_collection_stats()
    
    if stats['total_chunks'] == 0:
        st.warning("""
        ⚠️ **No documents uploaded yet!**
        
        Upload research papers first to see the dashboard.
        """)
        st.stop()
    
    st.success(f"**Analyzing {stats['unique_documents']} papers** with {stats['total_chunks']} chunks")
    st.markdown("---")
    
    # Extract paper metrics
    with st.spinner("Analyzing papers..."):
        metrics = rag_engine.extract_paper_metrics()
    
    # Section 1: Overview Metrics
    st.markdown("### 📈 Overview Metrics")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric("Total Papers", stats['unique_documents'])
    
    with metric_cols[1]:
        st.metric("Total Chunks", stats['total_chunks'])
    
    with metric_cols[2]:
        avg_chunks = stats['total_chunks'] / max(stats['unique_documents'], 1)
        st.metric("Avg Chunks/Paper", f"{avg_chunks:.0f}")
    
    with metric_cols[3]:
        st.metric("Search Mode", "Hybrid")
    
    st.markdown("---")
    
    # Section 2: Paper Comparison Table
    if len(metrics) > 1:
        st.markdown("### 📊 Paper Comparison Matrix")
        
        comparison_data = []
        for paper, data in metrics.items():
            comparison_data.append({
                "Paper": paper[:40] + "..." if len(paper) > 40 else paper,
                "Chunks": data['total_chunks'],
                "Avg Length": data['avg_chunk_length'],
                "Has Methods": "✅" if data['has_methodology'] else "❌",
                "Has Results": "✅" if data['has_results'] else "❌",
                "Has Conclusion": "✅" if data['has_conclusion'] else "❌",
                "Top Keywords": ", ".join(data['top_keywords'][:3])
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True, hide_index=True)
        
        st.markdown("---")
    
    # Section 3: Section Distribution
    st.markdown("### 📑 Content Distribution Across Papers")
    
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        # Section distribution pie chart
        section_data = []
        for section, count in stats['sections'].items():
            section_data.append({
                "Section": section.replace('_', ' ').title(),
                "Count": count
            })
        
        df_sections = pd.DataFrame(section_data)
        
        fig_pie = px.pie(
            df_sections,
            values='Count',
            names='Section',
            title='Section Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_dist2:
        # Top keywords across all papers
        all_keywords = []
        for paper, data in metrics.items():
            all_keywords.extend(data['top_keywords'][:5])
        
        from collections import Counter
        keyword_counts = Counter(all_keywords).most_common(10)
        
        df_keywords = pd.DataFrame(keyword_counts, columns=['Keyword', 'Frequency'])
        
        fig_bar = px.bar(
            df_keywords,
            x='Keyword',
            y='Frequency',
            title='Most Common Keywords',
            color='Frequency',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Section 4: Individual Paper Details
    st.markdown("### 📄 Detailed Paper Analysis")
    
    selected_paper = st.selectbox(
        "Select a paper to analyze:",
        options=list(metrics.keys())
    )
    
    if selected_paper:
        paper_data = metrics[selected_paper]
        
        col_detail1, col_detail2 = st.columns([2, 1])
        
        with col_detail1:
            st.markdown(f"#### 📋 {selected_paper}")
            
            st.info(f"""
            **Paper Statistics:**
            - Total Chunks: {paper_data['total_chunks']}
            - Average Chunk Length: {paper_data['avg_chunk_length']} characters
            - Methodology Section: {'✅ Present' if paper_data['has_methodology'] else '❌ Missing'}
            - Results Section: {'✅ Present' if paper_data['has_results'] else '❌ Missing'}
            - Conclusion Section: {'✅ Present' if paper_data['has_conclusion'] else '❌ Missing'}
            """)
            
            # Section breakdown
            section_breakdown = []
            for section, count in paper_data['sections'].items():
                section_breakdown.append({
                    "Section": section.replace('_', ' ').title(),
                    "Chunks": count,
                    "Percentage": f"{(count/paper_data['total_chunks']*100):.1f}%"
                })
            
            df_breakdown = pd.DataFrame(section_breakdown)
            st.dataframe(df_breakdown, use_container_width=True, hide_index=True)
        
        with col_detail2:
            st.markdown("#### 🔑 Top Keywords")
            
            for i, keyword in enumerate(paper_data['top_keywords'], 1):
                st.markdown(f"{i}. **{keyword}**")

# KNOWLEDGE BASE PAGE
elif page == "🗄️ Knowledge Base":
    
    st.markdown("## 📊 Knowledge Base Management")
    
    rag_engine = get_rag_engine()
    stats = rag_engine.get_collection_stats()
    
    # Top metrics
    st.markdown("### 📈 Database Statistics")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        st.metric("Total Documents", stats['unique_documents'], "Research papers")
    
    with metric_cols[1]:
        st.metric("Total Chunks", stats['total_chunks'], "Searchable segments")
    
    with metric_cols[2]:
        avg_chunks = stats['total_chunks'] / max(stats['unique_documents'], 1)
        st.metric("Avg Chunks/Doc", f"{avg_chunks:.0f}", "Per document")
    
    with metric_cols[3]:
        st.metric("Search Type", "Hybrid", "Semantic + BM25")
    
    st.markdown("---")
    
    if stats['total_chunks'] > 0:
        
        # Documents list
        st.markdown("### 📚 Loaded Documents")
        
        for i, doc in enumerate(stats['documents'], 1):
            st.markdown(f"""
            <div class="source-card">
                <strong>{i}. {doc}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Section breakdown
        st.markdown("### 📑 Content Distribution by Section")
        
        if stats['sections']:
            # Create dataframe
            section_data = []
            for section, count in sorted(stats['sections'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_chunks'] * 100)
                section_data.append({
                    "Section": section.replace('_', ' ').title(),
                    "Chunks": count,
                    "Percentage": f"{percentage:.1f}%"
                })
            
            df = pd.DataFrame(section_data)
            
            st.dataframe(
                df.style.background_gradient(subset=['Chunks'], cmap='Blues'),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("💡 Conclusion and Discussion sections get 2× boosting in search relevance")
        
        st.markdown("---")
        
        # Management controls
        st.markdown("### 🔧 Database Management")
        
        col_refresh, col_clear = st.columns(2)
        
        with col_refresh:
            if st.button("🔄 Refresh Statistics", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        
        with col_clear:
            if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
                if 'confirm_clear' not in st.session_state:
                    st.session_state.confirm_clear = False
                
                if st.session_state.confirm_clear:
                    result = rag_engine.clear_collection()
                    if result['status'] == 'success':
                        st.success("✅ Knowledge base cleared!")
                        st.session_state.confirm_clear = False
                        st.cache_resource.clear()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result['message']}")
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm deletion of ALL documents")
    
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("""
            ### 📭 Knowledge Base Empty
            
            No documents uploaded yet. Get started:
            
            1. Go to "📤 Upload Papers" section
            2. Select your PDF file
            3. Click "🚀 Process & Index Document"
            4. Wait for processing (~30-60 seconds)
            5. Return here to see your documents
            6. Go to "💬 Ask Questions" to query!
            """)
        
        with col2:
            st.markdown("### 💡 Upload Tips")
            
            st.markdown("""
            <div class="metric-card">
                <strong>Supported Formats:</strong><br><br>
                ✅ PDF (text-based)<br>
                ✅ Academic papers<br>
                ✅ Research articles<br>
                ✅ Technical reports<br><br>
                <strong>File Size:</strong><br><br>
                • Recommended: < 10 MB<br>
                • Maximum: 50 MB<br>
                • Typical paper: 2-5 MB<br><br>
                <strong>Quality Matters:</strong><br><br>
                • Clear, readable text<br>
                • Proper formatting<br>
                • Standard fonts<br>
                • Good resolution
            </div>
            """, unsafe_allow_html=True)

# TECHNICAL DETAILS PAGE
elif page == "🔬 Technical Details":
    
    st.markdown("## 🔬 Technical Implementation")
    
    tech_tabs = st.tabs(["🔍 RAG Architecture", "📐 Chunking Strategy", "🎯 Prompt Templates", "⚙️ Configuration"])
    
    with tech_tabs[0]:
        st.markdown("### 🔍 Hybrid RAG Architecture")
        
        st.markdown("""
        **Two-Stage Retrieval Process:**
        
        **Stage 1: Semantic Search (ChromaDB)**
        - Encode query using Sentence Transformers (all-MiniLM-L6-v2)
        - Find semantically similar chunks via cosine similarity
        - Returns top 2×K candidates for better coverage
        - Section-aware: Boosts conclusions and discussions (2×)
        
        **Stage 2: Keyword Search (BM25)**
        - Tokenize query and all documents
        - Calculate term frequency-inverse document frequency
        - Score based on keyword overlap and rarity
        - Captures exact terminology matches
        
        **Stage 3: Score Fusion**
        - Combine: **60% semantic + 40% keyword** (adjustable!)
        - Apply section boosting multipliers
        - Sort by final combined score
        - Return top K results
        """)
        
        st.code("""
# Hybrid Search Implementation (Now with adjustable parameters!)
def hybrid_search(query, semantic_weight=0.6, keyword_weight=0.4, section_boost=2.0):
    # Semantic similarity
    query_embedding = encoder.encode(query)
    semantic_results = chromadb.query(query_embedding, n=top_k*2)
    
    # Keyword matching
    bm25_scores = bm25_index.get_scores(query.split())
    
    # Combine scores with custom weights
    for doc in results:
        semantic = semantic_results[doc.id]
        keyword = bm25_scores[doc.id]
        
        # Section boost
        boost = section_boost if doc.section in ['conclusion', 'discussion'] else 1.0
        
        final_score = (semantic_weight * semantic + keyword_weight * keyword) * boost
    
    return sorted(results, key=lambda x: x.final_score)[:top_k]
        """, language="python")
        
        st.info("**Why Hybrid?** Semantic search finds conceptually related content even with different wording. BM25 finds exact keyword matches. Combined = comprehensive retrieval!")
    
    with tech_tabs[1]:
        st.markdown("### 📐 Smart Chunking Strategy")
        
        st.markdown("""
        **Section-Aware Document Chunking:**
        
        **Process Flow:**
        1. **Extract** text with page markers `[PAGE n]`
        2. **Detect** sections using keyword patterns
           - "Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion"
        3. **Split** into paragraphs (preserve structure)
        4. **Chunk** with size limit and overlap
        5. **Annotate** with metadata (page, section, length)
        
        **Parameters:**
        - Chunk size: **1,500 characters** (optimal for context)
        - Overlap: **300 characters** (continuity preservation)
        - Min paragraph: **50 characters** (filter noise)
        
        **Section Detection Algorithm:**
        """)
        
        st.code("""
def detect_section(text):
    text_lower = text.lower()[:500]
    
    sections = {
        'abstract': ['abstract'],
        'introduction': ['introduction', '1. introduction'],
        'method': ['method', 'methodology', 'procedure'],
        'results': ['results', 'findings'],
        'discussion': ['discussion', 'conclusion'],
        'references': ['references', 'bibliography']
    }
    
    for section_name, keywords in sections.items():
        for keyword in keywords:
            if keyword in text_lower:
                return section_name
    
    return 'body'  # Default
        """, language="python")
        
        st.markdown("""
        **Chunk Metadata Structure:**
        """)
        
        st.code("""
DocumentChunk(
    content="Full paragraph text...",
    page_number=5,
    chunk_id=12,
    source="smith_2024.pdf",
    metadata={
        "section": "results",
        "length": 1450,
        "has_conclusion": False
    }
)
        """, language="python")
    
    with tech_tabs[2]:
        st.markdown("### 🎯 Prompt Engineering Patterns")
        
        st.info("""
        **Five Different Prompting Patterns Demonstrated:**
        
        Each task type uses a strategically selected prompting pattern from advanced prompt engineering:
        """)
        
        st.markdown("**📚 Pattern Overview:**")
        
        pattern_overview = {
            "❓ Q&A": {
                "pattern": "Chain of Thought (CoT)",
                "description": "Forces step-by-step reasoning: Identify sources → Extract facts → Synthesize answer"
            },
            "📝 Summary": {
                "pattern": "Few-Shot Learning",
                "description": "Provides 3 high-quality examples before requesting new summary"
            },
            "⚖️ Compare": {
                "pattern": "Structured Reasoning",
                "description": "4-step framework: Identify → Extract → Organize → Synthesize"
            },
            "📋 Extract": {
                "pattern": "Role-Based Prompting",
                "description": "Assigns role: 'Expert research librarian with 15 years experience'"
            },
            "🔍 Critique": {
                "pattern": "Persona Pattern",
                "description": "Creates Dr. Sarah Chen: MIT PhD, 20+ years peer review experience"
            }
        }
        
        for task, info in pattern_overview.items():
            with st.expander(f"{task} - {info['pattern']}"):
                st.markdown(f"**Pattern Type:** {info['pattern']}")
                st.markdown(f"**How it works:** {info['description']}")
                st.markdown("---")
                
                # Show actual template
                template_examples = {
                    "❓ Q&A": """
**System Prompt:**
You are a helpful research assistant. Use step-by-step reasoning:
1. Identify which sources contain relevant information
2. Extract the key facts from those sources
3. Synthesize a clear answer with proper citations

**User Prompt:**
Let's solve this step-by-step:
Step 1: Identify relevant sources
Step 2: Extract key information  
Step 3: Provide answer with citations
                    """,
                    "📝 Summary": """
**System Prompt with Examples:**
Here are examples of high-quality research summaries:

Example 1:
Research: "Neural network optimization using Adam..."
Summary: "This paper investigates Adam optimizer, finding 23% faster 
convergence. Key contribution: adaptive learning rates improve efficiency."

Example 2:
Research: "Transformer attention mechanisms..."
Summary: "Authors analyze self-attention, showing head specialization. 
Head pruning reduces parameters 40% while maintaining 98% accuracy."

Now create a similar summary...
                    """,
                    "⚖️ Compare": """
**Structured Framework:**
STEP 1: IDENTIFY key aspects being compared
STEP 2: EXTRACT relevant information from each source
STEP 3: ORGANIZE similarities and differences
STEP 4: SYNTHESIZE into coherent comparison

Must include: Similarities, Differences, Complementary insights, 
Contradictions, Strengths/weaknesses
                    """,
                    "📋 Extract": """
**Role Assignment:**
"You are an expert research librarian and data extraction specialist 
with 15 years of experience in academic research.

Your expertise includes:
- Identifying specific information from complex texts
- Organizing information in structured formats
- Verifying accuracy of extracted data
- Providing proper source attribution"
                    """,
                    "🔍 Critique": """
**Persona Definition:**
"You are Dr. Sarah Chen, a senior research methodologist and peer 
reviewer for top-tier journals.

Background:
- Ph.D. in Research Methodology from MIT
- 20+ years reviewing CS and AI papers
- Published 50+ papers on experimental design
- Known for constructive, balanced critiques

Reviewing Style:
- Thorough but fair
- Evidence-based with specific examples
- Constructive suggestions for improvement
- Balanced acknowledgment of strengths and weaknesses"
                    """
                }
                
                if task in template_examples:
                    st.code(template_examples[task].strip(), language="text")
                    
                    if task == "📝 Summary":
                        st.success("**Why Few-Shot?** Examples teach the LLM the desired format and depth, dramatically improving consistency.")
                    elif task == "🔍 Critique":
                        st.success("**Why Persona?** Dr. Chen's credentials and style guide produce professional, nuanced critiques rather than generic feedback.")
                    elif task == "❓ Q&A":
                        st.success("**Why CoT?** Breaking reasoning into steps improves accuracy and reduces hallucination.")
    
    with tech_tabs[3]:
        st.markdown("### ⚙️ System Configuration")
        
        config_data = {
            "Component": [
                "LLM Model",
                "Embedding Model",
                "Vector Database",
                "Keyword Search",
                "Chunk Size",
                "Chunk Overlap",
                "Top-K Results",
                "Temperature",
                "Max Tokens",
                "Semantic Weight",
                "Keyword Weight",
                "Section Boost"
            ],
            "Value": [
                OLLAMA_MODEL,
                "all-MiniLM-L6-v2",
                "ChromaDB (persistent)",
                "BM25 Okapi",
                "1,500 characters",
                "300 characters",
                str(TOP_K_RESULTS),
                "0.3-1.0 (adjustable)",
                "1,000",
                "40-80% (adjustable)",
                "20-60% (adjustable)",
                "1-3× (adjustable)"
            ],
            "Purpose": [
                "Local LLM inference",
                "384-dim semantic vectors",
                "Persistent vector storage",
                "Term frequency matching",
                "Optimal context window",
                "Preserve continuity",
                "Retrieved chunks",
                "Response creativity",
                "Response length limit",
                "Semantic relevance",
                "Exact term matches",
                "Prioritize key sections"
            ]
        }
        
        df = pd.DataFrame(config_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        st.markdown("### 📊 Performance Characteristics")
        
        perf_cols = st.columns(3)
        
        with perf_cols[0]:
            st.metric("Retrieval Speed", "~0.5s", "Hybrid search")
        
        with perf_cols[1]:
            st.metric("Generation Speed", "~3-5s", f"{OLLAMA_MODEL}")
        
        with perf_cols[2]:
            st.metric("Total Latency", "~4-6s", "End-to-end")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Research Paper Assistant with Hybrid RAG</strong></p>
    <p>Portfolio Project | INFO 7375 | Saurabh Soni | December 2025</p>
    <p><em>Demonstrating mastery of RAG, prompt engineering, and information retrieval</em></p>
    <p style='font-size: 0.85rem; margin-top: 1rem;'>⭐ New Features: Interactive Dashboard | Adjustable RAG Parameters | Smart Query Suggestions</p>
</div>
""", unsafe_allow_html=True)