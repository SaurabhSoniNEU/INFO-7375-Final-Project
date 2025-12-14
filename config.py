"""Configuration file for the Research Assistant"""

import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Create directories if they don't exist
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# Model Configuration - USE FASTER MODEL
OLLAMA_MODEL = "llama3.2:3b"  # Smaller, faster model
# Alternative fast models:
# OLLAMA_MODEL = "phi3:mini"  # Very fast, 3.8B parameters
# OLLAMA_MODEL = "gemma2:2b"  # Fast and efficient
OLLAMA_BASE_URL = "http://localhost:11434"

# Embedding Model (Free from Hugging Face)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Configuration - IMPROVED FOR BETTER RETRIEVAL
CHUNK_SIZE = 1500  # Better context
CHUNK_OVERLAP = 300  # Better continuity
TOP_K_RESULTS = 5  # Get more results for better coverage

# Prompt Configuration
MAX_CONTEXT_LENGTH = 3000
TEMPERATURE = 0.7
MAX_TOKENS = 1000