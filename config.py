"""Phase 2A:  Centralized Configuration"""
from pathlib import Path

# LLM Configuration
MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TEMPERATURE = 0.3
MAX_TOKENS = 512

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG Configuration
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

# Paths
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "sample_docs"
CHROMA_PATH = BASE_DIR / "data" / "chroma_data"
LOGS_PATH = BASE_DIR / "logs"

# Ensure directories exist
for path in [DATA_PATH, CHROMA_PATH, LOGS_PATH]: 
    path.mkdir(parents=True, exist_ok=True)
