"""Phase 2B:  Production Configuration - OPTIMIZED"""
from pathlib import Path
import os

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
MODEL_NAME = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_HEALTH_URL = "http://localhost:11434/api/tags"

# Generation parameters (tuned for medical accuracy)
TEMPERATURE = 0.1  # Low = factual, deterministic (0.0-0.2 for medical)
TOP_P = 0.9        # Nucleus sampling
TOP_K = 40         # Top-k sampling
MAX_TOKENS = 300   # Max output length
STOP_SEQUENCES = [
    "\n\nUser:", 
    "\n\nQuestion:", 
    "\n\nDrug:", 
    "\n\n\n"  # Triple newline = end of response
]

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Output dimension
EMBEDDING_BATCH_SIZE = 32  # Optimal for CPU (Ryzen 3: 2-4 cores)
EMBEDDING_CACHE_SIZE = 256  # Increased from 128 (more caching)

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
CHUNK_SIZE = 2000          # Characters (not tokens) - ~500 tokens
CHUNK_OVERLAP = 400        # 20% overlap to preserve context
CHUNK_SEPARATORS = [
    "\n\n",  # Paragraph breaks (highest priority)
    "\n",    # Line breaks
    ". ",    # Sentence endings
    " "      # Word boundaries (last resort)
]

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
TOP_K_RETRIEVAL = 5                    # Number of chunks to retrieve
CONFIDENCE_THRESHOLD = 0.50            # Refuse answers below this (CRITICAL)
HALLUCINATION_THRESHOLD = 0.50         # Increased from 0.30 to reduce false positives

# Similarity thresholds for filtering
MIN_SIMILARITY_THRESHOLD = 0.30        # Ignore chunks below this
EXCELLENT_SIMILARITY = 0.75            # High-quality match
GOOD_SIMILARITY = 0.60                 # Acceptable match

# ============================================================================
# PERFORMANCE & RESOURCE LIMITS
# ============================================================================
CPU_THREADS = 4                        # Ryzen 3 logical cores (adjust for your CPU)
OLLAMA_TIMEOUT = 90                    # Seconds before timeout
OLLAMA_MAX_RETRIES = 3                 # Increased from 2 for reliability
OLLAMA_RETRY_DELAY = 1                 # Initial delay (exponential backoff)

# Request limits
MAX_QUERY_LENGTH = 500                 # Characters
MIN_QUERY_LENGTH = 5                   # Characters
MAX_CONCURRENT_REQUESTS = 5            # API rate limit

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "sample_docs"
CHROMA_PATH = BASE_DIR / "data" / "chroma_data"
LOGS_PATH = BASE_DIR / "logs"
BACKUP_PATH = BASE_DIR / "data" / "backups"  # For attack testing

# Ensure directories exist
for path in [DATA_PATH, CHROMA_PATH, LOGS_PATH, BACKUP_PATH]:
    path. mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_FILE = LOGS_PATH / "rag_chatbot.log"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")  # Override with env var
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB per log file
LOG_BACKUP_COUNT = 5               # Keep 5 rotated log files

# ============================================================================
# SAFETY & SECURITY
# ============================================================================
ENABLE_QUERY_LOGGING = True            # Log all queries for audit
SANITIZE_LOGS = True                   # Remove PII from logs
RATE_LIMIT_ENABLED = True              # Enable API rate limiting
RATE_LIMIT_REQUESTS = 20               # Requests per minute per IP

# Medical domain safety
REQUIRE_SOURCE_CITATION = True         # Always cite sources in answers
ENABLE_HALLUCINATION_DETECTION = True  # Check for invented facts
ENABLE_CONFIDENCE_GATING = True        # Refuse low-confidence queries

# ============================================================================
# DEVELOPMENT vs PRODUCTION
# ============================================================================
DEBUG_MODE = os.getenv("DEBUG", "False").lower() == "true"

if DEBUG_MODE:
    LOG_LEVEL = "DEBUG"
    OLLAMA_TIMEOUT = 120  # Longer timeout for debugging
    ENABLE_QUERY_LOGGING = True
    print("⚠️  DEBUG MODE ENABLED")

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate configuration on import"""
    errors = []
    
    # Check paths exist
    if not DATA_PATH.exists():
        errors.append(f"Data path does not exist:  {DATA_PATH}")
    
    # Check chunk overlap
    if CHUNK_OVERLAP >= CHUNK_SIZE:
        errors.append(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be < CHUNK_SIZE ({CHUNK_SIZE})")
    
    # Check confidence threshold
    if not 0.0 <= CONFIDENCE_THRESHOLD <= 1.0:
        errors.append(f"CONFIDENCE_THRESHOLD must be 0-1, got {CONFIDENCE_THRESHOLD}")
    
    # Check hallucination threshold
    if not 0.0 <= HALLUCINATION_THRESHOLD <= 1.0:
        errors.append(f"HALLUCINATION_THRESHOLD must be 0-1, got {HALLUCINATION_THRESHOLD}")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(errors))

# Validate on import
validate_config()

# ============================================================================
# EXPORT CONFIG AS DICT (for API /stats endpoint)
# ============================================================================
def get_config_dict():
    """Return config as dictionary for API"""
    return {
        "model":  MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap":  CHUNK_OVERLAP,
        "top_k":  TOP_K_RETRIEVAL,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "hallucination_threshold": HALLUCINATION_THRESHOLD,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "debug_mode": DEBUG_MODE
    }

if __name__ == "__main__":  
    # Print config for verification
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    for key, value in get_config_dict().items():
        print(f"{key:.<30} {value}")  # Fixed: removed space
    print("=" * 60)