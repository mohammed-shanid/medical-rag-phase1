"""Phase 2B: FastAPI Backend - OPTIMIZED & FIXED"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, Optional, List
import time
import logging
from contextlib import asynccontextmanager
from collections import defaultdict
import uvicorn

from rag_core import RAGChatbot
from config import *

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# RATE LIMITING (Simple in-memory implementation)
# ============================================================================
class RateLimiter:
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        window_start = now - self.window_seconds
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        
        # Record request
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=RATE_LIMIT_REQUESTS)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=MIN_QUERY_LENGTH, max_length=MAX_QUERY_LENGTH)
    
    @validator('query')
    def validate_query(cls, v):
        """Sanitize query"""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query": "What is the recommended aspirin dosage for adults?"
            }
        }

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence: float
    latency_ms: int
    error: bool
    error_type: Optional[str] = None
    warning: Optional[str] = None
    metadata: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    database_count: int
    uptime_seconds: float

class StatsResponse(BaseModel):
    total_documents: int
    cache_stats: Dict
    config:  Dict

# ============================================================================
# LIFESPAN CONTEXT MANAGER (Replaces deprecated on_event)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting FastAPI application...")
    
    try:
        global bot, startup_time
        startup_time = time.time()
        
        logger.info("Initializing RAGChatbot...")
        bot = RAGChatbot()
        logger.info(f"✓ RAGChatbot initialized with {bot.collection. count()} documents")
        
        # Warmup query to load model
        logger.info("Warming up model...")
        warmup_result = bot.chat("warmup query")
        logger.info(f"✓ Model warmed up ({warmup_result. get('latency_ms', 0)}ms)")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield  # Server runs here
    
    # Shutdown
    logger.info("🛑 Shutting down...")

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title="Medical RAG Chatbot API",
    description="RAG-based medical question answering with poisoning attack research",
    version="2.0.0",
    lifespan=lifespan  # Use lifespan instead of on_event
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production:  specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MIDDLEWARE
# ============================================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = time.time()
    
    # Rate limiting
    if RATE_LIMIT_ENABLED: 
        client_id = request. client.host
        if not rate_limiter.is_allowed(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "detail": f"Maximum {RATE_LIMIT_REQUESTS} requests per minute"
                }
            )
    
    # Process request
    response = await call_next(request)
    
    # Log
    duration = int((time.time() - start_time) * 1000)
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration}ms - "
        f"Client: {request.client.host}"
    )
    
    return response

# ============================================================================
# ENDPOINTS
# ============================================================================
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Medical RAG Chatbot API",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check Ollama
        import requests
        ollama_health = False
        try:
            resp = requests.get(OLLAMA_HEALTH_URL, timeout=2)
            ollama_health = resp.status_code == 200
        except: 
            pass
        
        # Check database
        db_count = bot.collection.count()
        
        uptime = time.time() - startup_time
        
        return HealthResponse(
            status="healthy" if ollama_health and db_count > 0 else "degraded",
            ollama_connected=ollama_health,
            database_count=db_count,
            uptime_seconds=round(uptime, 2)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    """Get system statistics"""
    try:
        stats = bot.get_stats()
        return StatsResponse(
            total_documents=stats["total_documents"],
            cache_stats=stats["cache_stats"],
            config=get_config_dict()
        )
    except Exception as e: 
        logger.error(f"Stats retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: QueryRequest):
    """
    Main chat endpoint - processes medical queries
    
    Returns medical information with source citations and confidence scores. 
    Refuses to answer if confidence is below threshold.
    """
    try:
        # Process query
        result = bot.chat(request.query)
        
        # Convert to response model
        return ChatResponse(**result)
    
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e)
            }
        )

@app.post("/chat/batch", tags=["Chat"])
async def batch_chat(queries: List[str]):
    """
    Batch chat endpoint - process multiple queries
    
    Limited to 10 queries per request to prevent abuse.
    """
    if len(queries) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 queries per batch request"
        )
    
    results = []
    for query in queries:
        try:
            result = bot.chat(query)
            results.append(result)
        except Exception as e:
            results.append({
                "error":  True,
                "error_message": str(e),
                "query": query
            })
    
    return {"results": results, "count": len(results)}

@app.get("/models", tags=["System"])
async def get_models():
    """Get information about loaded models"""
    return {
        "llm": {
            "name": MODEL_NAME,
            "url": OLLAMA_URL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS
        },
        "embedder": {
            "name":  EMBEDDING_MODEL,
            "dimension": EMBEDDING_DIMENSION,
            "batch_size": EMBEDDING_BATCH_SIZE
        }
    }

# ============================================================================
# ERROR HANDLERS
# ============================================================================
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "path": str(request.url.path),
            "available_endpoints": ["/", "/chat", "/health", "/stats", "/docs"]
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred.  Please try again."
        }
    )

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 STARTING MEDICAL RAG CHATBOT API")
    print("=" * 60)
    print(f"📊 Mode: {'DEBUG' if DEBUG_MODE else 'PRODUCTION'}")
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"🔢 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"📝 Logs: {LOG_FILE}")
    print(f"🌐 Docs: http://127.0.0.1:8000/docs")
    print("=" * 60)
    
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=DEBUG_MODE,
        log_level=LOG_LEVEL. lower()
    )