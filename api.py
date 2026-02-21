"""Phase 2B: FastAPI Backend - WITH DEFENSE TOGGLE (MedRAGShield)"""
from fastapi import FastAPI, HTTPException, Request, Header
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
from defense import DefenseMiddleware, DefenseMode   # ← NEW
from config import *

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ── Rate Limiter (unchanged) ──────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, max_requests: int = 20, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self.requests[client_id] = [
            t for t in self.requests[client_id] if t > window_start
        ]
        if len(self.requests[client_id]) >= self.max_requests:
            return False
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter(max_requests=RATE_LIMIT_REQUESTS)


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=MIN_QUERY_LENGTH, max_length=MAX_QUERY_LENGTH)

    @validator('query')
    def validate_query(cls, v):
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
    answer:          str
    sources:         List[str]
    confidence:      float
    latency_ms:      int
    error:           bool
    error_type:      Optional[str]  = None
    warning:         Optional[str]  = None
    metadata:        Optional[Dict] = None
    # ── Defense fields ────────────────────────────────────────────────────────
    blocked:         bool           = False
    flagged:         bool           = False
    defense_mode:    str            = "off"
    triggered_rules: List[str]      = []


class HealthResponse(BaseModel):
    status:           str
    ollama_connected: bool
    database_count:   int
    uptime_seconds:   float


class StatsResponse(BaseModel):
    total_documents: int
    cache_stats:     Dict
    config:          Dict
    defense_stats:   Dict           # ← NEW


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting FastAPI application...")
    try:
        global bot, defense, startup_time
        startup_time = time.time()

        logger.info("Initialising RAGChatbot...")
        bot = RAGChatbot()
        logger.info(f"✓ RAGChatbot ready — {bot.collection.count()} documents")

        # ── Initialise defense middleware ─────────────────────────────────────
        defense = DefenseMiddleware()
        logger.info("✓ DefenseMiddleware ready")

        # Warmup
        logger.info("Warming up model...")
        warmup = bot.chat("warmup query")
        logger.info(f"✓ Warmed up ({warmup.get('latency_ms', 0)}ms)")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield
    logger.info("🛑 Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Medical RAG Chatbot API — MedRAGShield",
    description="RAG-based medical QA with toggle-able poisoning defenses",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request logging + rate limiting (unchanged) ───────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    if RATE_LIMIT_ENABLED:
        client_id = request.client.host
        if not rate_limiter.is_allowed(client_id):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded",
                         "detail": f"Maximum {RATE_LIMIT_REQUESTS} requests per minute"},
            )
    response = await call_next(request)
    duration = int((time.time() - start_time) * 1000)
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration}ms "
        f"client={request.client.host}"
    )
    return response


# ── Helper: read defense mode ─────────────────────────────────────────────────
def resolve_mode(
    query_mode: Optional[str],          # ?mode=...  in query string
    header_mode: Optional[str],         # X-Defense-Mode header
) -> str:
    """
    Priority: query param > header > default (off).
    Accepts: off | confidence | full
    """
    raw = query_mode or header_mode or "off"
    try:
        return DefenseMode(raw.lower()).value
    except ValueError:
        return DefenseMode.OFF.value


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message":   "Medical RAG Chatbot API — MedRAGShield",
        "version":   "2.1.0",
        "status":    "running",
        "endpoints": {
            "chat":   "/chat",
            "health": "/health",
            "stats":  "/stats",
            "docs":   "/docs",
        },
        "defense_modes": ["off", "confidence", "full"],
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    try:
        import requests as _req
        ollama_ok = False
        try:
            r = _req.get(OLLAMA_HEALTH_URL, timeout=2)
            ollama_ok = r.status_code == 200
        except Exception:
            pass

        db_count = bot.collection.count()
        uptime   = time.time() - startup_time

        return HealthResponse(
            status="healthy" if ollama_ok and db_count > 0 else "degraded",
            ollama_connected=ollama_ok,
            database_count=db_count,
            uptime_seconds=round(uptime, 2),
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def get_stats():
    try:
        rag_stats = bot.get_stats()
        return StatsResponse(
            total_documents=rag_stats["total_documents"],
            cache_stats=rag_stats["cache_stats"],
            config=get_config_dict(),
            defense_stats=defense.get_stats(),   # ← NEW
        )
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN CHAT ENDPOINT — with defense toggle
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: QueryRequest,
    mode: Optional[str] = None,                         # ?mode=off|confidence|full
    x_defense_mode: Optional[str] = Header(default=None),   # X-Defense-Mode header
):
    """
    Medical RAG chat with toggle-able defense.

    **Defense modes** (pass as query param or header):
    - `off`        → raw RAG output, no filtering
    - `confidence` → confidence + hallucination warning layer
    - `full`       → all defenses including DASC medical safety rules

    Examples:
        POST /chat?mode=off
        POST /chat?mode=confidence
        POST /chat?mode=full
        POST /chat  (with header X-Defense-Mode: full)
    """
    try:
        defense_mode = resolve_mode(mode, x_defense_mode)

        # 1. Run core RAG pipeline (unchanged)
        raw_result = bot.chat(request.query)

        # 2. Apply defense layer
        protected = defense.apply(
            rag_result=raw_result,
            mode=defense_mode,
            query=request.query,
        )

        return ChatResponse(**protected)

    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": str(e)})


# ── Batch endpoint (with defense) ─────────────────────────────────────────────
@app.post("/chat/batch", tags=["Chat"])
async def batch_chat(
    queries: List[str],
    mode: Optional[str] = None,
    x_defense_mode: Optional[str] = Header(default=None),
):
    """
    Batch chat — up to 10 queries. Same defense toggle as /chat.
    """
    if len(queries) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 queries per batch")

    defense_mode = resolve_mode(mode, x_defense_mode)
    results = []

    for q in queries:
        try:
            raw = bot.chat(q)
            protected = defense.apply(raw, mode=defense_mode, query=q)
            results.append(protected)
        except Exception as e:
            results.append({"error": True, "error_message": str(e), "query": q})

    return {"results": results, "count": len(results), "defense_mode": defense_mode}


# ── Compare endpoint (demo helper) ────────────────────────────────────────────
@app.post("/chat/compare", tags=["Demo"])
async def compare_modes(request: QueryRequest):
    """
    Run the same query under all three defense modes and return side-by-side.
    Useful for live demos and evaluation.

    Returns:
        {
          "query": "...",
          "off":        { answer, confidence, blocked, flagged, triggered_rules },
          "confidence": { ... },
          "full":       { ... }
        }
    """
    try:
        query = request.query
        comparison = {"query": query}

        for m in ["off", "confidence", "full"]:
            raw = bot.chat(query)
            protected = defense.apply(raw, mode=m, query=query)
            comparison[m] = {
                "answer":          protected["answer"],
                "confidence":      protected.get("confidence"),
                "blocked":         protected.get("blocked"),
                "flagged":         protected.get("flagged"),
                "triggered_rules": protected.get("triggered_rules", []),
                "hallucination_score": protected.get("metadata", {}).get("hallucination_score"),
            }

        return comparison

    except Exception as e:
        logger.error(f"Compare error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Models info ───────────────────────────────────────────────────────────────
@app.get("/models", tags=["System"])
async def get_models():
    return {
        "llm":      {"name": MODEL_NAME, "url": OLLAMA_URL,
                     "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
        "embedder": {"name": EMBEDDING_MODEL, "dimension": EMBEDDING_DIMENSION,
                     "batch_size": EMBEDDING_BATCH_SIZE},
        "defense":  {"version": "MedRAGShield v1.0",
                     "modes": ["off", "confidence", "full"],
                     "dasc_rules": 5},
    }


# ── Error handlers (unchanged) ────────────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={
        "error": "Not found",
        "path": str(request.url.path),
        "available_endpoints": ["/", "/chat", "/health", "/stats", "/docs"],
    })

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={
        "error": "Internal server error",
        "message": "An unexpected error occurred. Please try again.",
    })


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 MEDICAL RAG CHATBOT — MedRAGShield v2.1")
    print("=" * 60)
    print(f"🤖 LLM:    {MODEL_NAME}")
    print(f"🛡️  Defense: off | confidence | full")
    print(f"📝 Logs:   {LOG_FILE}")
    print(f"🌐 Docs:   http://127.0.0.1:8000/docs")
    print(f"🔬 Compare: POST /chat/compare")
    print("=" * 60)

    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=DEBUG_MODE,
        log_level=LOG_LEVEL.lower(),
    )