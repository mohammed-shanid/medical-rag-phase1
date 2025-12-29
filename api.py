"""Phase 2A: FastAPI Backend"""
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_core import RAGChatbot


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG bot on startup"""
    app.state. rag_bot = RAGChatbot()
    yield


app = FastAPI(
    title="RAG Poisoning Research API",
    version="Phase 2A",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok", "phase": "2A"}


@app.post("/chat")
async def chat_endpoint(request: QueryRequest) -> Dict[str, Any]:
    try:
        response = app. state.rag_bot.chat(request.query)
        return response
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__": 
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
