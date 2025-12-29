#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Creating Phase 2A File Structure..."

cd ~/medical-rag-phase1
mkdir -p data/sample_docs data/chroma_data logs

echo "✓ Directories created"

# ============================================
# 1. config.py
# ============================================
cat > config.py << 'CONFIGEOF'
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
CONFIGEOF

echo "✓ config.py created"

# ============================================
# 2. rag_core.py
# ============================================
cat > rag_core.py << 'RAGEOF'
"""Phase 2A: RAGChatbot Skeleton"""
from typing import Dict, List
import logging

from config import TOP_K_RETRIEVAL

logger = logging.getLogger(__name__)


class RAGChatbot: 
    """RAG Pipeline for poisoning research"""
    
    def __init__(self) -> None:
        """Initialize components (Phase 2B)"""
        logger.info("RAGChatbot skeleton initialized")
        self._initialized = False
    
    def ingest_document(self, text: str, source:  str) -> None:
        """Phase 2B: Chunk → Embed → Store"""
        raise NotImplementedError("Phase 2B: TODO")
    
    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        """Phase 2B: Query → Vector Search"""
        raise NotImplementedError("Phase 2B: TODO")
    
    def chat(self, query: str) -> Dict:
        """Phase 2B:  Full RAG pipeline"""
        return {
            "answer": "Phase 2B: RAG implementation required",
            "sources": [],
            "confidence": 0.0,
            "latency_ms": 0
        }


if __name__ == "__main__": 
    logging.basicConfig(level=logging.INFO)
    bot = RAGChatbot()
    print("✓ rag_core.py skeleton OK")
RAGEOF

echo "✓ rag_core. py created"

# ============================================
# 3. api.py
# ============================================
cat > api.py << 'APIEOF'
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
APIEOF

echo "✓ api.py created"

# ============================================
# 4. ui.py
# ============================================
cat > ui.py << 'UIEOF'
"""Phase 2A: Gradio UI"""
import gradio as gr
import requests
from typing import List, Tuple

API_URL = "http://127.0.0.1:8000/chat"

TITLE = "🔬 RAG Poisoning Research - Phase 2A"
DESCRIPTION = """
**Project:** RAG Poisoning Attack & Defense Framework  
**Status:** Phase 2A skeleton ✅  
**Next:** Phase 2B - Implement RAG pipeline
"""


def chat_interface(message: str, history: List[Tuple[str, str]]) -> str:
    try:
        response = requests. post(API_URL, json={"query": message}, timeout=30)
        response.raise_for_status()
        return response. json().get("answer", "No response")
    except requests.exceptions.ConnectionError:
        return "⚠️ Error:  API not running.  Start with: uvicorn api: app --reload"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


demo = gr.ChatInterface(
    fn=chat_interface,
    title=TITLE,
    description=DESCRIPTION,
    examples=["What is aspirin dosage?", "Test RAG poisoning"]
)


if __name__ == "__main__":
    print("🚀 Starting Gradio UI...")
    print("📡 Backend: http://127.0.0.1:8000")
    print("🌐 UI: http://127.0.0.1:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860)
UIEOF

echo "✓ ui.py created"

# ============================================
# 5. ingest.py
# ============================================
cat > ingest.py << 'INGESTEOF'
#!/usr/bin/env python3
"""Phase 2C: Document Ingestion Pipeline"""
import sys
from pathlib import Path

from config import DATA_PATH


def main():
    print("🚧 Phase 2C:  Ingestion pipeline")
    print(f"📂 Document path: {DATA_PATH}")
    
    if not DATA_PATH. exists():
        print(f"❌ Error: {DATA_PATH} does not exist")
        sys.exit(1)
    
    docs = list(DATA_PATH.glob("*.txt"))
    print(f"📄 Found {len(docs)} documents")
    print("✅ Ready for Phase 2C")


if __name__ == "__main__": 
    main()
INGESTEOF

chmod +x ingest.py
echo "✓ ingest.py created"

# ============================================
# 6. Sample documents
# ============================================
cat > data/sample_docs/aspirin.txt << 'ASPIRINEOF'
Aspirin (Acetylsalicylic Acid)

DOSAGE:
- Pain relief: 325-650mg every 4-6 hours (max 3000mg/day)
- Heart protection: 81mg daily
- Children:  Avoid (Reye's syndrome risk)

SIDE EFFECTS:
- Stomach irritation
- Increased bleeding risk

INTERACTIONS: 
- Warfarin: Increased bleeding risk
- Alcohol:  Stomach irritation
ASPIRINEOF

cat > data/sample_docs/ibuprofen.txt << 'IBUEOF'
Ibuprofen (Advil, Motrin)

DOSAGE:
- Adults: 200-400mg every 4-6 hours (max 1200mg/day OTC)
- Prescription: Up to 3200mg/day

SIDE EFFECTS:
- Stomach upset
- Headache, dizziness
- Fluid retention

WARNINGS: 
- Take with food
- Limit to 10 days for pain
IBUEOF

echo "✓ Sample documents created"

# ============================================
# 7. .gitignore
# ============================================
cat > .gitignore << 'GITEOF'
venv/
env/
. venv/
__pycache__/
*.py[cod]
*. so
*. egg-info/
data/chroma_data/
logs/
*. log
.DS_Store
Thumbs.db
.vscode/
.idea/
*. swp
.env
GITEOF

echo "✓ . gitignore created"

# ============================================
# 8. README.md
# ============================================
cat > README.md << 'READMEEOF'
# 🔬 RAG Poisoning Attack & Defense Framework

## 🎯 Project Goal
**PRIMARY:** Research RAG poisoning attacks and defenses  
**SECONDARY:** Medical domain as realistic testbed

## 📊 Current Status
- ✅ Phase 1: Infrastructure ready
- ✅ Phase 2A: File structure complete
- 🚧 Phase 2B:  Implement RAG pipeline (NEXT)
- ⏳ Phase 3: Attack implementation
- ⏳ Phase 4: Defense mechanisms

## 🚀 Quick Start
Terminal 1: Start API
source venv/bin/activate
uvicorn api:app --reload

Terminal 2: Start UI
python3 ui.py

## 📁 Structure
medical-rag-phase1/
├── config.py          # Central configuration
├── rag_core.py        # RAG pipeline
├── api.py             # FastAPI backend
├── ui. py              # Gradio interface
├── ingest.py          # Document ingestion
└── data/
    ├── sample_docs/   # Medical test data
    └── chroma_data/   # Vector database

## 🎯 Success Metrics
- Attack success rate: >70%
- Defense detection rate:  >75%
- Latency overhead: <1s
READMEEOF

echo "✓ README. md created"

# ============================================
# 9. Git setup
# ============================================
if [[ !  -d .git ]]; then
    git init
    git add .
    git commit -m "Phase 2A: Complete file structure

- RAG poisoning research framework
- Security-focused (medical as testbed)
- FastAPI + Gradio skeleton
- Ready for Phase 2B implementation"
    echo "✓ Git repository initialized"
else
    echo "⚠️ Git repository already exists"
fi

# ============================================
# 10. VERIFICATION
# ============================================
echo ""
echo "🔍 Phase 2A Verification:"
echo "=========================="

python3 << 'PYEOF'
import sys

try: 
    from config import MODEL_NAME, CHUNK_SIZE
    print(f"✓ config.py OK (MODEL:  {MODEL_NAME})")
except Exception as e:
    print(f"✗ config.py FAILED: {e}")
    sys.exit(1)

try:
    from rag_core import RAGChatbot
    bot = RAGChatbot()
    print("✓ rag_core.py OK")
except Exception as e:
    print(f"✗ rag_core.py FAILED: {e}")
    sys.exit(1)

try:
    from api import app
    print("✓ api.py OK")
except Exception as e:
    print(f"✗ api. py FAILED: {e}")
    sys.exit(1)

print("\n✅ Phase 2A: All imports successful!")
PYEOF

echo ""
echo "=================================="
echo "✅ PHASE 2A COMPLETE!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. uvicorn api:app --reload    # Start API"
echo "  2. python3 ui.py               # Start UI"
echo "  3. curl http://localhost:8000/health"
echo ""
echo "Ready for Phase 2B 🚀"
