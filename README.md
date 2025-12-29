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
