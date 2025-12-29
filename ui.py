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
