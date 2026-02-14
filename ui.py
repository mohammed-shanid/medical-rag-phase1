"""Phase 2B:  Gradio UI - FIXED for Gradio compatibility"""
import gradio as gr
import requests
from typing import List, Tuple
import sys

API_URL = "http://127.0.0.1:8000/chat"
HEALTH_URL = "http://127.0.0.1:8000/health"

TITLE = "🔬 Medical RAG Chatbot"
DESCRIPTION = """
**Research Prototype** - Medical Question Answering System  
⚠️ For research purposes only - Not for clinical use
"""

def check_api_health() -> Tuple[bool, str]:
    """Check if API is running"""
    try:
        print("Checking API health...")
        response = requests.get(HEALTH_URL, timeout=5)
        if response.status_code == 200:
            data = response.json()
            msg = f"✅ API Healthy | Documents: {data.get('database_count', 0)}"
            print(msg)
            return True, msg
        else:
            msg = f"⚠️ API returned status {response.status_code}"
            print(msg)
            return False, msg
    except requests.exceptions.ConnectionError:
        msg = "❌ API Not Running - Start with: python api.py"
        print(msg)
        return False, msg
    except Exception as e:
        msg = f"❌ Error:  {str(e)}"
        print(msg)
        return False, msg

def chat_interface(message: str, history: List[Tuple[str, str]]) -> str:
    """Process user query"""
    
    if not message or len(message. strip()) < 5:
        return "⚠️ Please enter a question (at least 5 characters)"
    
    try:
        print(f"Sending query: {message}")
        response = requests.post(
            API_URL,
            json={"query": message},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "No answer")
            confidence = result.get("confidence", 0)
            sources = result.get("sources", [])
            latency = result.get("latency_ms", 0)
            error = result.get("error", False)
            warning = result.get("warning")
            
            # Format response
            output = f"{answer}\n\n"
            output += "---\n\n"
            
            # Confidence
            if confidence >= 0.70:
                output += f"🟢 **Confidence:** High ({confidence:.2f})\n"
            elif confidence >= 0.50:
                output += f"🟡 **Confidence:** Medium ({confidence:.2f})\n"
            else: 
                output += f"🔴 **Confidence:** Low ({confidence:.2f})\n"
            
            # Sources
            if sources:
                output += f"📚 **Sources:** {', '.join(sources)}\n"
            
            # Latency
            output += f"⏱️ **Response Time:** {latency/1000:.1f}s\n"
            
            # Warnings
            if error:
                output += f"\n⚠️ **Error Type:** {result.get('error_type', 'unknown')}\n"
            if warning:
                output += f"\n⚠️ **Warning:** {warning. replace('_', ' ').title()}\n"
            
            return output
        
        elif response.status_code == 429:
            return "⚠️ Rate limit exceeded. Please wait a moment."
        
        else:
            return f"❌ API Error: HTTP {response.status_code}"
    
    except requests.exceptions. ConnectionError:
        return "❌ API Not Running\n\nStart the API server with:\n```\npython api.py\n```"
    
    except requests.exceptions.Timeout:
        return "⏱️ Request timed out (> 120s). The query may be too complex."
    
    except Exception as e:
        print(f"Error: {e}")
        return f"❌ Unexpected Error: {str(e)}"

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 STARTING GRADIO UI")
    print("=" * 60)
    
    # Check API health
    api_healthy, health_message = check_api_health()
    print(health_message)
    
    # Example queries
    examples = [
        "What is aspirin dosage?",
        "Can I take ibuprofen with warfarin?",
        "What are sertraline side effects?",
        "Is acetaminophen safe during pregnancy?",
    ]
    
    print("=" * 60)
    print("🌐 Gradio UI: http://127.0.0.1:7860")
    print("📡 Backend API: http://127.0.0.1:8000")
    print("=" * 60)
    
    if not api_healthy:
        print("⚠️ WARNING: API is not running!")
        print("  Start API first: python api.py")
        print("=" * 60)
    
    try:
        # Create interface (compatible with all Gradio versions)
        demo = gr.ChatInterface(
            fn=chat_interface,
            title=TITLE,
            description=DESCRIPTION,
            examples=examples
        )
        
        # Launch
        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Failed to start:  {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)