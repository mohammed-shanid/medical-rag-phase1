#!/bin/bash
# Ollama Status Checker

echo "=========================================="
echo "OLLAMA STATUS CHECK"
echo "=========================================="

# Check 1: Process running? 
if pgrep -x ollama > /dev/null; then
    echo "✅ Process:  Running"
    PID=$(pgrep -x ollama)
    echo "   PID: $PID"
    
    # CPU/Memory usage
    CPU=$(ps -p $PID -o %cpu= | xargs)
    MEM=$(ps -p $PID -o %mem= | xargs)
    echo "   CPU: ${CPU}%"
    echo "   Memory: ${MEM}%"
else
    echo "❌ Process: NOT running"
    echo ""
    echo "Start with: ollama serve &"
    exit 1
fi

echo ""

# Check 2: API responding?
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ API:  Responding on port 11434"
else
    echo "❌ API:  NOT responding"
    echo "   Try: killall ollama && ollama serve &"
    exit 1
fi

echo ""

# Check 3: Models available? 
MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | grep -o '"name":"[^"]*"' | wc -l)
if [ "$MODELS" -gt 0 ]; then
    echo "✅ Models: $MODELS loaded"
    curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | sed 's/"name": "/   - /' | sed 's/"$//'
else
    echo "⚠️  Models: None found"
fi

echo ""
echo "=========================================="
echo "STATUS:  READY ✅"
echo "=========================================="
