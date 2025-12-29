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
