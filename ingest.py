#!/usr/bin/env python3
"""Phase 2B: Document Ingestion Pipeline"""
import sys
from pathlib import Path
import logging

from rag_core import RAGChatbot
from config import DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ingest_all_documents(data_dir: Path = DATA_PATH):
    """
    Ingest all . txt files from data directory
    
    Args:
        data_dir: Path to directory containing documents
    """
    logger.info(f"Starting document ingestion from {data_dir}")
    
    # Check directory exists
    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        sys.exit(1)
    
    # Find all .txt files
    txt_files = list(data_dir. glob("*.txt"))
    
    if not txt_files: 
        logger.warning(f"No .txt files found in {data_dir}")
        return
    
    logger.info(f"Found {len(txt_files)} documents to ingest")
    
    # Initialize RAG bot
    bot = RAGChatbot()
    
    # Ingest each file
    results = []
    for txt_file in txt_files: 
        result = bot.ingest_document(txt_file)
        results.append(result)
        
        if result.get("success"):
            print(f"✓ {result['file']}: {result['chunks']} chunks in {result['elapsed_seconds']}s")
        else:
            print(f"✗ {txt_file. name}: {result. get('error', 'Unknown error')}")
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    total_chunks = sum(r.get("chunks", 0) for r in results)
    
    print(f"\n{'='*60}")
    print(f"Ingestion complete:")
    print(f"  - Files processed: {successful}/{len(txt_files)}")
    print(f"  - Total chunks indexed: {total_chunks}")
    print(f"  - ChromaDB collection size: {bot.collection.count()}")
    print(f"{'='*60}")
    
    logger.info("Ingestion pipeline complete")


if __name__ == "__main__":
    ingest_all_documents()