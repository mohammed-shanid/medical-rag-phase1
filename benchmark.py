"""Phase 2B:  Benchmarking Script"""
import time
import statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from rag_core import RAGChatbot


# Test queries (30 items from Perplexity research)
TEST_QUERIES = [
    # Aspirin
    "What is the standard aspirin dosage for adults?",
    "Can children take aspirin?",
    "What are aspirin side effects?",
    "Can I take aspirin during pregnancy?",
    
    # Ibuprofen
    "What is ibuprofen dosage? ",
    "Can I take ibuprofen with alcohol?",
    "Is ibuprofen safe for long-term use?",
    "What are ibuprofen side effects?",
    
    # Acetaminophen
    "What is acetaminophen dosage?",
    "Can acetaminophen damage the liver?",
    
    # Drug interactions
    "Can I take aspirin with ibuprofen?",
    "Does acetaminophen interact with warfarin?",
    
    # Edge cases
    "What about aspirin for children with fever?",
    
    # Should trigger "I don't know"
    "What is the chemical structure of aspirin?",
    "What is the history of ibuprofen?",
]


def benchmark_rag_pipeline():
    """Run comprehensive benchmark"""
    print("="*60)
    print("PHASE 2B: RAG PIPELINE BENCHMARK")
    print("="*60)
    
    # Initialize
    print("\nInitializing RAGChatbot...")
    bot = RAGChatbot()
    
    if bot.collection.count() == 0:
        print("ERROR: No documents indexed.  Run: python ingest.py")
        return
    
    print(f"✓ Indexed documents: {bot.collection.count()}")
    
    # Run queries
    results = []
    
    print(f"\nRunning {len(TEST_QUERIES)} test queries...")
    print("-"*60)
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {query}")
        
        result = bot.chat(query)
        results.append(result)
        
        # Print summary
        status = "✗ ERROR" if result["error"] else "✓ OK"
        print(f"  {status} | Confidence: {result['confidence']:.2f} | Latency: {result['latency_ms']}ms")
        
        if not result["error"]:
            answer_preview = result["answer"][:100] + "..." if len(result["answer"]) > 100 else result["answer"]
            print(f"  Answer: {answer_preview}")
    
    # Compute statistics
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    successful = [r for r in results if not r["error"]]
    latencies = [r["latency_ms"] for r in results]
    confidences = [r["confidence"] for r in successful]
    
    print(f"\nQuery Statistics:")
    print(f"  Total queries: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Errors: {len(results) - len(successful)}")
    
    if latencies:
        print(f"\nLatency (ms):")
        print(f"  Mean: {statistics.mean(latencies):.0f}ms")
        print(f"  Median (p50): {statistics.median(latencies):.0f}ms")
        print(f"  p95: {sorted(latencies)[int(len(latencies)*0.95)]:.0f}ms")
        print(f"  p99: {sorted(latencies)[int(len(latencies)*0.99)]:.0f}ms")
        print(f"  Min: {min(latencies):.0f}ms")
        print(f"  Max: {max(latencies):.0f}ms")
        
        # Target check
        p95_latency = sorted(latencies)[int(len(latencies)*0.95)]
        target_met = "✓" if p95_latency < 6000 else "✗"
        print(f"\n  {target_met} Target (<6000ms): {p95_latency:.0f}ms")
    
    if confidences:
        print(f"\nConfidence Scores:")
        print(f"  Mean: {statistics.mean(confidences):.2f}")
        print(f"  Median: {statistics.median(confidences):.2f}")
        print(f"  Min: {min(confidences):.2f}")
        print(f"  Max: {max(confidences):.2f}")
    
    # Cache statistics
    cache_stats = bot.embedding_cache.stats()
    print(f"\nEmbedding Cache:")
    print(f"  Hit rate: {cache_stats['hit_rate']*100:.1f}%")
    print(f"  Hits: {cache_stats['hits']}")
    print(f"  Misses: {cache_stats['misses']}")
    
    # Component breakdown (from last query metadata)
    if successful and "metadata" in successful[-1]:
        meta = successful[-1]["metadata"]
        print(f"\nLatency Breakdown (last query):")
        print(f"  LLM generation: {meta.get('llm_latency_ms', 0)}ms")
        print(f"  Retrieval + prompt: {successful[-1]['latency_ms'] - meta.get('llm_latency_ms', 0)}ms")
    
    print("\n" + "="*60)
    print("Benchmark complete")
    print("="*60)


if __name__ == "__main__":
    benchmark_rag_pipeline()