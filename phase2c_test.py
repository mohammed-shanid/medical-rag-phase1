#!/usr/bin/env python3
"""
PHASE 2C - ULTIMATE VALIDATION TEST
Comprehensive medical RAG system test with diagnostic checks
"""

import time
import statistics
import requests
from rag_core import RAGChatbot
import sys

# Comprehensive test queries (20 diverse scenarios)
TEST_SUITE = {
    "Category 1: Simple Factual Queries": [
        "What is aspirin used for?",
        "What is the dosage of ibuprofen for adults?",
        "What are the side effects of warfarin?",
    ],
    
    "Category 2: Drug Interactions": [
        "Can I take aspirin with warfarin?",
        "Is it safe to take ibuprofen with alcohol?",
        "Does acetaminophen interact with warfarin?",
    ],
    
    "Category 3: Safety & Warnings": [
        "Is aspirin safe during pregnancy?",
        "Can children take ibuprofen?",
        "What should I avoid while taking warfarin?",
    ],
    
    "Category 4: Complex Medical Scenarios": [
        "What happens if I overdose on acetaminophen?",
        "What are signs of aspirin toxicity?",
        "How do I manage a missed warfarin dose?",
    ],
    
    "Category 5: New Drug Coverage (Phase 2C additions)": [
        "What is sertraline used for?",
        "What are naproxen side effects?",
        "Can I take metformin with kidney disease?",
    ],
    
    "Category 6: Multi-Drug Scenarios": [
        "Can I take naproxen and ibuprofen together?",
        "Is it safe to combine sertraline with aspirin?",
    ],
    
    "Category 7: Out-of-Scope (Should Refuse)": [
        "What is the chemical structure of aspirin?",
        "Who invented ibuprofen?",
    ],
}

def check_ollama():
    """Verify Ollama is running"""
    print("Checking Ollama status.. .", end=" ")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            print("✅ Active")
            return True
        else:
            print("❌ Not responding")
            return False
    except Exception as e:
        print(f"❌ Not running ({e})")
        return False

def check_config():
    """Display current configuration"""
    print("\n" + "=" * 70)
    print("SYSTEM CONFIGURATION CHECK")
    print("=" * 70)
    
    try:
        import config
        print(f"\n✓ MAX_TOKENS: {config.MAX_TOKENS}")
        print(f"✓ CONFIDENCE_THRESHOLD: {config. CONFIDENCE_THRESHOLD}")
        print(f"✓ HALLUCINATION_THRESHOLD: {config.HALLUCINATION_THRESHOLD}")
        
        # Check if values are correct
        warnings = []
        if config.MAX_TOKENS != 300:
            warnings.append(f"⚠️  MAX_TOKENS is {config.MAX_TOKENS}, should be 300")
        if config.HALLUCINATION_THRESHOLD != 0.30:
            warnings.append(f"⚠️  HALLUCINATION_THRESHOLD is {config.HALLUCINATION_THRESHOLD}, should be 0.30")
        
        if warnings:
            print("\n⚠️  CONFIGURATION WARNINGS:")
            for w in warnings:
                print(f"  {w}")
            return False
        else:
            print("\n✅ Configuration looks correct")
            return True
            
    except Exception as e:
        print(f"\n❌ Could not read config:  {e}")
        return False

def run_ultimate_test():
    """Run comprehensive validation"""
    
    print("=" * 70)
    print("PHASE 2C - ULTIMATE VALIDATION TEST")
    print("=" * 70)
    
    # Pre-flight checks
    if not check_ollama():
        print("\n❌ ABORT: Start Ollama first (ollama serve &)")
        return
    
    if not check_config():
        print("\n⚠️  WARNING: Configuration may not be optimal")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Test aborted.  Fix configuration first.")
            return
    
    # Initialize bot
    print("\nInitializing RAGChatbot...")
    bot = RAGChatbot()
    chunk_count = bot.collection.count()
    print(f"✓ Database loaded:  {chunk_count} chunks")
    
    if chunk_count < 900:
        print(f"\n⚠️  WARNING: Expected ~1000 chunks, found {chunk_count}")
        print("Data may not be fully ingested")
    
    # Results tracking
    results = {
        'total':  0,
        'successful': 0,
        'failed': 0,
        'refused': 0,
        'confidences': [],
        'latencies':  [],
        'hallucinations':  0,
        'categories': {},
    }
    
    # Run test suite
    print("\n" + "=" * 70)
    print("RUNNING TEST SUITE (20 QUERIES)")
    print("=" * 70)
    
    query_num = 0
    for category, queries in TEST_SUITE.items():
        results['categories'][category] = {'success': 0, 'total':  len(queries)}
        
        print(f"\n{'='*70}")
        print(f"{category}")
        print('='*70)
        
        for query in queries:
            query_num += 1
            results['total'] += 1
            
            print(f"\n[{query_num}/20] {query}")
            print("-" * 70)
            
            # Execute query
            start = time.time()
            result = bot.chat(query)
            latency = (time.time() - start) * 1000
            
            # Extract metrics
            confidence = result.get('confidence', 0)
            answer = result.get('answer', '')
            sources = result.get('sources', [])
            error = result.get('error', False)
            warning = result.get('warning', None)
            
            # Categorize result
            if error:
                status = "❌ FAILED"
                results['failed'] += 1
            elif confidence < 0.10 or "don't have" in answer.lower():
                status = "⚠️  REFUSED"
                results['refused'] += 1
                # Out-of-scope refusals are CORRECT
                if "Out-of-Scope" in category:
                    results['categories'][category]['success'] += 1
            else:
                status = "✅ SUCCESS"
                results['successful'] += 1
                results['categories'][category]['success'] += 1
                results['confidences'].append(confidence)
                results['latencies'].append(latency)
            
            if warning == 'hallucination_detected':
                results['hallucinations'] += 1
            
            # Display
            print(f"Confidence: {confidence:.2f}")
            print(f"Latency: {latency/1000:.1f}s")
            print(f"Sources: {', '.join(sources[: 2]) if sources else 'None'}")
            print(f"Answer: {answer[: 120]}...")
            if warning:
                print(f"⚠️  {warning}")
            print(f"{status}")
    
    # Calculate final statistics
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    
    success_rate = (results['successful'] / results['total']) * 100
    
    print(f"\n📊 OVERALL:")
    print(f"  Total queries: {results['total']}")
    print(f"  Successful: {results['successful']} ({success_rate:.1f}%)")
    print(f"  Failed: {results['failed']}")
    print(f"  Refused: {results['refused']}")
    
    if results['confidences']:
        print(f"\n📈 CONFIDENCE:")
        print(f"  Mean: {statistics.mean(results['confidences']):.2f}")
        print(f"  Median: {statistics.median(results['confidences']):.2f}")
        print(f"  Range: {min(results['confidences']):.2f} - {max(results['confidences']):.2f}")
    
    if results['latencies']:
        print(f"\n⏱️  LATENCY:")
        print(f"  Median: {statistics.median(results['latencies'])/1000:.1f}s")
        print(f"  Mean: {statistics.mean(results['latencies'])/1000:.1f}s")
        print(f"  Range: {min(results['latencies'])/1000:.1f}s - {max(results['latencies'])/1000:.1f}s")
    
    print(f"\n🔍 HALLUCINATIONS:")
    if results['successful'] > 0:
        hall_rate = (results['hallucinations'] / results['successful']) * 100
        print(f"  Flagged: {results['hallucinations']}/{results['successful']} ({hall_rate:.0f}%)")
    
    print(f"\n📋 BY CATEGORY:")
    for category, stats in results['categories'].items():
        rate = (stats['success'] / stats['total']) * 100
        status = "✅" if rate >= 66 else "⚠️"
        print(f"  {status} {category}: {stats['success']}/{stats['total']} ({rate:.0f}%)")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("PHASE 2C VERDICT")
    print("=" * 70)
    
    # Success criteria
    criteria_met = []
    criteria_failed = []
    
    if success_rate >= 75:
        criteria_met.append(f"✅ Success rate ≥75%:  {success_rate:.0f}%")
    else:
        criteria_failed.append(f"❌ Success rate <75%: {success_rate:.0f}%")
    
    if results['confidences'] and statistics.median(results['confidences']) >= 0.50:
        criteria_met.append(f"✅ Confidence ≥0.50: {statistics.median(results['confidences']):.2f}")
    elif results['confidences']:
        criteria_failed.append(f"⚠️  Confidence <0.50: {statistics.median(results['confidences']):.2f}")
    
    if results['latencies'] and statistics.median(results['latencies']) < 60000:
        criteria_met.append(f"✅ Latency <60s: {statistics. median(results['latencies'])/1000:.1f}s")
    elif results['latencies']:
        criteria_failed.append(f"⚠️  Latency >60s: {statistics.median(results['latencies'])/1000:.1f}s")
    
    if chunk_count >= 900:
        criteria_met.append(f"✅ Database size ≥900 chunks: {chunk_count}")
    else:
        criteria_failed.append(f"❌ Database size <900 chunks: {chunk_count}")
    
    print("\nCriteria Met:")
    for c in criteria_met:
        print(f"  {c}")
    
    if criteria_failed:
        print("\nCriteria Not Met:")
        for c in criteria_failed:
            print(f"  {c}")
    
    print("\n" + "=" * 70)
    
    if success_rate >= 75 and len(criteria_met) >= 3:
        print("✅ PHASE 2C:  COMPLETE")
        print("\nYour RAG system is ready for Phase 2D (baseline evaluation)")
        print("Success rate exceeds 75% goal.  Proceed to poisoning attacks.")
    elif success_rate >= 65: 
        print("⚠️  PHASE 2C: ACCEPTABLE")
        print("\nSystem is functional but below ideal performance.")
        print("You can proceed to Phase 2D or spend time optimizing.")
    else:
        print("❌ PHASE 2C: NEEDS WORK")
        print("\nSystem is not performing adequately.")
        print("Recommendations:")
        print("  1. Verify config. py settings (MAX_TOKENS=300, HALLUCINATION_THRESHOLD=0.30)")
        print("  2. Check rag_core.py has top_k=5")
        print("  3. Re-run ingestion:  python ingest.py")
        print("  4. Restart Ollama: killall ollama && ollama serve &")
    
    print("=" * 70)

if __name__ == "__main__":
    run_ultimate_test()
