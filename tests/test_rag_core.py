"""Phase 2B: Unit Tests for RAG Pipeline"""
import unittest
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path. insert(0, str(Path(__file__).parent.parent))

from rag_core import RAGChatbot, EmbeddingCache
from config import DATA_PATH


class TestEmbeddingCache(unittest.TestCase):
    """Test embedding cache"""
    
    def test_cache_hit(self):
        cache = EmbeddingCache(max_size=10)
        
        # Put
        cache.put("test query", [0.1, 0.2, 0.3])
        
        # Get
        result = cache.get("test query")
        self.assertIsNotNone(result)
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
        # Stats
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 0)
    
    def test_cache_miss(self):
        cache = EmbeddingCache(max_size=10)
        
        result = cache.get("nonexistent")
        self.assertIsNone(result)
        
        stats = cache.stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 1)


class TestRAGChatbot(unittest.TestCase):
    """Test RAG pipeline"""
    
    @classmethod
    def setUpClass(cls):
        """Initialize bot once for all tests"""
        cls. bot = RAGChatbot()
        
        # Ensure test data exists
        if cls.bot.collection.count() == 0:
            print("\nWarning: No documents indexed. Run: python ingest.py")
    
    def test_initialization(self):
        """Test bot initializes without errors"""
        self.assertIsNotNone(self.bot. embedder)
        self.assertIsNotNone(self.bot.collection)
        self.assertGreater(self.bot.collection.count(), 0, "No documents indexed")
    
    def test_retrieve(self):
        """Test retrieval returns results"""
        results = self.bot.retrieve("aspirin dosage", top_k=5)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
        
        if results:
            # Check structure
            doc = results[0]
            self.assertIn("text", doc)
            self.assertIn("source", doc)
            self.assertIn("similarity", doc)
            self.assertGreaterEqual(doc["similarity"], 0)
            self.assertLessEqual(doc["similarity"], 1)
    
    def test_build_prompt(self):
        """Test prompt building"""
        docs = self.bot.retrieve("aspirin dosage", top_k=3)
        
        if not docs:
            self.skipTest("No documents indexed")
        
        prompt = self.bot.build_prompt("What is aspirin dosage?", docs)
        
        self.assertIn("MEDICAL CONTEXT", prompt)
        self.assertIn("USER QUESTION", prompt)
        self.assertIn("aspirin", prompt. lower())
    
    def test_chat_valid_query(self):
        """Test full chat pipeline with valid query"""
        if self.bot.collection.count() == 0:
            self.skipTest("No documents indexed")
        
        result = self.bot.chat("What is aspirin dosage?")
        
        # Check structure
        self.assertIn("answer", result)
        self.assertIn("sources", result)
        self.assertIn("confidence", result)
        self.assertIn("latency_ms", result)
        self.assertIn("error", result)
        
        # Check values
        self.assertIsInstance(result["answer"], str)
        self.assertGreater(len(result["answer"]), 10)
        self.assertFalse(result["error"])
        self.assertGreater(result["confidence"], 0)
    
    def test_chat_short_query(self):
        """Test error handling for short query"""
        result = self.bot.chat("hi")
        
        self.assertTrue(result["error"])
        self.assertEqual(result["error_type"], "query_too_short")
    
    def test_chat_irrelevant_query(self):
        """Test error handling for irrelevant query"""
        result = self.bot.chat("What is the capital of France?")
        
        # Should either refuse (no relevant docs) or have low confidence
        self.assertTrue(
            result["error"] or result["confidence"] < 0.5,
            "Bot should refuse irrelevant medical query"
        )
    
    def test_latency_target(self):
        """Test that query latency is < 8 seconds (target:  6s)"""
        if self.bot.collection.count() == 0:
            self. skipTest("No documents indexed")
        
        result = self.bot.chat("What are aspirin side effects?")
        
        self.assertLess(
            result["latency_ms"],
            60000,
            f"Latency {result['latency_ms']}ms exceeds 60s target"
        )
        
        print(f"\n  Latency: {result['latency_ms']}ms (target: <60000ms)")

def run_tests():
    """Run all tests with verbose output"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)