"""Phase 2B: Complete RAG Pipeline Implementation - FIXED & OPTIMIZED"""
from typing import Dict, List, Optional, Tuple
import logging
import time
import hashlib
import json
import requests
from pathlib import Path
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb. config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import *

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for query embeddings with improved eviction"""
    
    def __init__(self, max_size:  int = EMBEDDING_CACHE_SIZE):
        self.cache = {}
        self.access_times = {}  # Track last access for LRU
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def _hash(self, text: str) -> str:
        return hashlib.md5(text. encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        key = self._hash(text)
        if key in self.cache:
            self.hits += 1
            self.access_times[key] = time.time()  # Update access time
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, text: str, embedding: List[float]):
        key = self._hash(text)
        
        # Evict LRU item if cache full
        if len(self.cache) >= self.max_size:
            # Find least recently used
            lru_key = min(self. access_times, key=self.access_times. get)
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = embedding
        self.access_times[key] = time. time()
    
    def stats(self) -> Dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self. hits,
            "misses":  self.misses,
            "hit_rate": round(hit_rate, 3),
            "cache_size": len(self.cache)
        }


class RAGChatbot:
    """Production RAG Pipeline for Medical QA - FIXED & OPTIMIZED"""
    
    def __init__(self):
        """Initialize all components"""
        logger.info("Initializing RAGChatbot...")
        
        # Optimize PyTorch for CPU
        torch.set_num_threads(CPU_THREADS)
        torch.set_num_interop_threads(1)
        
        # Load embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        
        # Initialize embedding cache
        self.embedding_cache = EmbeddingCache()
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {CHROMA_PATH}")
        self.chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.chroma_client.get_or_create_collection(
            name="medical_documents",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CHUNK_SEPARATORS,
            length_function=len
        )
        
        # Verify Ollama connection
        self._verify_ollama()
        
        logger.info("RAGChatbot initialized successfully")
        logger.info(f"Indexed documents: {self.collection.count()}")
    
    def _verify_ollama(self):
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                
                if MODEL_NAME in model_names:
                    logger.info(f"Ollama connection verified - {MODEL_NAME} available")
                else:
                    logger.warning(f"{MODEL_NAME} not found.  Available:  {model_names}")
            else:
                logger.warning(f"Ollama returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            raise RuntimeError("Ollama not running.  Start with: ollama serve")
    
    def _extract_metadata(self, text: str, source_file: str) -> Dict:
        """Extract medical metadata from text - OPTIMIZED"""
        
        # Medical sections (common patterns)
        sections = {
            "DOSAGE": ["dosage", "administration", "how to take"],
            "SIDE EFFECTS": ["side effects", "adverse reactions"],
            "INTERACTIONS": ["drug interactions", "interactions"],
            "CONTRAINDICATIONS": ["contraindications", "do not use"],
            "WARNINGS": ["warnings", "precautions", "boxed warning"]
        }
        
        text_lower = text.lower()
        section = "GENERAL"
        
        # Find first matching section
        for sec_name, keywords in sections.items():
            if any(keyword in text_lower for keyword in keywords):
                section = sec_name
                break
        
        # Extract drug name from filename (more reliable than text parsing)
        drug_name = source_file.replace('. txt', '').replace('_', ' ').lower()
        
        return {
            "source": source_file,
            "section": section,
            "drug_name": drug_name
        }
    
    def ingest_document(self, file_path: Path) -> Dict:
        """
        Ingest a single document:  chunk → embed → store
        
        Returns:
            Dict with ingestion statistics
        """
        logger.info(f"Ingesting document: {file_path}")
        start_time = time.time()
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            return {"error": str(e), "success": False}
        
        # Chunk text
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Prepare data
        chunk_ids = [f"{file_path. stem}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                **self._extract_metadata(chunk, file_path. name),
                "chunk_index": i,
                "ingestion_date": datetime.now().isoformat()
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Embed in batches (OPTIMIZED)
        all_embeddings = []
        for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch = chunks[i:i + EMBEDDING_BATCH_SIZE]
            batch_embeddings = self.embedder.encode(
                batch,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=EMBEDDING_BATCH_SIZE
            )
            all_embeddings.extend(batch_embeddings. tolist())
            
            if (i + EMBEDDING_BATCH_SIZE) % 128 == 0:
                logger.info(f"Embedded {min(i + EMBEDDING_BATCH_SIZE, len(chunks))}/{len(chunks)} chunks")
        
        # Store in ChromaDB
        try: 
            self.collection.add(
                ids=chunk_ids,
                embeddings=all_embeddings,
                documents=chunks,
                metadatas=metadatas
            )
        except Exception as e:
            logger.error(f"ChromaDB insertion failed: {e}")
            return {"error": str(e), "success": False}
        
        elapsed = time.time() - start_time
        logger.info(f"Ingested {file_path. name} in {elapsed:.2f}s")
        
        return {
            "success": True,
            "file":  file_path.name,
            "chunks": len(chunks),
            "elapsed_seconds": round(elapsed, 2)
        }
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieve top-k most relevant documents - FIXED BUG
        
        Returns:
            List of dicts with keys: text, similarity, source, distance
        """
        start_time = time.time()
        
        # Check cache first
        cached_embedding = self.embedding_cache.get(query)
        if cached_embedding:
            query_embedding = cached_embedding
        else: 
            # Generate query embedding
            query_embedding = self.embedder.encode(query, convert_to_numpy=True).tolist()
            self.embedding_cache.put(query, query_embedding)
        
        # Query ChromaDB
        try: 
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []
        
        # Check if empty
        if not results['documents'] or not results['documents'][0]:
            logger.warning(f"No relevant docs for:  {query}")
            return []
        
        # ✅ FIX: Transform ChromaDB format → our format
        documents = []
        for i in range(len(results['documents'][0])):
            distance = results['distances'][0][i]
            
            # ✅ FIX: Convert cosine distance (0-2) → similarity (0-1)
            # ChromaDB cosine distance:  0 = identical, 2 = opposite
            similarity = 1 - (distance / 2)  # Normalize to 0-1
            similarity = max(0.0, min(1.0, similarity))  # Clamp to valid range
            
            doc = {
                'text': results['documents'][0][i],
                'similarity': round(similarity, 3),
                'source': results['metadatas'][0][i]. get('source', 'unknown'),
                'distance': round(distance, 3),  # Keep for debugging
                'section': results['metadatas'][0][i].get('section', 'GENERAL'),
                'drug_name': results['metadatas'][0][i].get('drug_name', 'unknown')
            }
            documents.append(doc)
        
        retrieval_time = int((time.time() - start_time) * 1000)
        logger.info(f"Retrieved {len(documents)} docs in {retrieval_time}ms")
        
        # Log top similarities for debugging
        if documents:
            top_sims = [d['similarity'] for d in documents[: 3]]
            logger.debug(f"Top-3 similarities: {top_sims}")
        
        return documents
    
    def _calculate_confidence(self, documents: List[Dict]) -> float:
        """
        Calculate confidence score from top-3 similarities - OPTIMIZED
        
        Uses weighted average:  top result gets more weight
        """
        if not documents:
            return 0.0
        
        # Get top-3 similarities
        top_3 = documents[:3]
        similarities = [d["similarity"] for d in top_3]
        
        # Weighted average (top result = 50%, 2nd = 30%, 3rd = 20%)
        weights = [0.5, 0.3, 0.2]
        weighted_sum = sum(sim * weight for sim, weight in zip(similarities, weights[: len(similarities)]))
        
        # Normalize if less than 3 results
        total_weight = sum(weights[: len(similarities)])
        confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return round(confidence, 3)
    
    def _detect_hallucination(self, answer: str, documents: List[Dict]) -> float:
        """
        Hallucination detection - IMPROVED
        
        Returns: 
            Score 0-1 (0=no hallucination, 1=complete hallucination)
        """
        if not documents:
            return 1.0
        
        # Combine source text
        source_text = " ".join([d["text"]. lower() for d in documents])
        
        # Extract meaningful terms from answer (words > 4 chars, exclude common words)
        stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 
                      'your', 'there', 'their', 'what', 'which', 'when', 'where',
                      'should', 'could', 'would', 'about', 'these', 'those'}
        
        answer_terms = [
            w. lower() for w in answer.split() 
            if len(w) > 4 and w.lower() not in stop_words and w.isalpha()
        ]
        
        if not answer_terms:
            return 0.0
        
        # Count how many terms are in sources
        found_count = sum(1 for term in answer_terms if term in source_text)
        coverage = found_count / len(answer_terms)
        
        # Hallucination score = 1 - coverage
        hallucination_score = 1 - coverage
        
        # Log if high hallucination detected
        if hallucination_score > 0.5:
            logger.debug(f"Hallucination check: {found_count}/{len(answer_terms)} terms found")
        
        return round(hallucination_score, 2)
    
    def build_prompt(self, query: str, documents: List[Dict]) -> str:
        """
        Build RAG prompt with medical safety emphasis - OPTIMIZED
        
        Args:
            query: User question
            documents: Retrieved context documents
        
        Returns:
            Complete prompt for LLM
        """
        system_prompt = """You are a medical information assistant. Your role is to provide accurate, evidence-based medical information. 

CRITICAL RULES:
1. Answer ONLY using the provided context/documents below
2. If the context doesn't contain the answer, respond EXACTLY:  "I don't have enough information in my knowledge base to answer this question."
3. ALWAYS cite your sources using this format: (Source: filename.txt)
4. Be precise and concise - avoid unnecessary elaboration
5. NEVER make up medical information or extrapolate beyond the sources
6. NEVER provide personal medical advice - always suggest consulting a healthcare provider for medical decisions
7. If dosages are mentioned, cite them EXACTLY as written in the source

RESPONSE FORMAT:
- Direct answer first
- Source citation in parentheses
- Keep responses under 150 words unless complex query requires more"""
        
        # Format context (top 3 documents for quality, avoid noise)
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):
            source = doc.get("source", "unknown")
            text = doc.get("text", "")
            similarity = doc.get("similarity", 0)
            section = doc.get("section", "GENERAL")
            
            # Truncate very long chunks
            if len(text) > 800:
                text = text[:800] + "...  [truncated]"
            
            context_parts.append(f"""[Document {i} - {source} ({section} section, relevance: {similarity:.2f})]
{text}""")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build final prompt
        prompt = f"""{system_prompt}

MEDICAL CONTEXT:
{context}

USER QUESTION: {query}

ANSWER (cite sources):"""
        
        return prompt
    
    def _call_ollama(self, prompt: str) -> Dict:
        """
        Call Ollama with retry logic and improved error handling
        
        Returns: 
            Dict with keys: text, latency_ms, tokens, error
        """
        payload = {
            "model": MODEL_NAME,
            "prompt":  prompt,
            "options": {
                "temperature":  TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "num_predict": MAX_TOKENS,
            },
            "stream": False
        }
        
        # Add stop sequences if defined
        if STOP_SEQUENCES:
            payload["options"]["stop"] = STOP_SEQUENCES
        
        for attempt in range(OLLAMA_MAX_RETRIES):
            start_time = time.time()
            
            try:
                response = requests.post(
                    OLLAMA_URL,
                    json=payload,
                    timeout=OLLAMA_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    answer_text = result.get("response", "").strip()
                    
                    # Validate response
                    if not answer_text:
                        logger.warning("Empty response from Ollama")
                        if attempt < OLLAMA_MAX_RETRIES - 1:
                            continue
                    
                    return {
                        "text": answer_text,
                        "latency_ms": latency_ms,
                        "tokens": result.get("eval_count", 0),
                        "error": False
                    }
                else: 
                    logger.warning(f"Ollama returned {response.status_code}:  {response.text[: 200]}")
            
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{OLLAMA_MAX_RETRIES}")
                if attempt < OLLAMA_MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff:  1s, 2s, 4s
            
            except Exception as e:
                logger.error(f"Ollama error on attempt {attempt + 1}: {e}")
                if attempt < OLLAMA_MAX_RETRIES - 1:
                    time.sleep(1)
        
        # All retries failed
        return {
            "text": "I'm experiencing technical difficulties connecting to the language model. Please try again in a moment.",
            "latency_ms": int((time.time() - start_time) * 1000),
            "tokens": 0,
            "error": True
        }
    
    def generate_response(self, prompt: str) -> Dict:
        """
        Generate LLM response via Ollama
        
        Args:
            prompt: Complete RAG prompt
        
        Returns:
            Dict with keys: text, latency_ms, tokens, error
        """
        return self._call_ollama(prompt)
    
    def chat(self, query: str) -> Dict:
        """
        Complete RAG pipeline:  retrieve → prompt → generate → validate - FIXED ALL BUGS
        
        Args:
            query: User question
        
        Returns:
            Dict with keys: answer, sources, confidence, latency_ms, error, metadata
        """
        pipeline_start = time.time()
        
        # Validation:  Query too short
        query = query.strip()
        if len(query) < 5:
            return {
                "answer": "Please ask a more specific question (at least 5 characters).",
                "sources": [],
                "confidence": 0.0,
                "latency_ms": int((time.time() - pipeline_start) * 1000),
                "error": True,
                "error_type": "query_too_short"
            }
        
        # Validation:  Query too long
        if len(query) > 500:
            query = query[:500]
            logger.warning("Query truncated to 500 characters")
        
        try:
            # ✅ STEP 1: Retrieve documents
            documents = self.retrieve(query, top_k=5)
            
            # ✅ CHECK: No relevant documents found
            if not documents: 
                logger.warning(f"No relevant docs for: {query}")
                return {
                    "answer": "I don't have information about this in my knowledge base.  Please consult a healthcare provider.",
                    "sources": [],
                    "confidence": 0.0,
                    "latency_ms":  int((time.time() - pipeline_start) * 1000),
                    "error": True,
                    "error_type":  "no_relevant_docs"
                }
            
            # ✅ STEP 2: Calculate confidence (FIXED - now uses correct similarity values)
            confidence = self._calculate_confidence(documents)
            
            logger.debug(f"Query: '{query}' | Confidence: {confidence:.3f} | Threshold: {CONFIDENCE_THRESHOLD}")
            
            # ✅ CRITICAL CHECK: Refuse if confidence too low (BEFORE LLM call)
            if confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"🚫 REFUSING query - Low confidence ({confidence:.3f} < {CONFIDENCE_THRESHOLD}) for: {query}")
                return {
                    "answer": f"I'm not confident enough to answer this question (confidence: {confidence:.2f}). The available information may not directly address your query.  Please consult a healthcare provider for accurate medical advice.",
                    "sources": [d["source"] for d in documents[: 2]],
                    "confidence": confidence,
                    "latency_ms":  int((time.time() - pipeline_start) * 1000),
                    "error": True,
                    "error_type":  "low_confidence"
                }
            
            # ✅ Only proceed to LLM if confidence is acceptable
            logger.info(f"✅ Confidence OK ({confidence:.3f}) - proceeding to LLM")
            
            # ✅ STEP 3: Build prompt
            prompt = self.build_prompt(query, documents)
            
            # ✅ STEP 4: Generate response
            llm_result = self.generate_response(prompt)
            
            if llm_result["error"]: 
                return {
                    "answer": llm_result["text"],
                    "sources": [],
                    "confidence": 0.0,
                    "latency_ms": int((time.time() - pipeline_start) * 1000),
                    "error": True,
                    "error_type": "llm_error"
                }
            
            # ✅ STEP 5: Hallucination detection
            hallucination_score = self._detect_hallucination(llm_result["text"], documents)
            
            answer = llm_result["text"]
            warning = None
            
            # ✅ Hallucination warning
            if hallucination_score > HALLUCINATION_THRESHOLD: 
                logger.warning(f"⚠️ Hallucination detected ({hallucination_score:.2f})")
                warning = "hallucination_detected"
                # Penalize confidence for hallucinations
                confidence = confidence * (1 - hallucination_score * 0.5)
            
            # ✅ Success - return complete response
            total_latency = int((time.time() - pipeline_start) * 1000)
            
            result = {
                "answer": answer,
                "sources": list(set([d["source"] for d in documents[: 3]])),  # Unique sources
                "confidence": round(confidence, 3),
                "latency_ms": total_latency,
                "error":  False,
                "metadata": {
                    "retrieved_docs": len(documents),
                    "top_similarities": [d["similarity"] for d in documents[:3]],
                    "llm_tokens": llm_result["tokens"],
                    "llm_latency_ms": llm_result["latency_ms"],
                    "hallucination_score": hallucination_score,
                    "cache_stats": self.embedding_cache.stats()
                }
            }
            
            if warning:
                result["warning"] = warning
            
            # ✅ Log query
            self._log_query(query, documents, result)
            
            return result
        
        except Exception as e: 
            logger.error(f"❌ Unexpected error in chat(): {e}", exc_info=True)
            return {
                "answer": "An unexpected error occurred. Please try again.",
                "sources": [],
                "confidence": 0.0,
                "latency_ms":  int((time.time() - pipeline_start) * 1000),
                "error": True,
                "error_type": "unexpected",
                "error_message": str(e)
            }
    
    def _log_query(self, query:  str, documents: List[Dict], result: Dict):
        """Log query with all pipeline data - ENHANCED"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_count": len(documents),
            "top_sources": [d["source"] for d in documents[:3]],
            "top_similarities": [d["similarity"] for d in documents[:3]],
            "confidence": result.get("confidence", 0),
            "latency_ms":  result.get("latency_ms", 0),
            "error": result.get("error", False),
            "warning": result.get("warning")
        }
        
        if result.get("error"):
            logger.error(json.dumps(log_entry))
        elif result.get("warning"):
            logger.warning(json.dumps(log_entry))
        else:
            logger.info(json. dumps(log_entry))
    
    def get_stats(self) -> Dict:
        """Get system statistics - NEW"""
        return {
            "total_documents": self.collection.count(),
            "cache_stats": self.embedding_cache.stats(),
            "model":  MODEL_NAME,
            "embedding_model": EMBEDDING_MODEL,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "hallucination_threshold": HALLUCINATION_THRESHOLD
        }


if __name__ == "__main__": 
    # Quick test
    print("Initializing RAGChatbot...")
    bot = RAGChatbot()
    print(f"✓ RAGChatbot initialized")
    print(f"✓ Indexed documents: {bot.collection.count()}")
    
    # Test query
    test_query = "What is aspirin dosage?"
    print(f"\nTesting query: '{test_query}'")
    result = bot.chat(test_query)
    
    print(f"\nResult:")
    print(f"  Answer: {result['answer'][: 200]}...")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Sources: {result['sources']}")
    print(f"  Latency: {result['latency_ms']}ms")
    print(f"  Error: {result['error']}")
    
    print(f"\n✓ Ready for queries")