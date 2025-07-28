
# embedding_engine.py

from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import time
from typing import Dict, List

# --------------------------------------------
# Available models (you can add more if needed)
# --------------------------------------------
AVAILABLE_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2":  {
        "size": "~90MB",
        "dimensions": 384,
        "description": "Fast and efficient model for semantic similarity tasks, good balance of speed and performance",
    },
    "sentence-transformers/all-MiniLM-L12-v2": {
        "size": "~120MB", 
        "dimensions": 384,
        "description": "Slightly larger MiniLM model with better performance than L6",
    },
    "intfloat/e5-base-v2": {
        "size": "~438MB",
        "dimensions": 768,
        "description": "Designed for generating text embeddings for various NLP tasks",
    },
}

# --------------------------------------------
# Default model to use
# --------------------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print(f"🔄 Loading embedding model: {MODEL_NAME}")
print(f"   Size: {AVAILABLE_MODELS[MODEL_NAME]['size']}")
print(f"   Description: {AVAILABLE_MODELS[MODEL_NAME]['description']}")

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_time = time.time()
    llm = SentenceTransformer(MODEL_NAME)
    llm.to(device)
    load_time = time.time() - start_time
    print(f"✅ Model loaded on {device} in {load_time:.2f}s")
    MODEL_INFO = AVAILABLE_MODELS[MODEL_NAME]
except Exception as e:
    print(f"❌ Error loading model {MODEL_NAME}: {e}")
    exit(1)

# ------------------------------------------------
# Generate embedding for a single text with debug output
# ------------------------------------------------
def embed(text: str, debug: bool = False) -> np.ndarray:
    try:
        if debug:
            print(f"🔍 Embedding text: {text[:100]}...")
            
        embedding = llm.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        result = embedding.flatten()
        
        if debug:
            print(f"📊 Embedding shape: {result.shape}, norm: {np.linalg.norm(result):.4f}")
            
        return result
    except Exception as e:
        print(f"❌ Failed to embed: {e}")
        return np.zeros(MODEL_INFO["dimensions"], dtype=np.float32)

# ------------------------------------------------
# Generate enhanced embedding based on persona and job_to_be_done
# ------------------------------------------------
def embed_with_context(text: str, persona: str, job_to_be_done: str, debug: bool = False) -> np.ndarray:
    try:
        enhanced_text = f"{persona} {job_to_be_done}: {text}"
        
        if debug:
            print(f"🎯 Enhanced text for embedding: {enhanced_text[:150]}...")
            
        embedding = llm.encode(
            enhanced_text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.flatten()
    except Exception as e:
        print(f"❌ Failed to embed with context: {e}")
        return np.zeros(MODEL_INFO["dimensions"], dtype=np.float32)

# ------------------------------------------------
# Batch embedding for list of texts with progress tracking
# ------------------------------------------------
def batch_embed(texts: List[str], batch_size: int = 32, debug: bool = False) -> np.ndarray:
    try:
        if debug:
            print(f"📦 Batch embedding {len(texts)} texts with batch_size={batch_size}")
            
        embeddings = llm.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10 or debug
        )
        
        if debug:
            print(f"✅ Batch embedding complete. Shape: {embeddings.shape}")
            
        return embeddings
    except Exception as e:
        print(f"❌ Batch embedding error: {e}")
        return np.zeros((len(texts), MODEL_INFO["dimensions"]), dtype=np.float32)

# ------------------------------------------------
# Batch embedding with context for list of texts
# ------------------------------------------------
def batch_embed_with_context(texts: List[str], persona: str, job_to_be_done: str, batch_size: int = 32, debug: bool = False) -> np.ndarray:
    try:
        enhanced_texts = [f"{persona} {job_to_be_done}: {text}" for text in texts]
        
        if debug:
            print(f"🎯 Batch embedding {len(texts)} texts with context")
            print(f"   Sample enhanced text: {enhanced_texts[0][:100]}...")
            
        embeddings = llm.encode(
            enhanced_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10 or debug
        )
        return embeddings
    except Exception as e:
        print(f"❌ Batch embedding with context error: {e}")
        return np.zeros((len(texts), MODEL_INFO["dimensions"]), dtype=np.float32)

# ------------------------------------------------
# Return model info for debugging or UI
# ------------------------------------------------
def get_model_info() -> Dict:
    return {
        "name": MODEL_NAME,
        "dimensions": MODEL_INFO["dimensions"],
        "size": MODEL_INFO["size"],
        "description": MODEL_INFO["description"],
        "device": device
    }

# ------------------------------------------------
# Find top-k most similar texts to a query with detailed output
# ------------------------------------------------
def similarity_search(query: str, texts: List[str], top_k: int = 20, debug: bool = False) -> List[tuple]:
    try:
        if debug:
            print(f"🔍 Similarity search for: {query[:100]}...")
            print(f"   Searching through {len(texts)} texts for top {top_k} matches")
            
        query_vec = embed(query, debug=debug)
        text_vecs = batch_embed(texts, debug=debug)

        similarities = []
        for i, vec in enumerate(text_vecs):
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            similarities.append((texts[i], float(score)))

        results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        if debug:
            print(f"🏆 Top 3 similarity scores:")
            for i, (text, score) in enumerate(results[:3]):
                print(f"   {i+1}. Score: {score:.4f} - {text[:60]}...")
                
        return results
    except Exception as e:
        print(f"❌ Similarity search error: {e}")
        return []

# ------------------------------------------------
# Find top-k most similar texts to a query with context
# ------------------------------------------------
def similarity_search_with_context(query: str, texts: List[str], persona: str, job_to_be_done: str, top_k: int = 20, debug: bool = False) -> List[tuple]:
    try:
        if debug:
            print(f"🎯 Context-aware similarity search")
            print(f"   Query: {query[:80]}...")
            print(f"   Persona: {persona}")
            print(f"   Job: {job_to_be_done}")
            
        query_vec = embed_with_context(query, persona, job_to_be_done, debug=debug)
        text_vecs = batch_embed_with_context(texts, persona, job_to_be_done, debug=debug)

        similarities = []
        for i, vec in enumerate(text_vecs):
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            similarities.append((texts[i], float(score)))

        results = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
        
        if debug:
            print(f"🏆 Top 3 context-aware similarity scores:")
            for i, (text, score) in enumerate(results[:3]):
                print(f"   {i+1}. Score: {score:.4f} - {text[:60]}...")
                
        return results
    except Exception as e:
        print(f"❌ Similarity search with context error: {e}")
        return []

# ------------------------------------------------
# Test embedding quality
# ------------------------------------------------
def test_embedding_quality(test_texts: List[str] = None):
    """Test the embedding model with sample texts"""
    if test_texts is None:
        test_texts = [
            "I need budget-friendly restaurant recommendations",
            "Looking for cheap dining options for students",
            "Expensive fine dining establishments",
            "Business analysis and market research",
            "Financial investment strategies"
        ]
    
    print("🧪 Testing embedding quality...")
    embeddings = batch_embed(test_texts, debug=True)
    
    # Compute similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    print(f"📊 Similarity matrix shape: {sim_matrix.shape}")
    print("🔍 Sample similarities:")
    for i in range(min(3, len(test_texts))):
        for j in range(i+1, min(3, len(test_texts))):
            print(f"   '{test_texts[i][:30]}...' vs '{test_texts[j][:30]}...' = {sim_matrix[i,j]:.4f}")

if __name__ == "__main__":
    print("🚀 Embedding engine loaded successfully!")
    test_embedding_quality()
