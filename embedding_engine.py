# embedding_engine.py

from sentence_transformers import SentenceTransformer
import numpy as np
import os
import torch
import time
from typing import Dict, List

# --------------------------------------------
# Available models (you can add more if needed)
# --------------------------------------------
AVAILABLE_MODELS = {
    "multi-qa-mpnet-base-dot-v1": {
        "size": "~420MB",
        "dimensions": 768,
        "description": "Optimized for Q&A and information retrieval"
    },
    "all-mpnet-base-v2": {
        "size": "~420MB",
        "dimensions": 768,
        "description": "Best overall quality, state-of-the-art performance, excellent for general tasks"
    },
    "paraphrase-mpnet-base-v2": {
        "size": "~420MB",
        "dimensions": 768,
        "description": "Optimized for paraphrase detection and semantic similarity"
    },
    "all-MiniLM-L12-v2": {
        "size": "~120MB",
        "dimensions": 384,
        "description": "Best quality under 120MB limit"
    },
    "all-MiniLM-L6-v2": {
        "size": "~90MB",
        "dimensions": 384,
        "description": "Good balance of size and performance"
    },
    "paraphrase-MiniLM-L6-v2": {
        "size": "~90MB",
        "dimensions": 384,
        "description": "Optimized for paraphrase detection, compact"
    }
}
        
# --------------------------------------------
# Default model to use
# --------------------------------------------
MODEL_NAME = "all-mpnet-base-v2"

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
    print("🔄 Trying fallback model: all-MiniLM-L12-v2")
    try:
        MODEL_NAME = "all-MiniLM-L12-v2"
        llm = SentenceTransformer(MODEL_NAME)
        llm.to(device)
        MODEL_INFO = AVAILABLE_MODELS[MODEL_NAME]
        print(f"✅ Fallback model loaded successfully")
    except Exception as fallback_error:
        print(f"❌ Fallback failed: {fallback_error}")
        exit(1)

# ------------------------------------------------
# Generate embedding for a single text
# ------------------------------------------------
def embed(text: str) -> np.ndarray:
    try:
        embedding = llm.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.flatten()
    except Exception as e:
        print(f"❌ Failed to embed: {e}")
        return np.zeros(MODEL_INFO["dimensions"], dtype=np.float32)

# ------------------------------------------------
# Generate enhanced embedding based on persona and job_to_be_done
# ------------------------------------------------
def embed_with_context(text: str, persona: str, job_to_be_done: str) -> np.ndarray:
    try:
        enhanced_text = f"{persona} {job_to_be_done}: {text}"
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
# Batch embedding for list of texts
# ------------------------------------------------
def batch_embed(texts: List[str], batch_size: int = 32) -> np.ndarray:
    try:
        embeddings = llm.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10
        )
        return embeddings
    except Exception as e:
        print(f"❌ Batch embedding error: {e}")
        return np.zeros((len(texts), MODEL_INFO["dimensions"]), dtype=np.float32)

# ------------------------------------------------
# Batch embedding with context for list of texts
# ------------------------------------------------
def batch_embed_with_context(texts: List[str], persona: str, job_to_be_done: str, batch_size: int = 32) -> np.ndarray:
    try:
        enhanced_texts = [f"{persona} {job_to_be_done}: {text}" for text in texts]
        embeddings = llm.encode(
            enhanced_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 10
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
# Find top-k most similar texts to a query
# ------------------------------------------------
def similarity_search(query: str, texts: List[str], top_k: int = 20) -> List[tuple]:
    try:
        query_vec = embed(query)
        text_vecs = batch_embed(texts)

        similarities = []
        for i, vec in enumerate(text_vecs):
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            similarities.append((texts[i], float(score)))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    except Exception as e:
        print(f"❌ Similarity search error: {e}")
        return []

# ------------------------------------------------
# Find top-k most similar texts to a query with context
# ------------------------------------------------
def similarity_search_with_context(query: str, texts: List[str], persona: str, job_to_be_done: str, top_k: int = 20) -> List[tuple]:
    try:
        query_vec = embed_with_context(query, persona, job_to_be_done)
        text_vecs = batch_embed_with_context(texts, persona, job_to_be_done)

        similarities = []
        for i, vec in enumerate(text_vecs):
            score = np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec) + 1e-8)
            similarities.append((texts[i], float(score)))

        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
    except Exception as e:
        print(f"❌ Similarity search with context error: {e}")
        return []