# content_ranker.py - Advanced content ranking and diversity optimization
"""
Optimized content ranking system for local Gemma model with diversity considerations.
This module implements sophisticated ranking algorithms for document sections.
"""

from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from pathlib import Path
from vector_engine import embed

# Try to import required libraries for local model inference
try:
    # First try llama-cpp-python for GGUF files
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = False
    print("✅ Using llama-cpp-python for GGUF model")
except ImportError:
    try:
        # Fallback to transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        TRANSFORMERS_AVAILABLE = True
        LLAMA_CPP_AVAILABLE = False
        print("✅ Using transformers for model loading")
    except ImportError:
        TRANSFORMERS_AVAILABLE = False
        LLAMA_CPP_AVAILABLE = False
        print("⚠️ Neither llama-cpp-python nor transformers available. Please install one of them.")

# Model configuration - direct path to GGUF file
MODEL_PATH = Path("./models/Gemma-1B.Q4_K_M.gguf")

# Global model and tokenizer instances
_model = None
_tokenizer = None

def check_local_model():
    """
    Check if the local Gemma model exists at the specified path.
    
    Returns:
        bool: True if model exists, False otherwise
    """
    if MODEL_PATH.exists() and MODEL_PATH.is_file():
        print(f"✅ Model file found at: {MODEL_PATH}")
        return True
    else:
        print(f"❌ Model file not found at: {MODEL_PATH}")
        print(f"   Please ensure the model file 'Gemma-1B.Q4_K_M.gguf' exists in the models directory")
        return False

def load_local_model():
    """
    Load the local Gemma model using the appropriate library.
    
    Returns:
        tuple: (model, tokenizer) pair
    """
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    # Check if model exists
    if not check_local_model():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    print(f"🔄 Loading Gemma model from: {MODEL_PATH}")
    
    if LLAMA_CPP_AVAILABLE:
        try:
            _model = Llama(
                model_path=str(MODEL_PATH),
                n_ctx=2048,  # Context window
                n_threads=4,  # Number of threads
                n_gpu_layers=0,  # Set to -1 if you have GPU support, 0 for CPU only
                verbose=False
            )
            _tokenizer = None  # Not needed for llama-cpp-python
            print(f"✅ Model loaded successfully with llama-cpp-python!")
            return _model, _tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model with llama-cpp-python: {e}")
            raise
    
    elif TRANSFORMERS_AVAILABLE:
        # This won't work with GGUF files, but keeping for compatibility
        try:
            # This will fail for GGUF files - transformers can't load them directly
            model_dir = MODEL_PATH.parent
            _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            _model = AutoModelForCausalLM.from_pretrained(
                str(model_dir),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            
            print(f"✅ Model loaded successfully with transformers!")
            return _model, _tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model with transformers: {e}")
            print("   Note: GGUF files require llama-cpp-python, not transformers")
            raise
    
    else:
        raise ImportError("No suitable model loading library available. Install llama-cpp-python for GGUF files or transformers for standard models.")

def call_gemma_local(prompt: str, max_tokens: int = 200) -> str:
    """
    Call the local Gemma model with the given prompt.
    
    Args:
        prompt: Input prompt for the model
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        str: Generated text response
    """
    global _model, _tokenizer
    
    try:
        if _model is None:
            _model, _tokenizer = load_local_model()
        
        if LLAMA_CPP_AVAILABLE:
            # Use llama-cpp-python for GGUF models
            response = _model(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["\n\n", "###", "END"]
            )
            return response['choices'][0]['text'].strip()
            
        elif TRANSFORMERS_AVAILABLE:
            # Use transformers for other model formats
            inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = _model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=_tokenizer.eos_token_id
                )
            
            response = _tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
            
        else:
            print("❌ No model backend available")
            return ""
            
    except Exception as e:
        print(f"❌ Error calling local model: {e}")
        return ""

def extract_keywords_and_create_vector_optimized_sentence(persona: str, job_to_be_done: str) -> str:
    """
    Extract keywords and create an optimized sentence for vector search.
    
    Args:
        persona: User persona/role
        job_to_be_done: Task description
        
    Returns:
        str: Optimized sentence for vector search
    """
    try:
        # Create a comprehensive prompt for keyword extraction
        prompt = f"""
        Given the following persona and job description, extract the most important keywords and create an optimized search query.
        
        Persona: {persona}
        Job to be done: {job_to_be_done}
        
        Please provide a concise, optimized search query that captures the key requirements and context.
        Focus on the most relevant terms for document section ranking.
        """
        
        # Call the local model
        response = call_gemma_local(prompt, max_tokens=150)
        
        if response and len(response.strip()) > 10:
            # Clean and optimize the response
            optimized_sentence = response.strip()
            optimized_sentence = optimized_sentence.replace('\n', ' ').replace('  ', ' ')
            optimized_sentence = optimized_sentence[:200]  # Limit length
            
            print(f"🎯 Generated optimized sentence: {optimized_sentence}")
            return optimized_sentence
        else:
            # Fallback to basic optimization
            return create_fallback_optimized_sentence(persona, job_to_be_done)
            
    except Exception as e:
        print(f"❌ Error in keyword extraction: {e}")
        return create_fallback_optimized_sentence(persona, job_to_be_done)

def create_fallback_optimized_sentence(persona: str, job_to_be_done: str) -> str:
    """
    Create a fallback optimized sentence when model extraction fails.
    
    Args:
        persona: User persona/role
        job_to_be_done: Task description
        
    Returns:
        str: Fallback optimized sentence
    """
    # Basic keyword extraction and optimization
    keywords = []
    
    # Extract key terms from persona
    persona_lower = persona.lower()
    if 'chef' in persona_lower or 'cook' in persona_lower:
        keywords.extend(['cooking', 'recipes', 'food', 'kitchen', 'culinary'])
    if 'manager' in persona_lower or 'supervisor' in persona_lower:
        keywords.extend(['management', 'leadership', 'team', 'organization'])
    if 'student' in persona_lower or 'learner' in persona_lower:
        keywords.extend(['learning', 'education', 'study', 'academic'])
    if 'developer' in persona_lower or 'programmer' in persona_lower:
        keywords.extend(['programming', 'development', 'coding', 'software'])
    if 'designer' in persona_lower or 'creative' in persona_lower:
        keywords.extend(['design', 'creative', 'art', 'visual'])
    
    # Extract key terms from job description
    job_lower = job_to_be_done.lower()
    if 'breakfast' in job_lower:
        keywords.extend(['breakfast', 'morning', 'meal', 'food'])
    if 'lunch' in job_lower:
        keywords.extend(['lunch', 'midday', 'meal', 'food'])
    if 'dinner' in job_lower:
        keywords.extend(['dinner', 'evening', 'meal', 'food'])
    if 'recipe' in job_lower:
        keywords.extend(['recipe', 'cooking', 'ingredients', 'instructions'])
    if 'plan' in job_lower or 'planning' in job_lower:
        keywords.extend(['planning', 'strategy', 'organization'])
    if 'learn' in job_lower or 'study' in job_lower:
        keywords.extend(['learning', 'education', 'knowledge'])
    if 'develop' in job_lower or 'create' in job_lower:
        keywords.extend(['development', 'creation', 'building'])
    
    # Combine keywords with original text
    combined_text = f"{persona} {job_to_be_done}"
    if keywords:
        keyword_text = " ".join(set(keywords))
        combined_text = f"{combined_text} {keyword_text}"
    
    print(f"🔄 Using fallback optimization: {combined_text}")
    return combined_text

def rank_sections_by_relevance(sections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
    """
    Rank sections by relevance using vector similarity and local model enhancement.
    
    Args:
        sections: List of document sections
        persona: User persona/role
        job_to_be_done: Task description
        
    Returns:
        List[Dict]: Ranked sections with relevance scores
    """
    if not sections:
        return []
    
    print(f"🔄 Ranking {len(sections)} sections for relevance...")
    
    # Create optimized query using local model
    optimized_query = extract_keywords_and_create_vector_optimized_sentence(persona, job_to_be_done)
    
    # Generate embeddings for all sections
    section_texts = [sec.get("content", "") for sec in sections]
    section_embeddings = []
    
    for i, text in enumerate(section_texts):
        try:
            # Create contextual embedding
            embedding = embed_with_context(text, persona, job_to_be_done)
            section_embeddings.append(embedding)
        except Exception as e:
            print(f"❌ Error embedding section {i}: {e}")
            # Use zero embedding as fallback
            section_embeddings.append(np.zeros(384, dtype=np.float32))
    
    # Generate query embedding
    try:
        query_embedding = embed_with_context(optimized_query, persona, job_to_be_done)
    except Exception as e:
        print(f"❌ Error embedding query: {e}")
        query_embedding = np.zeros(384, dtype=np.float32)
    
    # Calculate similarities
    similarities = []
    for embedding in section_embeddings:
        try:
            similarity = cosine_similarity([query_embedding], [embedding])[0][0]
            similarities.append(similarity)
        except Exception as e:
            print(f"❌ Error calculating similarity: {e}")
            similarities.append(0.0)
    
    # Add scores to sections
    for i, section in enumerate(sections):
        section["relevance_score"] = similarities[i]
    
    # Sort by relevance score (descending)
    ranked_sections = sorted(sections, key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    print(f"✅ Ranking completed. Top score: {ranked_sections[0].get('relevance_score', 0):.3f}")
    return ranked_sections

def rerank_for_diversity(ranked_sections: List[Dict], top_n: int = 5, diversity_penalty: float = 0.8) -> List[Dict]:
    """
    Re-rank sections for diversity using MMR-like algorithm.
    
    Args:
        ranked_sections: Pre-ranked sections
        top_n: Number of sections to select
        diversity_penalty: Penalty factor for similarity
        
    Returns:
        List[Dict]: Diversity-optimized section selection
    """
    if not ranked_sections:
        return []
    
    print(f"🔄 Applying diversity re-ranking for top {top_n} sections...")
    
    selected_sections = []
    remaining_sections = ranked_sections.copy()
    
    # Always include the top-ranked section
    if remaining_sections:
        selected_sections.append(remaining_sections.pop(0))
    
    # Select remaining sections with diversity consideration
    for _ in range(min(top_n - 1, len(remaining_sections))):
        if not remaining_sections:
            break
            
        best_section = None
        best_score = -1
        
        for section in remaining_sections:
            # Calculate relevance score
            relevance_score = section.get("relevance_score", 0)
            
            # Calculate diversity penalty
            diversity_penalty_score = 0
            for selected in selected_sections:
                try:
                    # Simple text similarity as diversity measure
                    selected_text = selected.get("content", "")[:200]
                    current_text = section.get("content", "")[:200]
                    
                    # Calculate Jaccard similarity
                    selected_words = set(selected_text.lower().split())
                    current_words = set(current_text.lower().split())
                    
                    if selected_words and current_words:
                        intersection = len(selected_words.intersection(current_words))
                        union = len(selected_words.union(current_words))
                        similarity = intersection / union if union > 0 else 0
                        diversity_penalty_score = max(diversity_penalty_score, similarity)
                except Exception as e:
                    print(f"❌ Error calculating diversity: {e}")
                    diversity_penalty_score = 0
            
            # Calculate final score
            final_score = relevance_score - (diversity_penalty * diversity_penalty_score)
            
            if final_score > best_score:
                best_score = final_score
                best_section = section
        
        if best_section:
            selected_sections.append(best_section)
            remaining_sections.remove(best_section)
    
    print(f"✅ Diversity re-ranking completed. Selected {len(selected_sections)} sections.")
    return selected_sections

def full_ranking_pipeline(sections: List[Dict], persona: str, job_to_be_done: str, top_n: int = 5) -> List[Dict]:
    """
    Complete ranking pipeline combining relevance and diversity.
    
    Args:
        sections: List of document sections
        persona: User persona/role
        job_to_be_done: Task description
        top_n: Number of sections to return
        
    Returns:
        List[Dict]: Final ranked and diverse section selection
    """
    # Step 1: Rank by relevance
    ranked_sections = rank_sections_by_relevance(sections, persona, job_to_be_done)
    
    # Step 2: Apply diversity re-ranking
    diverse_sections = rerank_for_diversity(ranked_sections, top_n)
    
    return diverse_sections

def test_local_model():
    """
    Test the local model with a simple prompt.
    """
    try:
        test_prompt = "What are the key considerations for meal planning?"
        print(f"🧪 Testing local model with prompt: {test_prompt}")
        
        response = call_gemma_local(test_prompt, max_tokens=100)
        
        if response:
            print(f"✅ Model test successful. Response: {response[:100]}...")
            return True
        else:
            print("❌ Model test failed - no response")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed with error: {e}")
        return False

def expand_query_with_gemma(persona: str, job_to_be_done: str) -> str:
    """
    Expand query using local Gemma model for better search results.
    
    Args:
        persona: User persona/role
        job_to_be_done: Task description
        
    Returns:
        str: Expanded query
    """
    try:
        prompt = f"Expand this search query for better document retrieval: Persona: {persona}, Task: {job_to_be_done}"
        return call_gemma_local(prompt, max_tokens=100)
    except Exception as e:
        print(f"❌ Error expanding query: {e}")
        return expand_query_fallback(persona, job_to_be_done)

def expand_query_fallback(persona: str, job_to_be_done: str) -> str:
    """
    Fallback query expansion when model is unavailable.
    
    Args:
        persona: User persona/role
        job_to_be_done: Task description
        
    Returns:
        str: Basic expanded query
    """
    return f"{persona} {job_to_be_done}"

if __name__ == "__main__":
    # Check if required libraries are available and model exists
    if not LLAMA_CPP_AVAILABLE and not TRANSFORMERS_AVAILABLE:
        print("❌ Please install either llama-cpp-python or transformers:")
        print("   For GGUF files: pip install llama-cpp-python")
        print("   For standard models: pip install transformers torch")
        exit(1)
    
    if not check_local_model():
        print("❌ Please ensure the model file exists at the specified path")
        exit(1)
    
    test_local_model()