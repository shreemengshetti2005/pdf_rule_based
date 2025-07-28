# relevance_ranker.py (Optimized for Single Gemma Call)

from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import json
from embedding_engine import embed

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"
GEMMA_MODEL = "gemma3:1b"  # Updated to gemma2 for better performance

def call_gemma(prompt: str, max_tokens: int = 200) -> str:
    """
    Call Gemma through Ollama API to generate keyword-optimized sentences
    """
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": GEMMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Lower temperature for more focused keyword extraction
                "top_p": 0.8,
                "num_predict": max_tokens,
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        
        result = response.json()
        generated_text = result.get("response", "").strip()
        
        print(f"🤖 Gemma Response: {generated_text[:100]}...")
        return generated_text
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling Ollama API: {e}")
        return ""
    except Exception as e:
        print(f"❌ Error processing Gemma response: {e}")
        return ""

def extract_keywords_and_create_vector_optimized_sentence(persona: str, job_to_be_done: str) -> str:
    """
    Use Gemma to extract keywords and create a sentence optimized for vector similarity search
    This is the ONLY Gemma call in the entire ranking process
    """
    print(f"\n🔍 Creating vector-optimized query sentence...")
    print(f"   Persona: {persona}")
    print(f"   Job: {job_to_be_done}")
    
    prompt = f"""You are a vector similarity optimization expert specializing in semantic search. Your mission is to create the most effective search vector by analyzing user personas and extracting domain-specific keywords that will achieve maximum cosine similarity with target documents.

PERSONA: {persona}
GOAL/TASK: {job_to_be_done}

**VECTOR OPTIMIZATION TASK:** Create a search sentence optimized for maximum cosine similarity with target documents.

**STEPS:**
1. **DOMAIN**: Identify field and core terminology
2. **KEYWORDS** (8-10 total):
   - PRIMARY (3-4): Core technical terms
   - SECONDARY (3-4): Process/methodology terms  
   - CONTEXT (2-3): Industry qualifiers
3. **OPTIMIZED SENTENCE**: Professional sentence with strategic keyword density

**RESPONSE FORMAT:**

**DOMAIN:** [Field identification]

**KEYWORDS:**
PRIMARY: [keyword] - [why it maximizes similarity]
SECONDARY: [keyword] - [process relevance]
CONTEXT: [keyword] - [domain qualifier value]

**OPTIMIZED SENTENCE:** [Professional sentence with keyword density]

**REASONING:** [Brief optimization strategy explanation]

**Target Topic:** [INSERT TOPIC HERE]"""

    response = call_gemma(prompt, max_tokens=300)
    
    if response:
        # Parse the enhanced vector-optimized response format
        lines = response.split('\n')
        domain_analysis = ""
        primary_keywords = []
        secondary_keywords = []
        context_keywords = []
        optimized_sentence = ""
        reasoning = ""
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('DOMAIN ANALYSIS:'):
                current_section = 'domain'
                continue
            elif line.startswith('KEYWORDS WITH VECTOR WEIGHTS:'):
                current_section = 'keywords_header'
                continue
            elif line.startswith('PRIMARY (High Vector Weight):'):
                current_section = 'primary'
                continue
            elif line.startswith('SECONDARY (Medium Vector Weight):'):
                current_section = 'secondary'
                continue
            elif line.startswith('CONTEXT (Supporting Vector Weight):'):
                current_section = 'context'
                continue
            elif line.startswith('VECTOR-OPTIMIZED SENTENCE:'):
                current_section = 'sentence'
                continue
            elif line.startswith('VECTOR OPTIMIZATION REASONING:'):
                current_section = 'reasoning'
                continue
            
            # Parse content based on current section
            if current_section == 'domain' and line:
                domain_analysis += line + " "
            elif current_section == 'primary' and line.startswith('•'):
                primary_keywords.append(line)
            elif current_section == 'secondary' and line.startswith('•'):
                secondary_keywords.append(line)
            elif current_section == 'context' and line.startswith('•'):
                context_keywords.append(line)
            elif current_section == 'sentence' and line and not line.startswith('VECTOR OPTIMIZATION REASONING:'):
                if not optimized_sentence:  # Take first substantial line
                    optimized_sentence = line
            elif current_section == 'reasoning' and line:
                reasoning += line + " "
        
        # Fallback parsing if structured format fails
        if not optimized_sentence:
            # Look for sentence after various markers
            sentence_markers = ["VECTOR-OPTIMIZED SENTENCE:", "OPTIMIZED SENTENCE:", "SENTENCE:"]
            for marker in sentence_markers:
                if marker in response:
                    parts = response.split(marker)
                    if len(parts) > 1:
                        potential_sentences = [line.strip() for line in parts[1].split('\n') 
                                             if line.strip() and not line.strip().startswith('VECTOR OPTIMIZATION REASONING:')]
                        if potential_sentences:
                            optimized_sentence = potential_sentences[0]
                            break
        
        # Final fallback - use the last substantial line that's not a keyword
        if not optimized_sentence:
            substantial_lines = [line.strip() for line in response.split('\n') 
                               if line.strip() and len(line.strip()) > 25 and not line.strip().startswith('•')]
            if substantial_lines:
                optimized_sentence = substantial_lines[-1]
        
        # Enhanced logging with vector optimization details
        if domain_analysis:
            print(f"🎯 Domain Analysis: {domain_analysis.strip()[:80]}...")
        
        if primary_keywords:
            print(f"🔥 Primary Keywords (High Vector Weight):")
            for keyword in primary_keywords[:3]:
                print(f"   {keyword}")
        
        if secondary_keywords:
            print(f"⚡ Secondary Keywords (Medium Vector Weight):")
            for keyword in secondary_keywords[:2]:
                print(f"   {keyword}")
        
        if context_keywords:
            print(f"📍 Context Keywords (Supporting Weight):")
            for keyword in context_keywords[:2]:
                print(f"   {keyword}")
        
        if reasoning:
            print(f"🧠 Vector Optimization Strategy: {reasoning.strip()[:120]}...")
        
        print(f"✨ Vector-Optimized Sentence: {optimized_sentence}")
        
        return optimized_sentence if optimized_sentence else create_fallback_optimized_sentence(persona, job_to_be_done)
    
    else:
        # Fallback to keyword-based construction
        print("⚠️ Falling back to manual keyword extraction")
        return create_fallback_optimized_sentence(persona, job_to_be_done)

def create_fallback_optimized_sentence(persona: str, job_to_be_done: str) -> str:
    """
    Enhanced fallback method with vector optimization focus
    """
    print("🔧 Using vector-optimized fallback...")
    
    # Domain-specific high-value keywords for vector similarity
    domain_vectors = {
        'hr': {
            'primary': ['onboarding', 'compliance', 'employee', 'workforce'],
            'secondary': ['recruitment', 'benefits', 'payroll', 'performance', 'policies'],
            'context': ['management', 'human resources', 'organizational']
        },
        'analyst': {
            'primary': ['analysis', 'metrics', 'data', 'insights'],
            'secondary': ['dashboard', 'reporting', 'trends', 'analytics', 'business intelligence'],
            'context': ['strategic', 'operational', 'performance']
        },
        'developer': {
            'primary': ['programming', 'development', 'software', 'code'],
            'secondary': ['framework', 'algorithm', 'database', 'API', 'architecture'],
            'context': ['technical', 'engineering', 'implementation']
        },
        'student': {
            'primary': ['learning', 'education', 'academic', 'study'],
            'secondary': ['research', 'curriculum', 'assignment', 'knowledge', 'training'],
            'context': ['educational', 'scholarly', 'instructional']
        },
        'machine learning': {
            'primary': ['classification', 'model', 'algorithm', 'training'],
            'secondary': ['dataset', 'neural network', 'accuracy', 'prediction', 'feature'],
            'context': ['artificial intelligence', 'deep learning', 'supervised']
        },
        'forms': {
            'primary': ['fillable', 'templates', 'fields', 'forms'],
            'secondary': ['validation', 'submission', 'workflow', 'processing', 'automation'],
            'context': ['digital', 'electronic', 'interactive']
        },
        'financial': {
            'primary': ['financial', 'revenue', 'profit', 'budget'],
            'secondary': ['forecast', 'ROI', 'investment', 'market', 'analysis'],
            'context': ['economic', 'fiscal', 'monetary']
        }
    }
    
    # Extract high-value keywords based on vector clustering potential
    primary_keywords = []
    secondary_keywords = []
    context_keywords = []
    
    # Match persona and job against domain vectors
    persona_lower = persona.lower()
    job_lower = job_to_be_done.lower()
    combined_text = f"{persona_lower} {job_lower}"
    
    for domain, vectors in domain_vectors.items():
        if domain in combined_text:
            primary_keywords.extend(vectors['primary'][:2])
            secondary_keywords.extend(vectors['secondary'][:2])
            context_keywords.extend(vectors['context'][:1])
    
    # Extract meaningful terms from original text
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'create', 'find', 'manage', 'use', 'make', 'get', 'do', 'how', 'what', 'when', 'where'}
    
    persona_terms = [word for word in persona.lower().split() if word not in stopwords and len(word) > 2]
    job_terms = [word for word in job_to_be_done.lower().split() if word not in stopwords and len(word) > 2]
    
    # Combine and prioritize for vector optimization
    if not primary_keywords:
        primary_keywords = persona_terms[:2] + job_terms[:2]
    if not secondary_keywords:
        secondary_keywords = job_terms[2:4] + persona_terms[2:4]
    
    # Remove duplicates while preserving order
    seen = set()
    all_keywords = []
    for keyword_list in [primary_keywords, secondary_keywords, context_keywords]:
        for kw in keyword_list:
            if kw not in seen:
                all_keywords.append(kw)
                seen.add(kw)
    
    # Create vector-optimized sentence with strategic keyword placement
    if len(all_keywords) >= 4:
        # Front-load primary keywords for higher vector weight
        optimized_sentence = f"Comprehensive {all_keywords[0]} {all_keywords[1]} documentation and {all_keywords[2]} resources including {all_keywords[3]} methodologies and professional best practices"
    elif len(all_keywords) >= 2:
        optimized_sentence = f"Professional {all_keywords[0]} resources and {all_keywords[1]} documentation with implementation guidelines and industry standards"
    else:
        optimized_sentence = f"Professional documentation about {' '.join(all_keywords)} with comprehensive guidelines and best practices"
    
    print(f"🔧 Vector-Optimized Keywords: {all_keywords[:6]}")
    print(f"✨ Fallback Vector Sentence: {optimized_sentence}")
    return optimized_sentence

def rank_sections_by_relevance(sections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
    """
    Rank sections using keyword-optimized vector similarity approach
    Only calls Gemma ONCE for query optimization, then uses simple title+content for sections
    """
    if not sections:
        return []

    try:
        print(f"\n📊 Ranking {len(sections)} sections using vector-optimized approach...")
        
        # Step 1: Create vector-optimized query sentence (ONLY Gemma call)
        try:
            optimized_query = extract_keywords_and_create_vector_optimized_sentence(persona, job_to_be_done)
        except Exception as e:
            print(f"⚠️ Keyword extraction failed, using fallback: {e}")
            optimized_query = create_fallback_optimized_sentence(persona, job_to_be_done)
        
        # Step 2: Embed the optimized query
        print("🔄 Embedding optimized query...")
        q_vec = embed(optimized_query)
        query_embedding = np.vstack([q_vec])

        # Step 3: Create simple section representations (NO more Gemma calls)
        print("🔄 Creating section representations (title + content)...")
        section_representations = []
        
        for section in sections:
            # Simple representation: title + content (no Gemma processing)
            title = section.get('section_title', '')
            content = section.get('content', '')[:500]  # First 500 chars of content
            simple_rep = f"{title}. {content}"
            section_representations.append(simple_rep)

        # Step 4: Embed all section representations
        print("🔄 Computing section embeddings...")
        section_embeddings = np.vstack([embed(rep) for rep in section_representations])

        # Step 5: Compute cosine similarities
        print("🔄 Computing vector similarities...")
        similarities = cosine_similarity(query_embedding, section_embeddings)[0]

        # Step 6: Attach similarity scores and sort
        for i, section in enumerate(sections):
            section["relevance_score"] = float(similarities[i])
            section["vector_optimized"] = True  # Flag to indicate this used vector optimization
        
        # Sort by relevance score (highest first)
        sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Log top results
        print(f"✅ Vector-optimized ranking complete!")
        print(f"🏆 Top 3 similarity scores:")
        for i, sec in enumerate(sections[:3]):
            print(f"   {i+1}. Score: {sec['relevance_score']:.4f} - {sec.get('section_title', 'Untitled')[:50]}...")
        
        return sections

    except Exception as e:
        print(f"❌ Error during vector-optimized ranking: {e}")
        return []

def rerank_for_diversity(ranked_sections: List[Dict], top_n: int = 20, diversity_penalty: float = 0.8) -> List[Dict]:
    """
    Apply diversity reranking while preserving vector similarity optimization
    """
    if not ranked_sections:
        return []

    print(f"\n🎯 Applying diversity reranking for top {top_n} sections...")

    final_selection = []
    candidates = [dict(s) for s in ranked_sections]
    doc_counts = {}
    
    for iteration in range(min(top_n, len(candidates))):
        # Apply document diversity penalty to relevance scores
        for cand in candidates:
            doc = cand.get("document", "")
            penalty = doc_counts.get(doc, 0) * diversity_penalty
            cand["penalized_score"] = cand.get("relevance_score", 0.0) - penalty
        
        # Sort by penalized score (maintaining vector optimization benefits)
        candidates.sort(key=lambda x: x["penalized_score"], reverse=True)
        
        # Select best candidate
        best = candidates.pop(0)
        final_selection.append(best)
        
        # Update document count for diversity
        doc_counts[best.get("document", "")] = doc_counts.get(best.get("document", ""), 0) + 1
        
        if (iteration + 1) % 5 == 0:
            print(f"   Selected {iteration + 1}/{top_n} sections...")

    print(f"✅ Diversity reranking complete. Final selection: {len(final_selection)} sections")
    print(f"📊 Average similarity score: {np.mean([s['relevance_score'] for s in final_selection]):.4f}")
    
    return final_selection

def full_ranking_pipeline(sections: List[Dict], persona: str, job_to_be_done: str, top_n: int = 20) -> List[Dict]:
    """
    Run the complete ranking pipeline with single Gemma call optimization
    """
    print(f"\n🚀 Starting optimized ranking pipeline...")
    print(f"   Persona: {persona}")
    print(f"   Job to be done: {job_to_be_done}")
    print(f"   Input sections: {len(sections)}")
    print("-" * 60)
    
    # Step 1: Relevance ranking (only 1 Gemma call)
    ranked_sections = rank_sections_by_relevance(sections, persona, job_to_be_done)
    
    # Step 2: Diversity reranking (no Gemma calls)
    final_sections = rerank_for_diversity(ranked_sections, top_n)
    
    print(f"\n🎉 Pipeline complete! Returning {len(final_sections)} sections")
    print(f"💡 Total Gemma calls made: 1 (for query optimization only)")
    return final_sections

def test_single_gemma_call():
    """
    Test the optimized approach with single Gemma call
    """
    print("🧪 Testing single Gemma call optimization...")
    
    test_cases = [
        {
            "persona": "HR professional",
            "job": "Create and manage fillable forms for onboarding and compliance"
        },
        {
            "persona": "college student studying computer science",
            "job": "find machine learning algorithms for image classification"
        },
        {
            "persona": "business analyst",
            "job": "analyze market trends and financial performance"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        result = extract_keywords_and_create_vector_optimized_sentence(
            test_case["persona"], 
            test_case["job"]
        )
        print(f"✅ Single call result: {result}\n")

# Backward compatibility with existing main.py
def expand_query_with_gemma(persona: str, job_to_be_done: str) -> str:
    """
    Backward compatibility function - calls the optimized version
    """
    return extract_keywords_and_create_vector_optimized_sentence(persona, job_to_be_done)

def expand_query_fallback(persona: str, job_to_be_done: str) -> str:
    """
    Backward compatibility function - calls the optimized fallback
    """
    return create_fallback_optimized_sentence(persona, job_to_be_done)

if __name__ == "__main__":
    test_single_gemma_call()