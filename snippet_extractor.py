# contextual_snippet_extractor.py

from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
import re
from dataclasses import dataclass
import logging

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from embedding_engine import embed_with_context

@dataclass
class ContextualSnippet:
    text: str
    context_before: str
    context_after: str
    full_context: str
    relevance_score: float
    diversity_score: float
    combined_score: float
    position_in_section: int
    sentence_length: int

class ContextualEmbeddingEngine:
    def __init__(self, context_window: int = 2, overlap_ratio: float = 0.3):
        self.context_window = context_window
        self.overlap_ratio = overlap_ratio
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
    def create_contextual_embedding(self, sentence: str, context: str, persona: str, job_to_be_done: str) -> np.ndarray:
        sentence_emb = np.array(embed_with_context(sentence, persona, job_to_be_done))
        context_emb = np.array(embed_with_context(context, persona, job_to_be_done))
        
        query_sentence_sim = cosine_similarity([embed_with_context("", persona, job_to_be_done)], [sentence_emb])[0][0]
        query_context_sim = cosine_similarity([embed_with_context("", persona, job_to_be_done)], [context_emb])[0][0]
        
        sentence_weight = 0.7 + 0.2 * query_sentence_sim
        context_weight = 0.3 + 0.2 * query_context_sim
        
        total_weight = sentence_weight + context_weight
        sentence_weight /= total_weight
        context_weight /= total_weight
        
        contextual_embedding = (sentence_weight * sentence_emb + context_weight * context_emb)
        return contextual_embedding

class EnhancedSnippetExtractor:
    def __init__(self, 
                 context_window: int = 3,
                 min_sentence_length: int = 10,
                 max_sentence_length: int = 300,
                 diversity_threshold: float = 0.3):
        self.context_window = context_window
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.diversity_threshold = diversity_threshold
        self.embedding_engine = ContextualEmbeddingEngine(context_window)
        
    def advanced_sentence_cleaning(self, text: str, header: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\'\/]', ' ', text)
        
        raw_sentences = nltk.sent_tokenize(text)
        
        quality_sentences = []
        for sentence in raw_sentences:
            sentence = sentence.strip()
            
            if sentence.lower() == header.lower():
                continue
                
            word_count = len(sentence.split())
            char_count = len(sentence)
            
            if word_count < 5 or char_count < self.min_sentence_length:
                continue
            if char_count > self.max_sentence_length:
                continue
                
            if not re.search(r'[a-zA-Z]', sentence):
                continue
            if sentence.count('.') > word_count * 0.3:
                continue
            if re.match(r'^[\d\s\.\-\(\)]+$', sentence):
                continue
                
            content_words = len(re.findall(r'\b[a-zA-Z]{3,}\b', sentence))
            if content_words < 3:
                continue
                
            quality_sentences.append(sentence)
            
        return quality_sentences
    
    def create_contextual_windows(self, sentences: List[str]) -> List[Dict]:
        contextual_data = []
        
        for i, sentence in enumerate(sentences):
            start_idx = max(0, i - self.context_window)
            end_idx = min(len(sentences), i + self.context_window + 1)
            
            context_before = ' '.join(sentences[start_idx:i]) if i > 0 else ""
            context_after = ' '.join(sentences[i+1:end_idx]) if i < len(sentences)-1 else ""
            full_context = ' '.join(sentences[start_idx:end_idx])
            
            contextual_data.append({
                'sentence': sentence,
                'context_before': context_before,
                'context_after': context_after,
                'full_context': full_context,
                'position': i,
                'total_sentences': len(sentences)
            })
            
        return contextual_data
    
    def calculate_advanced_relevance(self, 
                                   contextual_data: List[Dict], 
                                   persona: str, 
                                   job_to_be_done: str) -> List[float]:
        query_emb = np.array(embed_with_context("", persona, job_to_be_done))
        
        relevance_scores = []
        
        for data in contextual_data:
            contextual_emb = self.embedding_engine.create_contextual_embedding(
                data['sentence'], 
                data['full_context'], 
                persona,
                job_to_be_done
            )
            
            base_relevance = cosine_similarity([query_emb], [contextual_emb])[0][0]
            
            position_ratio = data['position'] / max(1, data['total_sentences'] - 1)
            position_weight = 1.0 - abs(0.5 - position_ratio) * 0.2
            
            sentence_length = len(data['sentence'].split())
            length_score = min(1.0, sentence_length / 20.0)
            if sentence_length > 50:
                length_score *= 0.8
            
            context_richness = len(data['full_context'].split()) / 100.0
            context_richness = min(1.0, context_richness)
            
            final_relevance = (base_relevance * 0.7 + 
                             position_weight * 0.1 + 
                             length_score * 0.1 + 
                             context_richness * 0.1)
            
            relevance_scores.append(final_relevance)
            
        return relevance_scores
    
    def enhanced_mmr_selection(self, 
                             contextual_data: List[Dict], 
                             relevance_scores: List[float],
                             persona: str,
                             job_to_be_done: str,
                             top_k: int = 20, 
                             lambda_param: float = 0.7) -> List[int]:
        if not contextual_data or not relevance_scores:
            return []
        
        sentence_embeddings = []
        for data in contextual_data:
            emb = self.embedding_engine.create_contextual_embedding(
                data['sentence'], 
                data['full_context'], 
                persona,
                job_to_be_done
            )
            sentence_embeddings.append(emb)
        
        sentence_embeddings = np.array(sentence_embeddings)
        
        pairwise_similarities = cosine_similarity(sentence_embeddings)
        np.fill_diagonal(pairwise_similarities, 0)
        
        selected_indices = []
        remaining_indices = list(range(len(contextual_data)))
        
        if remaining_indices:
            best_idx = max(remaining_indices, key=lambda i: relevance_scores[i])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        while len(selected_indices) < top_k and remaining_indices:
            best_mmr_score = -float('inf')
            best_candidate = None
            
            for candidate_idx in remaining_indices:
                relevance_component = lambda_param * relevance_scores[candidate_idx]
                
                if selected_indices:
                    max_similarity = max(pairwise_similarities[candidate_idx][selected_indices])
                    diversity_component = (1 - lambda_param) * max_similarity
                else:
                    diversity_component = 0
                
                mmr_score = relevance_component - diversity_component
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
            else:
                break
        
        return selected_indices

def extract_enhanced_snippets(
    section_content: str,
    header: str,
    persona: str,
    job_to_be_done: str,
    query: str = "",
    top_k: int = 20,
    lambda_param: float = 0.7,
    context_window: int = 3
) -> List[ContextualSnippet]:
    if not section_content or not section_content.strip():
        return []
    
    try:
        extractor = EnhancedSnippetExtractor(
            context_window=context_window,
            diversity_threshold=0.3
        )
        
        sentences = extractor.advanced_sentence_cleaning(section_content, header)
        if not sentences:
            logging.warning("No valid sentences found after cleaning")
            return []
        
        contextual_data = extractor.create_contextual_windows(sentences)
        
        relevance_scores = extractor.calculate_advanced_relevance(
            contextual_data, persona, job_to_be_done
        )
        
        selected_indices = extractor.enhanced_mmr_selection(
            contextual_data, relevance_scores, persona, job_to_be_done, top_k, lambda_param
        )
        
        enhanced_snippets = []
        for i, idx in enumerate(selected_indices):
            data = contextual_data[idx]
            
            diversity_score = 1.0 - (i * 0.05)
            combined_score = (relevance_scores[idx] * 0.8 + diversity_score * 0.2)
            
            snippet = ContextualSnippet(
                text=data['sentence'],
                context_before=data['context_before'],
                context_after=data['context_after'],
                full_context=data['full_context'],
                relevance_score=relevance_scores[idx],
                diversity_score=diversity_score,
                combined_score=combined_score,
                position_in_section=data['position'],
                sentence_length=len(data['sentence'].split())
            )
            
            enhanced_snippets.append(snippet)
        
        enhanced_snippets.sort(key=lambda x: x.combined_score, reverse=True)
        
        logging.info(f"Successfully extracted {len(enhanced_snippets)} contextual snippets")
        return enhanced_snippets
        
    except Exception as e:
        logging.error(f"Error in enhanced snippet extraction: {e}")
        return []

def extract_top_snippets(
    section_content: str,
    header: str,
    persona: str,
    job_to_be_done: str,
    top_k: int = 20,
    lambda_param: float = 0.6
) -> List[Dict]:
    enhanced_snippets = extract_enhanced_snippets(
        section_content, header, persona, job_to_be_done, 
        query="", top_k=top_k, lambda_param=lambda_param
    )
    
    return [
        {
            "refined_text": snippet.text,
            "relevance_score": float(snippet.relevance_score)
        }
        for snippet in enhanced_snippets
    ]