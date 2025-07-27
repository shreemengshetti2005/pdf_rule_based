# relevance_ranker.py

from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from embedding_engine import embed

def expand_query(persona: str, job_to_be_done: str) -> str:
    base_query = f"{persona}. {job_to_be_done}"
    expansion_terms = []
    if "college friends" in job_to_be_done.lower() or "student" in persona.lower():
        expansion_terms.extend(["nightlife","bars","budget-friendly","cheap eats","adventure","group activities","social"])
    elif "researcher" in persona.lower():
        expansion_terms.extend(["methodology","dataset","results","conclusion","literature review"])
    elif "analyst" in persona.lower():
        expansion_terms.extend(["revenue","trends","market","investment","financials","strategy"])
    if expansion_terms:
        return f"{base_query}. Focus on: {', '.join(expansion_terms)}"
    return base_query

def rank_sections_by_relevance(sections: List[Dict], persona: str, job_to_be_done: str) -> List[Dict]:
    if not sections:
        return []

    try:
        enriched_query = expand_query(persona, job_to_be_done)

        # 1) Embed query and force 2D: shape (1, D)
        q_vec = embed(enriched_query)
        query_embedding = np.vstack([q_vec])

        # 2) Prepare and embed each section
        contents = [
            f"Given the task to '{job_to_be_done}', consider this section titled '{s.get('section_title','')}': {s.get('content','')}"
            for s in sections
        ]
        # vstack will ensure (N, D) even if each embed returns (D,) or (1,D)
        section_embeddings = np.vstack([embed(txt) for txt in contents])

        # 3) Compute similarities
        sims = cosine_similarity(query_embedding, section_embeddings)[0]

        # 4) Attach scores & sort
        for i, sec in enumerate(sections):
            sec["relevance_score"] = float(sims[i])
        sections.sort(key=lambda x: x["relevance_score"], reverse=True)
        return sections

    except Exception as e:
        print(f"❌ Error during task-oriented ranking: {e}")
        return []

def rerank_for_diversity(ranked_sections: List[Dict], top_n: int = 20, diversity_penalty: float = 0.8) -> List[Dict]:
    if not ranked_sections:
        return []

    final_selection = []
    candidates = [dict(s) for s in ranked_sections]
    doc_counts = {}
    for _ in range(min(top_n, len(candidates))):
        for cand in candidates:
            doc = cand.get("document","")
            penalty = doc_counts.get(doc, 0) * diversity_penalty
            cand["penalized_score"] = cand.get("relevance_score", 0.0) - penalty
        candidates.sort(key=lambda x: x["penalized_score"], reverse=True)
        best = candidates.pop(0)
        final_selection.append(best)
        doc_counts[best.get("document","")] = doc_counts.get(best.get("document",""), 0) + 1
    return final_selection
