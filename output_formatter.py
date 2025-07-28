from typing import List, Dict
import json
import datetime

def finalfr(input_documents: List[str], persona: str, job_to_be_done: str, ranked_sections_with_snippets: List[Dict]) -> str:
    timestamp = datetime.datetime.now().isoformat()

    extracted_sections = []
    for section in ranked_sections_with_snippets:
        doc_title = section.get("document", "").replace(".pdf", "")
        extracted_sections.append({
            "document": doc_title,
            "section_title": section.get("section_title", "Untitled Section"),
            "importance_rank": section.get("importance_rank"),
            "page_number": section.get("page_number")
        })

    subsection_analysis = []
    for section in ranked_sections_with_snippets:
        doc_title = section.get("document", "").replace(".pdf", "")
        for snippet in section.get("top_snippets", []):
            subsection_analysis.append({
                "document": doc_title,
                "refined_text": snippet.get("refined_text", ""),
                "page_number": section.get("page_number")
            })

    output = {
        "metadata": {
            "input_documents": [doc.replace(".pdf", "") for doc in input_documents],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": timestamp
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    return json.dumps(output, indent=2)

def finalfr(input_documents: List[str], persona: str, job_to_be_done: str, ranked_sections_with_snippets: List[Dict]) -> str:
    timestamp = datetime.datetime.now().isoformat()
    extracted_sections = []
    for section in ranked_sections_with_snippets:
        doc_title = section.get("document", "").replace(".pdf", "")
        extracted_sections.append({
            "document": doc_title,
            "section_title": section.get("section_title", "Untitled Section"),
            "importance_rank": section.get("importance_rank"),
            "page_number": section.get("page_number")
        })
    subsection_analysis = []
    for section in ranked_sections_with_snippets:
        doc_title = section.get("document", "").replace(".pdf", "")
        for snippet in section.get("top_snippets", []):
            subsection_analysis.append({
                "document": doc_title,
                "refined_text": snippet.get("refined_text", ""),
                "page_number": section.get("page_number")
            })
    output = {
        "metadata": {
            "input_documents": [doc.replace(".pdf", "") for doc in input_documents],
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": timestamp
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    return json.dumps(output, indent=2)