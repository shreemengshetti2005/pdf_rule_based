# output_formatter.py (Corrected Version)
"""
Generates the final structured JSON output.
"""
from typing import List, Dict
import json
import datetime

def generate_final_output(input_documents: List[str], persona: str, job_to_be_done: str, ranked_sections_with_snippets: List[Dict]) -> str:
    """
    Generates the final structured JSON output as a string, matching the required format.
    """
    timestamp = datetime.datetime.now().isoformat()

    # Build extracted_sections list
    extracted_sections = []
    for section in ranked_sections_with_snippets:
        # Use the title from the document info if available, otherwise format the filename
        doc_title = section.get("document", "").replace(".pdf", "")
        extracted_sections.append({
            "document": doc_title,
            "section_title": section.get("section_title", "Untitled Section"),
            "importance_rank": section.get("importance_rank"),
            "page_number": section.get("page_number")
        })

    # Build subsection_analysis list
    subsection_analysis = []
    for section in ranked_sections_with_snippets:
        doc_title = section.get("document", "").replace(".pdf", "")
        # The page number comes from the snippet object itself, which was populated in main.py
        for snippet in section.get("top_snippets", []):
            subsection_analysis.append({
                "document": doc_title,
                "refined_text": snippet.get("refined_text", ""),
                # This is the corrected part
                "page_number": section.get("page_number")
            })

    # Assemble final output
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