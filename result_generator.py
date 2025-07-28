# result_generator.py - Final output formatting and JSON generation
"""
Generates the final structured JSON output for the document processing pipeline.
This module handles the formatting and assembly of the final results.
"""
from typing import List, Dict
import json
import datetime

def generate_final_output(input_documents: List[str], persona: str, job_to_be_done: str, ranked_sections_with_snippets: List[Dict]) -> str:
    """
    Generates the final structured JSON output as a string, matching the required format.
    
    This function takes the processed sections and snippets and formats them into
    the final JSON structure expected by the application.
    
    Args:
        input_documents: List of input document names
        persona: User persona/role
        job_to_be_done: Task description
        ranked_sections_with_snippets: List of ranked sections with their snippets
        
    Returns:
        JSON string containing the formatted output
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

    # Assemble final output structure
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