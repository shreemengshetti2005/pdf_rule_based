import os
import json
from document_parser import extract_sections_from_pdf
from content_ranker import rank_sections_by_relevance, rerank_for_diversity
from text_extractor import extract_top_snippets
from result_generator import generate_final_output

# Configuration constants for the processing pipeline
SOURCE_DIRECTORY = "input"
RESULT_DIRECTORY = "output"
INPUT_JSON_FILENAME = "challenge1b_input.json"
OUTPUT_JSON_FILENAME = "challenge1b_output.json"
DEBUG_LOG_FILEPATH = os.path.join(RESULT_DIRECTORY, "checking.json")


def record_sections_for_verification(section_data):
    """
    Records section data for debugging and verification purposes.
    This function maintains a log of all processed sections.
    """
    os.makedirs(RESULT_DIRECTORY, exist_ok=True)
    previous_logs = []

    # Safely read existing checking.json if it exists and is valid
    if os.path.exists(DEBUG_LOG_FILEPATH):
        try:
            with open(DEBUG_LOG_FILEPATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    previous_logs = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: 'checking.json' was invalid or empty. Resetting it.")
            previous_logs = []

    # Append new section data
    for sec in section_data:
        previous_logs.append({
            "section_title": sec.get("section_title", "Untitled"),
            "paragraph": sec.get("content", "").strip()
        })

    # Save updated logs
    with open(DEBUG_LOG_FILEPATH, "w", encoding="utf-8") as f:
        json.dump(previous_logs, f, indent=2, ensure_ascii=False)


def execute_pipeline():
    """
    Main execution function that orchestrates the entire document processing pipeline.
    This function coordinates all the different processing stages.
    """
    print("🚀 Initializing Document Processing Pipeline (Offline + CPU)...")
    print("📋 Pipeline stages: Load → Parse → Rank → Extract → Generate")

    # Stage 1: Load and parse input configuration
    print("\n=== STAGE 1: Loading Input Configuration ===")
    input_filepath = os.path.join(SOURCE_DIRECTORY, INPUT_JSON_FILENAME)
    if not os.path.exists(input_filepath):
        print(f"❌ Fatal Error: {INPUT_JSON_FILENAME} not found in {SOURCE_DIRECTORY}")
        return

    with open(input_filepath, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    user_role = input_data.get("persona", {}).get("role", "").strip()
    task_description = input_data.get("job_to_be_done", {}).get("task", "").strip()
    document_list = input_data.get("documents", [])
    pdf_filenames = [d.get("filename") for d in document_list]
    
    print(f"✅ Loaded configuration for {len(pdf_filenames)} documents")
    print(f"👤 User role: {user_role}")
    print(f"📝 Task: {task_description}")

    # Stage 2: Extract sections from PDF documents
    complete_sections = []
    print("\n=== STAGE 2: Processing PDF Documents ===")
    for pdf_filename in pdf_filenames:
        pdf_filepath = os.path.join(SOURCE_DIRECTORY, pdf_filename)
        if os.path.exists(pdf_filepath):
            sections = extract_sections_from_pdf(pdf_filepath)
            record_sections_for_verification(sections)  # 🔍 Safe logging of titles and content
            complete_sections.extend(sections)
            print(f"✅ Extracted {len(sections)} sections from {pdf_filename}")
        else:
            print(f"⚠️ Warning: Skipping missing file {pdf_filename}")

    if not complete_sections:
        print("❌ Fatal Error: No sections were extracted from any PDF.")
        return

    print(f"📊 Total sections extracted: {len(complete_sections)}")

    # Stage 3: Rank sections by relevance
    print("\n=== STAGE 3: Ranking Sections for Relevance ===")
    ranked_sections = rank_sections_by_relevance(complete_sections, user_role, task_description)
    if not ranked_sections:
        print("❌ Fatal Error: Section ranking failed.")
        return

    # Stage 4: Apply diversity re-ranking
    print("\n=== STAGE 4: Applying Diversity Re-ranking ===")
    diverse_top_sections = rerank_for_diversity(ranked_sections, top_n=5)

    print("\n🏆 Final Top 5 Sections Selected:")
    for i, sec in enumerate(diverse_top_sections, 1):
        print(f"{i}. {sec['document']} (p.{sec['page_number']}): '{sec['section_title']}' | Score: {sec['relevance_score']:.3f}")

    # Stage 5: Extract diverse snippets
    print("\n=== STAGE 5: Extracting Diverse Snippets (MMR) ===")
    final_sections_with_snippets = []
    for idx, sec in enumerate(diverse_top_sections, 1):
        snippets = extract_top_snippets(
            section_content=sec["content"],
            header=sec["section_title"],
            persona=user_role,
            job_to_be_done=task_description,
            top_k=5  # Changed from 3 to 20
        )
        if not snippets:
            first_sentence = sec.get("content", "").split('.')[0]
            snippets = [{"refined_text": first_sentence + '.', "relevance_score": 0.0}]

        sec["top_snippets"] = snippets
        sec["importance_rank"] = idx
        final_sections_with_snippets.append(sec)

    # Stage 6: Generate final output
    print("\n=== STAGE 6: Generating Final Output JSON ===")
    final_output_str = generate_final_output(
        input_documents=[d.get("title", "") for d in document_list],
        persona=user_role,
        job_to_be_done=task_description,
        ranked_sections_with_snippets=final_sections_with_snippets
    )

    os.makedirs(RESULT_DIRECTORY, exist_ok=True)
    output_filepath = os.path.join(RESULT_DIRECTORY, OUTPUT_JSON_FILENAME)
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write(final_output_str)

    print(f"\n🎉 Success! Pipeline finished. Output saved to {output_filepath}")
    print(f"📊 Extracted up to 20 snippets per section from top 5 sections")
    print("✨ Pipeline execution completed successfully!")


if __name__ == "__main__":
    execute_pipeline()
