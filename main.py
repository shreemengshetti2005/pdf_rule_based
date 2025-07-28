import os
import json
from pdf_processor import extract_sections_from_pdf
from relevance_ranker import rank_sections_by_relevance, rerank_for_diversity
from snippet_extractor import extract_top_snippets
from output_formatter import generate_final_output

INPUT_FOLDER = "input"
OUTPUT_FOLDER = "output"
INPUT_JSON_FILE = "challenge1b_input.json"
OUTPUT_JSON_FILE = "challenge1b_output.json"
CHECKING_LOG_PATH = os.path.join(OUTPUT_FOLDER, "checking.json")


def log_sections_for_debug(sections):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    existing_logs = []

    # Safely read existing checking.json if it exists and is valid
    if os.path.exists(CHECKING_LOG_PATH):
        try:
            with open(CHECKING_LOG_PATH, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    existing_logs = json.loads(content)
        except json.JSONDecodeError:
            print(f"⚠️ Warning: 'checking.json' was invalid or empty. Resetting it.")
            existing_logs = []

    # Append new section data
    for sec in sections:
        existing_logs.append({
            "section_title": sec.get("section_title", "Untitled"),
            "paragraph": sec.get("content", "").strip()
        })

    # Save updated logs
    with open(CHECKING_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(existing_logs, f, indent=2, ensure_ascii=False)


def main():
    print("🚀 Starting Hackathon Pipeline (Offline + CPU)...")

    # 1. Load input JSON
    input_path = os.path.join(INPUT_FOLDER, INPUT_JSON_FILE)
    if not os.path.exists(input_path):
        print(f"❌ Fatal Error: {INPUT_JSON_FILE} not found in {INPUT_FOLDER}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    persona = data.get("persona", {}).get("role", "").strip()
    job = data.get("job_to_be_done", {}).get("task", "").strip()
    docs_info = data.get("documents", [])
    doc_filenames = [d.get("filename") for d in docs_info]

    # 2. Extract sections from PDFs
    all_sections = []
    print("\n--- 📄 Processing PDFs ---")
    for pdf_filename in doc_filenames:
        pdf_path = os.path.join(INPUT_FOLDER, pdf_filename)
        if os.path.exists(pdf_path):
            sections = extract_sections_from_pdf(pdf_path)
            log_sections_for_debug(sections)  # 🔍 Safe logging of titles and content
            all_sections.extend(sections)
            print(f"✅ Extracted {len(sections)} sections from {pdf_filename}")
        else:
            print(f"⚠️ Warning: Skipping missing file {pdf_filename}")

    if not all_sections:
        print("❌ Fatal Error: No sections were extracted from any PDF.")
        return

    # 3. Rank all sections
    print("\n--- 🧠 Ranking Sections for Relevance ---")
    ranked_sections = rank_sections_by_relevance(all_sections, persona, job)
    if not ranked_sections:
        print("❌ Fatal Error: Section ranking failed.")
        return

    # 4. Re-rank for diversity
    print("\n--- 🌿 Applying Diversity Re-ranking ---")
    diverse_top_sections = rerank_for_diversity(ranked_sections, top_n=5)

    print("\n🏆 Final Top 5 Sections Selected:")
    for i, sec in enumerate(diverse_top_sections, 1):
        print(f"{i}. {sec['document']} (p.{sec['page_number']}): '{sec['section_title']}' | Score: {sec['relevance_score']:.3f}")

    # 5. Extract snippets with top_k=20
    print("\n--- ✂️ Extracting Diverse Snippets (MMR) ---")
    final_sections_with_snippets = []
    for idx, sec in enumerate(diverse_top_sections, 1):
        snippets = extract_top_snippets(
            section_content=sec["content"],
            header=sec["section_title"],
            persona=persona,
            job_to_be_done=job,
            top_k=5  # Changed from 3 to 20
        )
        if not snippets:
            first_sentence = sec.get("content", "").split('.')[0]
            snippets = [{"refined_text": first_sentence + '.', "relevance_score": 0.0}]

        sec["top_snippets"] = snippets
        sec["importance_rank"] = idx
        final_sections_with_snippets.append(sec)

    # 6. Output JSON
    print("\n--- 💾 Generating Final Output JSON ---")
    final_output_str = generate_final_output(
        input_documents=[d.get("title", "") for d in docs_info],
        persona=persona,
        job_to_be_done=job,
        ranked_sections_with_snippets=final_sections_with_snippets
    )

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = os.path.join(OUTPUT_FOLDER, OUTPUT_JSON_FILE)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output_str)

    print(f"\n🎉 Success! Pipeline finished. Output saved to {output_path}")
    print(f"📊 Extracted up to 20 snippets per section from top 5 sections")


if __name__ == "__main__":
    main()
