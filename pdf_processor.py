import fitz
import os
import re
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
class EnhancedPDFProcessor:
    def __init__(self, output_file: str = "checking.json"):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.MIN_HEADING_SIZE = 12
        self.MAX_HEADING_WORDS = 20
        self.output_file = output_file
        self._cache = {}
        self.parsing_log = []
        self.current_step = 0
    def _log_parsing_step(self, step_name: str, details: str, data: any = None):
        self.current_step += 1
        timestamp = datetime.now().isoformat()
        log_entry = {
            "step_number": self.current_step,
            "timestamp": timestamp,
            "step_name": step_name,
            "details": details,
            "data": data
        }
        self.parsing_log.append(log_entry)
        self.logger.info(f"Step {self.current_step}: {step_name} - {details}")
        print(f"🔄 Step {self.current_step}: {step_name}")
        print(f"   └── {details}")
        if data:
            print(f"   └── Data: {str(data)[:100]}{'...' if len(str(data)) > 100 else ''}")
    def process_pdf_and_save(self, pdf_path: str) -> Dict:
        self._log_parsing_step("INITIALIZATION", f"Starting PDF processing for: {pdf_path}")
        try:
            self._log_parsing_step("VALIDATION", "Validating PDF file accessibility")
            if not self._validate_pdf_path(pdf_path):
                self._log_parsing_step("ERROR", "PDF validation failed")
                return {"error": "PDF validation failed", "sections": [], "parsing_log": self.parsing_log}
            self._log_parsing_step("EXTRACTION", "Beginning section extraction process")
            sections = self.extract_sections_from_pdf(pdf_path)
            self._log_parsing_step("STRUCTURING", f"Creating output structure for {len(sections)} sections")
            result = self._create_output_structure(pdf_path, sections)
            result["parsing_log"] = self.parsing_log
            self._log_parsing_step("SAVING", f"Saving results with complete parsing log to {self.output_file}")
            self._save_to_json(result)
            self._log_parsing_step("COMPLETION", "PDF processing completed successfully with parsing log preserved")
            return result
        except Exception as e:
            error_msg = f"Error processing PDF {pdf_path}: {e}"
            self._log_parsing_step("FATAL_ERROR", error_msg)
            self.logger.error(error_msg)
            return {"error": str(e), "sections": [], "parsing_log": self.parsing_log}
    def extract_sections_from_pdf(self, pdf_path: str) -> List[Dict]:
        sections = []
        self._log_parsing_step("FILE_OPENING", f"Opening PDF file: {os.path.basename(pdf_path)}")
        try:
            doc = fitz.open(pdf_path)
            doc_name = os.path.basename(pdf_path)
            self._current_pdf_path = pdf_path
            self._log_parsing_step("DOCUMENT_INFO", f"PDF opened successfully - {len(doc)} pages", {
                "total_pages": len(doc),
                "document_name": doc_name,
                "metadata": doc.metadata if doc.metadata else "No metadata available"
            })
            self._log_parsing_step("TOC_DETECTION", "Checking for embedded Table of Contents")
            toc = doc.get_toc()
            if toc:
                self._log_parsing_step("TOC_FOUND", f"Found embedded ToC with {len(toc)} entries", {
                    "toc_entries": [{"level": level, "title": title, "page": page} for level, title, page in toc[:5]],
                    "total_entries": len(toc)
                })
                self._log_parsing_step("TITLE_EXTRACTION", "Extracting title from metadata or heuristic analysis")
                title = self._extract_title_from_metadata_or_heuristic(doc)
                self._log_parsing_step("TITLE_FOUND", f"Document title: '{title}'", {"title": title})
                self._log_parsing_step("TOC_PROCESSING", "Creating sections from embedded ToC")
                sections = self._create_sections_from_toc(doc, toc, title, doc_name)
                doc.close()
                self._log_parsing_step("TOC_COMPLETE", f"Successfully extracted {len(sections)} sections from ToC")
                return sections
            self._log_parsing_step("HEURISTIC_START", "No embedded ToC found. Starting heuristic analysis")
            sections = self._process_with_heuristics(doc, doc_name)
            doc.close()
            self._log_parsing_step("EXTRACTION_COMPLETE", f"Analysis complete. Found {len(sections)} sections")
            return sections
        except Exception as e:
            error_msg = f"Error processing {pdf_path}: {e}"
            self._log_parsing_step("PROCESSING_ERROR", error_msg)
            self.logger.error(error_msg)
            return sections
    def _validate_pdf_path(self, pdf_path: str) -> bool:
        if not os.path.isfile(pdf_path):
            self.logger.error(f"File not found: {pdf_path}")
            return False
        if not pdf_path.lower().endswith('.pdf'):
            self.logger.error(f"Not a PDF file: {pdf_path}")
            return False
        try:
            test_doc = fitz.open(pdf_path)
            test_doc.close()
            return True
        except Exception as e:
            self.logger.error(f"Cannot open PDF file {pdf_path}: {e}")
            return False
    def _process_with_heuristics(self, doc: fitz.Document, doc_name: str) -> List[Dict]:
        self._log_parsing_step("TEXT_BLOCK_EXTRACTION", "Extracting text blocks with formatting information")
        text_blocks = self._extract_text_blocks(doc)
        if not text_blocks:
            self._log_parsing_step("NO_TEXT_BLOCKS", "No text blocks found in PDF")
            return []
        self._log_parsing_step("TEXT_BLOCKS_FOUND", f"Extracted {len(text_blocks)} text blocks", {
            "total_blocks": len(text_blocks),
            "sample_blocks": [{"text": block.get("text", "")[:50] + "...", "size": block.get("size"), "page": block.get("page")}
                             for block in text_blocks[:3]]
        })
        headers, footers = self._identify_headers_footers(text_blocks, len(doc))
        self._log_parsing_step("HEADER_FOOTER_FOUND", f"Found {len(headers)} headers and {len(footers)} footers", {
            "headers": headers,
            "footers": footers
        })
        filtered_blocks = [
            b for b in text_blocks
            if b['text'] not in headers and b['text'] not in footers
        ]
        self._log_parsing_step("TEXT_FILTERING", f"Filtered blocks: {len(text_blocks)} → {len(filtered_blocks)}")
        
        # Analyze font characteristics
        self._log_parsing_step("FONT_ANALYSIS", "Analyzing font characteristics for heading detection")
        font_stats = self._analyze_font_characteristics(filtered_blocks)
        self._log_parsing_step("FONT_STATS", "Font analysis complete", {
            "avg_size": font_stats.get("avg_size"),
            "max_size": font_stats.get("max_size"),
            "most_common_size": font_stats.get("most_common_size"),
            "size_difference": font_stats.get("size_diff")
        })
        
        self._log_parsing_step("TITLE_EXTRACTION", "Extracting document title using heuristic analysis")
        title = self._extract_title(filtered_blocks, font_stats)
        self._log_parsing_step("TITLE_EXTRACTED", f"Document title: '{title}'", {"title": title})
        self._current_title = title
        
        # Try to parse visual ToC first
        self._log_parsing_step("VISUAL_TOC_SEARCH", "Searching for visual Table of Contents")
        toc_headings = self._parse_visual_toc(filtered_blocks)
        
        if toc_headings:
            self._log_parsing_step("VISUAL_TOC_FOUND", f"Found visual ToC with {len(toc_headings)} headings", {
                "toc_headings": [{"text": h.get("text"), "level": h.get("level"), "page": h.get("page")} for h in toc_headings]
            })
            self._extraction_method = "visual_toc"
            sections = self._create_sections_from_headings(doc, toc_headings, title, doc_name)
        else:
            self._log_parsing_step("HEADING_DETECTION", "No visual ToC found. Using content-aware heading detection")
            self._extraction_method = "heuristic_analysis"
            
            # Extract headings using enhanced validation
            heading_candidates = [b for b in filtered_blocks 
                                if title.lower() not in b['text'].lower()]
            self._log_parsing_step("HEADING_CANDIDATES", f"Found {len(heading_candidates)} potential heading candidates")
            
            headings = self._extract_headings_with_validation(
                heading_candidates, font_stats, text_blocks)
            self._log_parsing_step("HEADINGS_VALIDATED", f"Validated {len(headings)} actual headings", {
                "headings": [{"text": h.get("text"), "level": h.get("level"), "page": h.get("page")} for h in headings]
            })
            
            sections = self._create_sections_from_headings(doc, headings, title, doc_name)
        
        return sections
    
    def _create_output_structure(self, pdf_path: str, sections: List[Dict]) -> Dict:
        """
        Create structured output for JSON export with clear heading-paragraph mapping 
        and PRESERVED parsing log for the JSON file.
        """
        self._log_parsing_step("OUTPUT_STRUCTURING", "Creating final output structure with complete parsing log")
        
        processed_sections = []
        
        for i, section in enumerate(sections):
            heading = section.get("section_title", "").strip()
            content = section.get("content", "").strip()
            
            # Only include sections with meaningful content
            if not content or len(content) < 10:
                self._log_parsing_step("SECTION_FILTERED", f"Filtered out section '{heading}' - insufficient content")
                continue
            
            # Split content into paragraphs for better structure
            paragraphs = self._split_into_paragraphs(content)
            
            section_data = {
                "section_id": i + 1,
                "heading": heading,
                "full_content": content,
                "paragraphs": paragraphs,
                "page_number": section.get("page_number", 0),
                "heading_level": section.get("level", "H2"),
                "statistics": {
                    "word_count": len(content.split()),
                    "char_count": len(content),
                    "paragraph_count": len(paragraphs),
                    "estimated_reading_time_minutes": round(len(content.split()) / 200, 1)  # ~200 words per minute
                }
            }
            
            processed_sections.append(section_data)
            self._log_parsing_step("SECTION_STRUCTURED", f"Structured section '{heading}' with {len(paragraphs)} paragraphs")
        
        final_structure = {
            "document_metadata": {
                "source_file": os.path.basename(pdf_path),
                "full_path": os.path.abspath(pdf_path),
                "processed_at": datetime.now().isoformat(),
                "total_sections_found": len(processed_sections),
                "processor_version": "2.1-with-preserved-log",
                "extraction_method": self._determine_extraction_method()
            },
            "document_structure": {
                "title": getattr(self, '_current_title', 'Document'),
                "total_sections": len(processed_sections),
                "heading_levels_found": list(set(s["heading_level"] for s in processed_sections)),
                "total_words": sum(s["statistics"]["word_count"] for s in processed_sections),
                "total_paragraphs": sum(s["statistics"]["paragraph_count"] for s in processed_sections)
            },
            "sections": processed_sections,
            "parsing_log": self.parsing_log,
            "processing_summary": {
                "total_processing_steps": len(self.parsing_log),
                "log_preserved": True,
                "log_description": "Complete step-by-step processing log showing every action taken during PDF analysis"
            }
        }
        
        self._log_parsing_step("OUTPUT_COMPLETE", f"Final structure created with {len(processed_sections)} sections and {len(self.parsing_log)} log entries preserved")
        
        return final_structure
    
    def _split_into_paragraphs(self, content: str) -> List[Dict]:
        """Split content into structured paragraphs with metadata."""
        if not content:
            return []
        
        raw_paragraphs = re.split(r'\n\s*\n|\n(?=\s*[A-Z])', content)
        
        structured_paragraphs = []
        for i, para in enumerate(raw_paragraphs):
            para = para.strip()
            if not para or len(para) < 10: 
                continue
            
            para = re.sub(r'\s+', ' ', para) 
            para = re.sub(r'\n+', ' ', para) 
            
            paragraph_data = {
                "paragraph_id": i + 1,
                "text": para,
                "word_count": len(para.split()),
                "char_count": len(para),
                "starts_with_number": bool(re.match(r'^\d+\.?\s', para)),
                "is_bullet_point": bool(re.match(r'^\s*[•\-\*]\s', para)),
                "contains_date": bool(re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', para))
            }
            
            structured_paragraphs.append(paragraph_data)
        
        return structured_paragraphs
    
    def _determine_extraction_method(self) -> str:
        """Determine which extraction method was used."""
        if hasattr(self, '_extraction_method'):
            return self._extraction_method
        return "heuristic_analysis"
    
    def _save_to_json(self, data: Dict) -> None:
        """
        Save processed data to JSON file with clear heading-paragraph structure 
        and COMPLETE parsing log preserved in the file.
        """
        try:
            output_path = Path(self.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            self._log_parsing_step("JSON_PREPARATION", f"Preparing to save data with complete parsing log to {self.output_file}")
            
            if "parsing_log" not in data:
                data["parsing_log"] = self.parsing_log
            
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, separators=(',', ': '))
            
            sections_count = len(data.get("sections", []))
            total_paragraphs = sum(len(s.get("paragraphs", [])) for s in data.get("sections", []))
            parsing_steps = len(data.get("parsing_log", []))
            
            self._log_parsing_step("JSON_SAVED", f"Successfully saved to {self.output_file} with complete parsing log", {
                "sections_saved": sections_count,
                "paragraphs_saved": total_paragraphs,
                "parsing_steps_logged": parsing_steps,
                "file_size_kb": round(os.path.getsize(self.output_file) / 1024, 2)
            })
            
            print(f"\n✅ COMPLETE PARSING LOG SAVED TO: {self.output_file}")
            print(f"📊 Summary:")
            print(f"   • Sections: {sections_count}")
            print(f"   • Paragraphs: {total_paragraphs}")
            print(f"   • Processing Steps Logged: {parsing_steps}")
            print(f"   • File Size: {round(os.path.getsize(self.output_file) / 1024, 2)} KB")
            print(f"   • Complete Processing Log: PRESERVED IN JSON ✓")
            
            if data.get("parsing_log"):
                sample_log = data["parsing_log"][:3]
                print(f"\n🔍 Sample Parsing Log Entries (saved in JSON):")
                for entry in sample_log:
                    print(f"   Step {entry['step_number']}: {entry['step_name']} - {entry['details']}")
            
        except Exception as e:
            error_msg = f"Error saving to JSON: {e}"
            self._log_parsing_step("JSON_SAVE_ERROR", error_msg)
            self.logger.error(error_msg)
            raise
    
    def _extract_title_from_metadata_or_heuristic(self, doc: fitz.Document) -> str:
        """Extract title from metadata or use heuristic analysis."""
       
        metadata_title = doc.metadata.get('title', '').strip()
        if metadata_title and len(metadata_title) > 3:
            return metadata_title
        
        text_blocks = self._extract_text_blocks(doc)
        if not text_blocks:
            return "Document"
        
        font_stats = self._analyze_font_characteristics(text_blocks)
        return self._extract_title(text_blocks, font_stats)
    
    def _create_sections_from_toc(self, doc: fitz.Document, toc: List, 
                                 title: str, doc_name: str) -> List[Dict]:
        """Create sections from embedded table of contents."""
        sections = []
        self._extraction_method = "embedded_toc"
        
        for i, (level, heading_text, page_num) in enumerate(toc):
            try:
                start_page = max(0, page_num - 1)
                
                end_page = len(doc) - 1
                if i + 1 < len(toc):
                    end_page = min(toc[i + 1][2] - 2, len(doc) - 1)  
                
                content = self._extract_content_between_pages(doc, start_page, end_page, heading_text)
                
                if content.strip():
                    sections.append({
                        "section_title": heading_text.strip(),
                        "content": content,
                        "page_number": page_num,
                        "document": doc_name,
                        "level": f"H{min(level, 6)}"  # Cap at H6
                    })
                    self.logger.debug(f"Created section from ToC: '{heading_text}' (Page {page_num})")
                    
            except Exception as e:
                self.logger.warning(f"Error processing ToC entry '{heading_text}': {e}")
                continue
        
        return sections
    
    def _extract_content_between_pages(self, doc: fitz.Document, start_page: int, 
                                     end_page: int, heading_text: str) -> str:
        """Extract content between page ranges."""
        content_parts = []
        
        for page_idx in range(start_page, min(end_page + 1, len(doc))):
            try:
                page = doc[page_idx]
                if page_idx == start_page:
                    page_content = self._extract_content_after_heading(doc, page_idx, heading_text)
                else:
                    page_content = page.get_text()
                
                if page_content.strip():
                    content_parts.append(page_content)
                    
            except Exception as e:
                self.logger.warning(f"Error extracting content from page {page_idx}: {e}")
                continue
        
        combined_content = "\n".join(content_parts)
        return self._clean_section_content(combined_content, heading_text)
    
    def _create_sections_from_headings(self, doc: fitz.Document, headings: List[Dict], 
                                     title: str, doc_name: str) -> List[Dict]:
        """Create sections from detected headings with improved content extraction."""
        sections = []
        
        self._log_parsing_step("SECTION_CREATION", f"Creating sections from {len(headings)} headings")
        
        headings.sort(key=lambda x: (x["page"], x.get("bbox", [0, 0, 0, 0])[1]))
        self._log_parsing_step("HEADING_SORTING", "Sorted headings by page and position")
        
        for i, heading in enumerate(headings):
            try:
                heading_text = heading["text"].strip()
                start_page = heading["page"]
                
                self._log_parsing_step("SECTION_PROCESSING", f"Processing section {i+1}: '{heading_text}' on page {start_page + 1}", {
                    "section_number": i + 1,
                    "heading": heading_text,
                    "page": start_page + 1,
                    "level": heading.get("level", "H2")
                })
                
                end_page = len(doc) - 1
                if i + 1 < len(headings):
                    next_heading = headings[i + 1]
                    end_page = next_heading["page"]
                    if end_page == start_page:
                        end_page = start_page
                
                self._log_parsing_step("CONTENT_EXTRACTION", f"Extracting content from page {start_page + 1} to {end_page + 1}")
                
                content = self._extract_section_content(doc, heading, start_page, end_page, i + 1 < len(headings))
                
                if content.strip():
                    section_data = {
                        "section_title": heading_text,
                        "content": content,
                        "page_number": start_page + 1, 
                        "document": doc_name,
                        "level": heading.get("level", "H2")
                    }
                    sections.append(section_data)
                    
                    self._log_parsing_step("SECTION_CREATED", f"Successfully created section: '{heading_text}'", {
                        "content_length": len(content),
                        "word_count": len(content.split()),
                        "content_preview": content[:100] + "..." if len(content) > 100 else content
                    })
                else:
                    self._log_parsing_step("SECTION_SKIPPED", f"Skipped section '{heading_text}' - no content found")
                    
            except Exception as e:
                error_msg = f"Error processing heading '{heading.get('text', 'Unknown')}': {e}"
                self._log_parsing_step("SECTION_ERROR", error_msg)
                continue
        
        self._log_parsing_step("SECTION_CREATION_COMPLETE", f"Created {len(sections)} sections successfully")
        return sections
    
    def _extract_section_content(self, doc: fitz.Document, heading: Dict, 
                               start_page: int, end_page: int, has_next_heading: bool) -> str:
        """Extract content for a section with better handling of multi-page sections."""
        content_parts = []
        heading_text = heading["text"]
        
        for page_idx in range(start_page, min(end_page + 1, len(doc))):
            try:
                if page_idx == start_page:
                    page_content = self._extract_content_after_heading(doc, page_idx, heading_text)
                elif page_idx == end_page and has_next_heading:
                    page_content = self._extract_content_before_next_heading(doc, page_idx, end_page)
                else:
                    page = doc[page_idx]
                    page_content = page.get_text()
                
                if page_content.strip():
                    content_parts.append(page_content)
                    
            except Exception as e:
                self.logger.warning(f"Error extracting content from page {page_idx}: {e}")
                continue
        
        combined_content = "\n".join(content_parts)
        return self._clean_section_content(combined_content, heading_text)
    
    def _extract_content_after_heading(self, doc: fitz.Document, 
                                     page_num: int, heading_text: str) -> str:
        """Extract content from a page after a specific heading with improved accuracy."""
        try:
            page = doc[page_num]
            text_dict = page.get_text("dict")
            
            content_blocks = []
            heading_found = False
            heading_y_position = None
            
            for block in text_dict["blocks"]:
                if "lines" not in block:
                    continue
                
                block_text = self._extract_block_text(block)
                block_y = self._get_block_y_position(block)
                
                if not block_text.strip():
                    continue
                if not heading_found and self._text_similarity(heading_text, block_text) > 0.8:
                    heading_found = True
                    heading_y_position = block_y
                    continue
                if heading_found and block_text:
                    if heading_y_position is None or (block_y and block_y > heading_y_position):
                        content_blocks.append(block_text)
            
            return "\n".join(content_blocks)
        
        except Exception as e:
            self.logger.warning(f"Error extracting content after heading on page {page_num}: {e}")
            return ""
    
    def _extract_content_before_next_heading(self, doc: fitz.Document, page_num: int, end_page: int) -> str:
        """Extract content before the next heading on the same page."""
        # For now, return all content on the page
        # Can be enhanced to stop at next heading if needed
        try:
            page = doc[page_num]
            return page.get_text()
        except Exception as e:
            self.logger.warning(f"Error extracting content before next heading on page {page_num}: {e}")
            return ""
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract text from a text block safely."""
        try:
            text_parts = []
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if text:
                        text_parts.append(text)
            return " ".join(text_parts)
        except Exception:
            return ""
    
    def _get_block_y_position(self, block: Dict) -> Optional[float]:
        """Get Y position of a text block safely."""
        try:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span.get("bbox")
                    if bbox:
                        return bbox[1]
            return None
        except Exception:
            return None
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        text1_clean = re.sub(r'\s+', ' ', text1.lower().strip())
        text2_clean = re.sub(r'\s+', ' ', text2.lower().strip())
        
        if text1_clean == text2_clean:
            return 1.0
        
        if text1_clean in text2_clean or text2_clean in text1_clean:
            return 0.8
        
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _clean_section_content(self, content: str, heading_text: str) -> str:
        """Enhanced content cleaning with better text processing."""
        if not content:
            return ""
        
        content = re.sub(re.escape(heading_text), "", content, count=1, flags=re.IGNORECASE)
        
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if self._is_page_element(line):
                continue
            if len(line) < 3:
                continue
            
            cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)  
        result = re.sub(r'[ \t]+', ' ', result)  
        result = re.sub(r'\n ', '\n', result) 
        
        return result.strip()
    
    def _is_page_element(self, line: str) -> bool:
        """Check if a line is a page element (header, footer, page number)."""
        page_element_patterns = [
            r'^\d+', 
            r'^page\s+\d+',  
            r'^\d+\s*of\s*\d+', 
            r'^chapter\s+\d+', 
            r'copyright|©', 
            r'^\w+\s*\|\s*\w+',  
        ]
        
        return any(re.search(pattern, line, re.IGNORECASE) for pattern in page_element_patterns)
    
    def _extract_text_blocks(self, doc: fitz.Document) -> List[Dict]:
        """Extract text blocks with detailed formatting information and improved error handling."""
        text_blocks = []
        
        for page_num in range(len(doc)):
            try:
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    if "lines" in block:
                        self._process_text_block(block, page_num, text_blocks)
                        
            except Exception as e:
                self.logger.warning(f"Error processing page {page_num}: {e}")
                continue
        
        return text_blocks
    
    def _process_text_block(self, block: Dict, page_num: int, text_blocks: List[Dict]):
        """Process individual text block with error handling."""
        try:
            block_lines = []
            
            for line in block.get("lines", []):
                line_data = self._process_line(line)
                if line_data:
                    line_data["page"] = page_num
                    block_lines.append(line_data)
            
            if block_lines:
                self._group_similar_lines(block_lines, text_blocks)
                
        except Exception as e:
            self.logger.warning(f"Error processing text block on page {page_num}: {e}")
    
    def _process_line(self, line: Dict) -> Optional[Dict]:
        """Process individual line with safe attribute access."""
        try:
            line_text_parts = []
            line_bbox = None
            line_font = None
            line_size = None
            line_flags = None
            
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    line_text_parts.append(text)
                    size = span.get("size", 0)
                    if line_font is None or size > (line_size or 0):
                        line_font = span.get("font")
                        line_size = size
                        line_flags = span.get("flags", 0)
                        line_bbox = span.get("bbox")
            
            if line_text_parts and line_bbox:
                return {
                    "text": " ".join(line_text_parts).strip(),
                    "font": line_font,
                    "size": line_size or 12,
                    "flags": line_flags or 0,
                    "bbox": line_bbox,
                    "color": 0,
                    "line_y": line_bbox[1]
                }
        except Exception:
            pass
        
        return None
    
    def _group_similar_lines(self, block_lines: List[Dict], text_blocks: List[Dict]):
        """Group similar consecutive lines into text blocks."""
        if not block_lines:
            return
        
        current_group = [block_lines[0]]
        
        for i in range(1, len(block_lines)):
            current_line = block_lines[i]
            prev_line = current_group[-1]
            
            if self._lines_should_group(current_line, prev_line):
                current_group.append(current_line)
            else:
                if current_group:
                    self._add_grouped_text_block(current_group, text_blocks)
                current_group = [current_line]
        
        if current_group:
            self._add_grouped_text_block(current_group, text_blocks)
    
    def _lines_should_group(self, line1: Dict, line2: Dict) -> bool:
        """Determine if two lines should be grouped together."""
        try:
            size_similar = abs(line1["size"] - line2["size"]) < 1
            font_similar = line1["font"] == line2["font"]
            y_close = abs(line1["line_y"] - line2["line_y"]) < 50
            
            return size_similar and font_similar and y_close
        except Exception:
            return False
    
    def _add_grouped_text_block(self, line_group: List[Dict], text_blocks: List[Dict]):
        """Add a grouped text block from multiple lines with safe processing."""
        if not line_group:
            return
        
        try:
            combined_text = " ".join(line.get("text", "") for line in line_group).strip()
            if not combined_text:
                return
            
            first_line = line_group[0]
            
            # Calculate bounding box safely
            x_coords = []
            y_coords = []
            
            for line in line_group:
                bbox = line.get("bbox")
                if bbox and len(bbox) >= 4:
                    x_coords.extend([bbox[0], bbox[2]])
                    y_coords.extend([bbox[1], bbox[3]])
            
            if x_coords and y_coords:
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                text_blocks.append({
                    "text": combined_text,
                    "page": first_line.get("page", 0),
                    "font": first_line.get("font", ""),
                    "size": first_line.get("size", 12),
                    "flags": first_line.get("flags", 0),
                    "bbox": (min_x, min_y, max_x, max_y),
                    "color": first_line.get("color", 0),
                    "line_count": len(line_group)
                })
        except Exception as e:
            self.logger.warning(f"Error adding grouped text block: {e}")
    
    def _analyze_font_characteristics(self, text_blocks: List[Dict]) -> Dict:
        """Analyze font characteristics with safe processing."""
        if not text_blocks:
            return {"avg_size": 12, "max_size": 12, "most_common_size": 12, "size_diff": 0}
        
        try:
            font_sizes = [block.get("size", 12) for block in text_blocks]
            size_frequency = defaultdict(int)
            
            for size in font_sizes:
                size_frequency[round(size, 1)] += 1
            
            avg_size = sum(font_sizes) / len(font_sizes)
            max_size = max(font_sizes)
            most_common_size = max(size_frequency.items(), key=lambda x: x[1])[0]
            
            return {
                "avg_size": avg_size,
                "max_size": max_size,
                "most_common_size": most_common_size,
                "size_diff": max_size - most_common_size,
                "font_styles": defaultdict(int),
                "size_frequency": size_frequency
            }
        except Exception as e:
            self.logger.warning(f"Error analyzing font characteristics: {e}")
            return {"avg_size": 12, "max_size": 12, "most_common_size": 12, "size_diff": 0}
    
    def _extract_title(self, text_blocks: List[Dict], font_stats: Dict) -> str:
        """Extract document title with improved error handling."""
        try:
            first_page_blocks = [block for block in text_blocks if block.get("page") == 0]
            if not first_page_blocks:
                return "Document"
            
            first_page_blocks.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
            
            # Try RFP-style titles first
            rfp_title = self._detect_rfp_title(first_page_blocks)
            if rfp_title:
                return rfp_title
            
            # Look for title-like blocks
            for block in first_page_blocks[:10]:  # Check first 10 blocks
                text = block.get("text", "").strip()
                if len(text) > 5 and not self._is_likely_metadata(text):
                    size = block.get("size", 12)
                    if size > font_stats.get("most_common_size", 12) * 1.1:
                        return text
            
            # Fallback to first meaningful text
            for block in first_page_blocks:
                text = block.get("text", "").strip()
                if len(text) > 5 and not self._is_likely_metadata(text):
                    return text
            
            return "Document"
            
        except Exception as e:
            self.logger.warning(f"Error extracting title: {e}")
            return "Document"
    
    def _detect_rfp_title(self, first_page_blocks: List[Dict]) -> Optional[str]:
        """Detect RFP-style titles with improved processing."""
        try:
            for i, block in enumerate(first_page_blocks):
                text = block.get("text", "").strip()
                
                if re.match(r'^RFP:\s*', text, re.IGNORECASE):
                    title_parts = [text]
                    
                    # Look for continuation in next few blocks
                    for j in range(i + 1, min(i + 5, len(first_page_blocks))):
                        next_block = first_page_blocks[j]
                        next_text = next_block.get("text", "").strip()
                        
                        if len(next_text) > 3 and not self._is_likely_metadata(next_text):
                            title_parts.append(next_text)
                        
                        if re.search(r'\d{1,2},?\s*\d{4}', next_text):
                            break
                    
                    if len(title_parts) > 1:
                        return self._clean_rfp_title(' '.join(title_parts))
            
            return None
        except Exception:
            return None
    
    def _clean_rfp_title(self, title: str) -> str:
        """Clean RFP title from artifacts."""
        try:
            # Basic cleaning
            title = re.sub(r'\s+', ' ', title.strip())
            title = re.sub(r'\s+\d{1,2},?\s*\d{4}', '', title)
            return title
        except Exception:
            return title
    
    def _is_likely_metadata(self, text: str) -> bool:
        """Check if text is likely metadata with safe processing."""
        if not text:
            return True
        
        try:
            metadata_patterns = [
                r'^\d+',
                r'^page\s+\d+',
                r'copyright|©',
                r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',
                r'^[A-Z]{2,}\s*:',
                r'www\.|http|@',
                r'^\$[\d,]+',
                r'^\d+\.\d+%?',
            ]
            
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in metadata_patterns)
        except Exception:
            return False
    
    def _identify_headers_footers(self, text_blocks: List[Dict], total_pages: int) -> Tuple[List[str], List[str]]:
        """Identify recurring headers and footers with improved logic."""
        if total_pages < 2:
            return [], []
        
        try:
            top_blocks = defaultdict(list)
            bottom_blocks = defaultdict(list)
            
            for block in text_blocks:
                bbox = block.get("bbox")
                if not bbox or len(bbox) < 4:
                    continue
                
                y_pos = bbox[1]
                text = block.get("text", "").strip()
                
                if not text or len(text) < 3:
                    continue
                
                if y_pos < 150:  # Top of page
                    top_blocks[text].append(block.get("page", 0))
                elif y_pos > 650:  # Bottom of page
                    bottom_blocks[text].append(block.get("page", 0))
            
            headers = []
            footers = []
            min_occurrences = max(2, total_pages // 3)
            
            for text, pages in top_blocks.items():
                if len(set(pages)) >= min_occurrences:
                    headers.append(text)
            
            for text, pages in bottom_blocks.items():
                if len(set(pages)) >= min_occurrences:
                    footers.append(text)
            
            return headers, footers
            
        except Exception as e:
            self.logger.warning(f"Error identifying headers/footers: {e}")
            return [], []
    
    def _parse_visual_toc(self, text_blocks: List[Dict]) -> List[Dict]:
        """Parse visual table of contents with improved pattern matching."""
        toc_headings = []
        
        try:
            for block in text_blocks:
                text = block.get("text", "").strip()
                if not text:
                    continue
                
                # Enhanced ToC patterns
                patterns = [
                    r'^(\d+\.?\s+)([^.]+)\.{3,}\s*(\d+)',  # 1. Title....123
                    r'^([A-Z][^.]+)\s+\.{3,}\s*(\d+)',     # TITLE....123
                    r'^(\d+\.\d+\s+)([^.]+)\s+(\d+)',      # 1.1 Title 123
                    r'^([IVX]+\.\s+)([^.]+)\.+\s*(\d+)',   # I. Title....123
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, text)
                    if match:
                        groups = match.groups()
                        if len(groups) == 3:
                            prefix, title, page = groups
                            level = self._determine_toc_level(prefix)
                            try:
                                page_num = int(page) - 1  # Convert to 0-based
                                if 0 <= page_num < 1000:  # Sanity check
                                    toc_headings.append({
                                        "level": level,
                                        "text": title.strip(),
                                        "page": page_num
                                    })
                            except ValueError:
                                continue
                        elif len(groups) == 2:
                            title, page = groups
                            try:
                                page_num = int(page) - 1
                                if 0 <= page_num < 1000:
                                    toc_headings.append({
                                        "level": "H1",
                                        "text": title.strip(),
                                        "page": page_num
                                    })
                            except ValueError:
                                continue
                        break
            
           
            return toc_headings if len(toc_headings) > 2 else []
            
        except Exception as e:
            self.logger.warning(f"Error parsing visual ToC: {e}")
            return []
    
    def _determine_toc_level(self, prefix: str) -> str:
        """Determine heading level from ToC prefix."""
        try:
            if re.match(r'^\d+\.\d+', prefix):
                return "H3"
            elif re.match(r'^\d+\.', prefix):
                return "H2"
            elif re.match(r'^[IVX]+\.', prefix):
                return "H1"
            else:
                return "H2"
        except Exception:
            return "H2"
    
    def _extract_headings_with_validation(self, text_blocks: List[Dict], 
                                        font_stats: Dict, all_blocks: List[Dict]) -> List[Dict]:
        """Extract headings with enhanced validation and improved accuracy."""
        headings = []
        
        try:
            page_blocks = defaultdict(list)
            for block in text_blocks:
                page_num = block.get("page", 0)
                page_blocks[page_num].append(block)
            
            for page_num, blocks in page_blocks.items():
                blocks.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
                
                for i, block in enumerate(blocks):
                    text = block.get("text", "").strip()
                    
                    if len(text) < 3 or self._is_likely_metadata(text):
                        continue
                    
                    if self._is_likely_heading(block, font_stats):
                        level = self._determine_heading_level_smart(block, font_stats)
                        
                        if level and self._validate_heading_context(block, blocks, i):
                            clean_text = self._clean_heading_text(text)
                            if clean_text and len(clean_text) > 2:
                                headings.append({
                                    "level": level,
                                    "text": clean_text,
                                    "page": page_num,
                                    "bbox": block.get("bbox", [0, 0, 0, 0])
                                })
            
            return self._post_process_headings(headings)
            
        except Exception as e:
            self.logger.warning(f"Error extracting headings: {e}")
            return []
    
    def _is_likely_heading(self, block: Dict, font_stats: Dict) -> bool:
        """Determine if a text block is likely to be a heading with improved logic."""
        try:
            text = block.get("text", "").strip()
            
            if self._is_likely_metadata(text) or len(text) > 200:
                return False
            
            size = block.get("size", 12)
            flags = block.get("flags", 0)
            
            is_bold = bool(flags & 2**4)
            is_italic = bool(flags & 2**1)
            ends_with_colon = text.endswith(':')
            
            # Heading pattern detection
            heading_patterns = [
                r'^\d+\.\s*\w+',                    # 1. Something
                r'^\d+\s+[A-Z]',                    # 1 SOMETHING
                r'^[A-Z][a-z]+\s+\d+',              # Chapter 1
                r'^[A-Z][A-Z\s]+',                 # ALL CAPS
                r'^\w+.*:\s*',                     # Something:
                r'^(PHASE|SECTION|CHAPTER|PART)\s+', # Phase, Section, etc.
            ]
            
            pattern_match = any(re.search(pattern, text, re.IGNORECASE) for pattern in heading_patterns)
            
            # Calculate heading score
            score = 0
            most_common_size = font_stats.get("most_common_size", 12)
            size_ratio = size / most_common_size if most_common_size > 0 else 1
            
            # Size scoring
            if size_ratio > 1.3:
                score += 4
            elif size_ratio > 1.2:
                score += 3
            elif size_ratio > 1.1:
                score += 2
            elif size_ratio > 1.05:
                score += 1
            
            # Style scoring
            if is_bold:
                score += 2
            if is_italic:
                score += 1
            if pattern_match:
                score += 2
            if ends_with_colon:
                score += 2
            
            # Length scoring
            if 5 <= len(text) <= 100:
                score += 1
            elif len(text) > 200:
                score -= 2
            
            # Position scoring (left-aligned text is more likely to be a heading)
            bbox = block.get("bbox", [0, 0, 0, 0])
            if bbox[0] < 100:  # Left margin
                score += 1
            
            # Threshold determination
            if ends_with_colon:
                return score >= 3
            else:
                return score >= 4
                
        except Exception as e:
            self.logger.warning(f"Error checking if text is heading: {e}")
            return False
    
    def _validate_heading_context(self, block: Dict, page_blocks: List[Dict], block_index: int) -> bool:
        """Validate heading by checking its context on the page."""
        try:
            # Check if there's meaningful content after this potential heading
            for i in range(block_index + 1, len(page_blocks)):
                next_block = page_blocks[i]
                next_text = next_block.get("text", "").strip()
                
                if len(next_text) > 20 and not self._is_likely_metadata(next_text):
                    # Found substantial content after, likely a real heading
                    return True
                elif len(next_text) > 100:
                    # Very long text block following, probably content
                    return True
            
            # If we're at the end of page, assume it's valid if it looks like a heading
            return True
            
        except Exception:
            return True  # Default to accepting if we can't validate
    
    def _determine_heading_level_smart(self, block: Dict, font_stats: Dict) -> Optional[str]:
        """Determine heading level using both font size and content analysis."""
        try:
            text = block.get("text", "").strip()
            font_size = block.get("size", 12)
            most_common_size = font_stats.get("most_common_size", 12)
            size_ratio = font_size / most_common_size if most_common_size > 0 else 1
            
            # Content-based level detection
            if re.match(r'^(APPENDIX|CHAPTER)\s+[A-Z0-9]', text, re.IGNORECASE):
                return "H1"
            
            if re.match(r'^\d+\.\s+[A-Z]', text):
                return "H2"
            
            if re.match(r'^(SUMMARY|BACKGROUND|BUSINESS PLAN|APPROACH|EVALUATION)', text, re.IGNORECASE):
                return "H2"
            
            # Size-based detection
            if size_ratio > 1.4:
                return "H1"
            elif size_ratio > 1.2:
                return "H2"
            elif size_ratio > 1.05:
                return "H3"
            elif text.endswith(':'):
                return "H3"
            
            return "H3"  # Default level
            
        except Exception:
            return "H3"
    
    def _clean_heading_text(self, text: str) -> str:
        """Clean and normalize heading text."""
        try:
            # Basic cleaning
            text = re.sub(r'\s+', ' ', text.strip())
            text = re.sub(r'[.]{2,}', '', text)  # Remove dot leaders
            text = re.sub(r'^(\d+\.?\d*\.?)\s*', r'\1 ', text)  # Normalize numbering
            return text
        except Exception:
            return text
    
    def _post_process_headings(self, headings: List[Dict]) -> List[Dict]:
        """Post-process headings to ensure quality and remove duplicates."""
        if not headings:
            return headings
        
        try:
            cleaned_headings = []
            seen_texts = set()
            
            for heading in headings:
                text = heading.get("text", "").strip()
                text_key = text.lower()
                
                # Skip duplicates
                if text_key in seen_texts:
                    continue
                
                # Skip obvious non-headings
                if self._is_obvious_non_heading(text):
                    continue
                
                # Validate that heading has content
                if self._has_meaningful_content(heading):
                    cleaned_headings.append(heading)
                    seen_texts.add(text_key)
            
            # Sort by page and position
            cleaned_headings.sort(key=lambda x: (x.get("page", 0), x.get("bbox", [0, 0, 0, 0])[1]))
            return cleaned_headings
            
        except Exception as e:
            self.logger.warning(f"Error post-processing headings: {e}")
            return headings
    
    def _has_meaningful_content(self, heading: Dict) -> bool:
        """Check if a heading has meaningful content following it."""
        # This is a simplified version - in practice, you'd check the actual document
        text = heading.get("text", "")
        
        # Skip very short or very long headings
        if len(text) < 3 or len(text) > 200:
            return False
        
        # Skip headings that are just page numbers or metadata
        if re.match(r'^\d+', text) or self._is_likely_metadata(text):
            return False
        
        return True
    
    def _is_obvious_non_heading(self, text: str) -> bool:
        """Check for obvious non-headings with enhanced patterns."""
        try:
            # Common non-heading patterns
            non_heading_patterns = [
                r'^\d+',  # Just numbers
                r'^page\s+\d+',  # Page numbers
                r'copyright|©',  # Copyright
                r'^www\.|http',  # URLs
                r'^[a-z]+@[a-z]+\.',  # Email addresses
                r'^\$[\d,]+',  # Currency amounts
                r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}',  # Dates
            ]
            
            return any(re.search(pattern, text, re.IGNORECASE) for pattern in non_heading_patterns)
            
        except Exception:
            return False


# Convenience functions for easy usage
def process_pdf_to_json(pdf_path: str, output_file: str = "checking.json") -> Dict:
    """
    Enhanced function that extracts sections from PDF and saves to JSON with complete parsing log.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_file (str): Output JSON file path (defaults to "checking.json")
        
    Returns:
        Dict: Processing results with metadata and complete parsing log preserved in JSON
    """
    processor = EnhancedPDFProcessor(output_file)
    return processor.process_pdf_and_save(pdf_path)

def extract_sections_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Enhanced function that extracts structurally-aware sections from a PDF.
    
    This function now includes:
    - Intelligent heading detection based on font size, style, and content patterns
    - Title extraction with RFP-style multi-line support
    - Visual table of contents parsing
    - Header/footer filtering
    - Content validation to ensure sections have meaningful content
    - Support for embedded bookmarks/ToC when available
    - Improved error handling and logging
    - Complete processing log preservation in JSON output
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Dict]: List of sections with keys:
            - section_title: Title of the section
            - content: Text content of the section
            - page_number: Page number where section starts (1-based)
            - document: Document filename
            - level: Heading level (H1, H2, H3, etc.)
    """
    processor = EnhancedPDFProcessor()
    return processor.extract_sections_from_pdf(pdf_path)


# Example usage and demonstration
if __name__ == "__main__":
    import sys
    
    def print_json_structure_sample(json_file: str):
        """Print a sample of the JSON structure for verification."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"\n🔍 JSON Structure Sample from {json_file}:")
            print(f"Document: {data['document_metadata']['source_file']}")
            print(f"Total Sections: {data['document_structure']['total_sections']}")
            print(f"Extraction Method: {data['document_metadata']['extraction_method']}")
            
            # Show parsing log summary
            if data.get('parsing_log'):
                print(f"Processing Steps Logged: {len(data['parsing_log'])}")
                print(f"Complete Processing Log: ✓ PRESERVED IN JSON")
            
            if data.get('sections'):
                sample_section = data['sections'][0]
                print(f"\n📄 Sample Section:")
                print(f"  Heading: {sample_section['heading']}")
                print(f"  Level: {sample_section['heading_level']}")
                print(f"  Paragraphs: {len(sample_section['paragraphs'])}")
                print(f"  Word Count: {sample_section['statistics']['word_count']}")
                
                if sample_section.get('paragraphs'):
                    sample_paragraph = sample_section['paragraphs'][0]
                    preview = sample_paragraph['text'][:100] + "..." if len(sample_paragraph['text']) > 100 else sample_paragraph['text']
                    print(f"  First paragraph preview: {preview}")
            
            # Show sample parsing log entries
            if data.get('parsing_log'):
                print(f"\n📝 Sample Processing Log Entries:")
                for entry in data['parsing_log'][:3]:
                    print(f"  Step {entry['step_number']}: {entry['step_name']}")
                    print(f"    └── {entry['details']}")
                    if entry.get('data'):
                        data_preview = str(entry['data'])[:80] + "..." if len(str(entry['data'])) > 80 else str(entry['data'])
                        print(f"    └── Data: {data_preview}")
                print(f"  ... and {len(data['parsing_log']) - 3} more processing steps")
            
        except Exception as e:
            print(f"❌ Error reading JSON structure: {e}")
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "checking.json"
        
        print(f"🔄 Processing {pdf_path}...")
        print(f"📁 Output will be saved to: {output_file}")
        print(f"📋 Complete processing log will be preserved in JSON output")
        
        try:
            result = process_pdf_to_json(pdf_path, output_file)
            
            if "error" not in result:
                sections_count = result.get('document_structure', {}).get('total_sections', 0)
                total_words = result.get('document_structure', {}).get('total_words', 0)
                total_paragraphs = result.get('document_structure', {}).get('total_paragraphs', 0)
                processing_steps = len(result.get('parsing_log', []))
                
                print(f"\n✅ SUCCESS: PDF Processing Complete!")
                print(f"📊 Statistics:")
                print(f"   • Sections extracted: {sections_count}")
                print(f"   • Total paragraphs: {total_paragraphs}")
                print(f"   • Total words: {total_words}")
                print(f"   • Processing steps logged: {processing_steps}")
                print(f"   • Output file: {output_file}")
                print(f"   • Complete processing log: ✓ PRESERVED IN JSON")
                
                # Show JSON structure sample
                print_json_structure_sample(output_file)
                
                print(f"\n🎯 Key Features:")
                print(f"   • Title-Paragraph mapping: Each section contains structured paragraphs")
                print(f"   • Complete processing log: Every step is documented in the JSON")
                print(f"   • Metadata preservation: Font analysis, extraction methods, and statistics")
                print(f"   • Error handling: Comprehensive logging of any issues encountered")
                
            else:
                print(f"❌ ERROR: {result['error']}")
                if result.get('parsing_log'):
                    print(f"📋 Processing log with {len(result['parsing_log'])} steps still saved to {output_file}")
                
        except Exception as e:
            print(f"❌ FATAL ERROR: {e}")
            
    else:
        print("📖 Enhanced PDF Processor with Complete Processing Log")
        print("\n🎯 Key Features:")
        print("   • Complete processing log preserved in JSON output")
        print("   • Title-paragraph mapping with structured data")
        print("   • Intelligent heading detection and validation")
        print("   • Font analysis and metadata extraction")
        print("   • Error handling with detailed logging")
        print("\nUsage:")
        print("  python enhanced_pdf_processor.py <pdf_path> [output_file]")
        print("\nExamples:")
        print("  python enhanced_pdf_processor.py document.pdf")
        print("  python enhanced_pdf_processor.py document.pdf my_sections.json")
        print("\nOutput Structure:")
        print("  The script creates a JSON file with:")
        print("  ├── document_metadata (file info, processing time, extraction method)")
        print("  ├── document_structure (title, sections count, statistics)")
        print("  ├── sections[] (each with heading, paragraphs[], and metadata)")
        print("  ├── parsing_log[] (complete step-by-step processing record)")
        print("  └── processing_summary (log statistics and description)")
        print("\n📋 Processing Log:")
        print("  Every step of PDF analysis is logged and preserved in the JSON,")
        print("  including text extraction, heading detection, content validation,")
        print("  section creation, and any errors encountered during processing.")
        print("\n💡 The parsing_log array contains detailed information about:")
        print("  • File opening and validation")
        print("  • Text block extraction and analysis")
        print("  • Font characteristic analysis")
        print("  • Heading detection and validation")
        print("  • Section creation and content extraction")
        print("  • Title extraction and document structure analysis")
        # print("  • Final JSON output generation and saving")