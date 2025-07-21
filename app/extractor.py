import fitz
from utils import (
    is_text_block, in_main_region,
    detect_table_rects, block_is_table,
    looks_like_form_field,
    is_numeric_bullet, assign_heading_level
)

def extract_outline(pdf_path: str) -> dict:
    """
    Extract title and H1/H2/H3 outline from PDF, skipping tables, headers/footers, and form fields.
    """
    doc = fitz.open(pdf_path)
    outline = []
    seen = set()

    # Compute global average font size to filter labels
    all_sizes = []
    for page in doc:
        for blk in page.get_text("dict")["blocks"]:
            if not is_text_block(blk): continue
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    sz = sp.get("size", 0)
                    if sz: all_sizes.append(sz)
    avg_global = sum(all_sizes) / len(all_sizes) if all_sizes else 0

    # Analyze page 0 for title candidates
    page0 = doc[0]
    page0_blocks = page0.get_text("dict")["blocks"]
    cand_stats = []  # (text, avg_size)
    for blk in page0_blocks:
        if not is_text_block(blk): continue
        texts, sizes = [], []
        for ln in blk.get("lines", []):
            for sp in ln.get("spans", []):
                t = sp.get("text","\n").strip()
                if t:
                    texts.append(t)
                    sizes.append(sp.get("size",0))
        if texts:
            avg_sz = sum(sizes)/len(sizes)
            text = " ".join(texts)
            cand_stats.append((text, avg_sz))
    # Determine max size in candidates
    max_size = max((sz for _, sz in cand_stats), default=0)
    # Title candidates=within 1pt of max_size
    titles = [t for t, sz in cand_stats if sz >= max_size - 1]
    # Choose longest string
    title = max(titles, key=len) if titles else ""

    # Iterate pages for headings
    for pno, page in enumerate(doc):
        # per-page font stats for heading detection
        page_blocks = [blk for blk in page.get_text("dict")["blocks"] if is_text_block(blk) and in_main_region(blk["bbox"], page.rect.height) and not block_is_table(blk, detect_table_rects(page))]
        page_sizes = []
        for blk in page_blocks:
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    sz = sp.get("size", 0)
                    if sz: page_sizes.append(sz)
        page_max = max(page_sizes) if page_sizes else 0

        for blk in page.get_text("dict")["blocks"]:
            if not is_text_block(blk): continue
            if not in_main_region(blk["bbox"], page.rect.height): continue
            if block_is_table(blk, detect_table_rects(page)): continue

            texts, sizes = [], []
            for ln in blk.get("lines", []):
                for sp in ln.get("spans", []):
                    t = sp.get("text"," ").strip()
                    if t:
                        texts.append(t)
                        sizes.append(sp.get("size",0))
            if not texts: continue

            text = " ".join(texts)
            if text == title or text in seen: continue
            seen.add(text)

            if looks_like_form_field(text): continue
            avg_sz = sum(sizes)/len(sizes)
            # skip tiny
            if avg_sz < avg_global * 1.1: continue

            # title is already detected
            # numbered headings
            if text[0].isdigit() and not is_numeric_bullet(text):
                lvl = assign_heading_level(text)
            # large non-numbered headings (e.g., Revision History)
            elif avg_sz >= page_max * 0.9:
                lvl = "H1"
            else:
                continue

            outline.append({"level": lvl, "text": text, "page": pno})

    return {"title": title, "outline": outline} 
