import fitz
import re

def is_text_block(block: dict) -> bool:
    # Only pure text blocks (type 0)
    return block.get("type", 1) == 0

def in_main_region(bbox: list, page_height: float) -> bool:
    # Exclude top/bottom 10%
    y0, y1 = bbox[1], bbox[3]
    return y0 > 0.1 * page_height and y1 < 0.9 * page_height

def block_is_table(block: dict, table_rects: list) -> bool:
    # Skip if >50% of its area overlaps any table‐like rect
    bb = fitz.Rect(block["bbox"])
    for tr in table_rects:
        inter = bb & tr
        if inter.get_area() / bb.get_area() > 0.5:
            return True
    return False

def detect_table_rects(page) -> list:
    # Any drawing primitives that look like rectangles → possible tables
    rects = []
    for draw in page.get_drawings():
        r = draw.get("rect")
        if r and r.width > 20 and r.height > 10:
            rects.append(fitz.Rect(r))
    return rects

def looks_like_form_field(text: str) -> bool:
    # skip underscored lines or purely blank fields
    return "_" in text and len(text.strip("_")) < len(text)

def is_numeric_bullet(text: str) -> bool:
    return bool(re.fullmatch(r"\d+\.", text))

def assign_heading_level(text: str) -> str:
    # numbering heuristic
    if re.match(r"^\d+\.\d+\.\d+", text):
        return "H3"
    if re.match(r"^\d+\.\d+", text):
        return "H2"
    return "H1"
