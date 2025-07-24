import os, json
from extractor import extract_outline

INPUT = "app/input"
OUTPUT = "app/output"
os.makedirs(OUTPUT, exist_ok=True)

for fn in os.listdir(INPUT):
    if not fn.lower().endswith(".pdf") :
        continue
    src = os.path.join(INPUT, fn)
    print(f"Processing {fn}…")
    data = extract_outline(src)
    dst = os.path.join(OUTPUT, fn.replace(".pdf",".json"))
    with open(dst, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("→", os.path.basename(dst))
