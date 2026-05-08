"""
scan_headings.py
----------------
Bütün PDF-ləri oxuyur və hər table-ın üstündəki başlığı çıxarır.
Nəticəni iki şəkildə göstərir:
  1. Hər unikal başlıq neçə dəfə görünüb (frequency)
  2. Hansı başlıqlar hələ heç bir table type-a map edilməyib (unmapped)

İstifadə:
  python scan_headings.py --input /path/to/pdf/folder
  python scan_headings.py --input /path/to/pdf/folder --limit 50
"""

import re
import sys
import json
import argparse
import traceback
from pathlib import Path
from collections import Counter, defaultdict

import pdfplumber
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers — eyni funksiyalar extract_pipeline_fixed.py-dən
# ---------------------------------------------------------------------------

def fix_double_letter_bug(text: str) -> str:
    if not isinstance(text, str):
        return text
    words = text.split()
    fixed = []
    for w in words:
        n = len(w)
        if n >= 4 and n % 2 == 0 and all(w[i] == w[i + 1] for i in range(0, n, 2)):
            fixed.append(w[::2])
        else:
            fixed.append(w)
    return " ".join(fixed)


def _get_heading_above_table(page, table_bbox: tuple):
    table_top = table_bbox[1]
    try:
        words = page.extract_words()
    except Exception:
        return None
    if not words:
        return None

    above = [w for w in words if w["bottom"] <= table_top]
    if not above:
        return None

    closest_bottom = max(w["bottom"] for w in above)
    if table_top - closest_bottom > 60:
        return None

    heading_words = [w for w in above if abs(w["bottom"] - closest_bottom) <= 4]
    heading_words.sort(key=lambda w: w["x0"])
    raw = " ".join(fix_double_letter_bug(w["text"]) for w in heading_words)
    return re.sub(r"\s+", " ", raw).strip()


# Hazırkı mapping — bunlara uyğun gəlməyənlər "UNMAPPED" kimi işarələnəcək
KNOWN_HEADINGS = {
    "operations":        "Operations Table",
    "drilling fluid":    "Drilling Fluid Table",
    "equipment failure": "Equipment Failure Table",
    "gas reading":       "Gas Reading Table",
    "survey station":    "Survey Table",
    "pore pressure":     "Pore Pressure Table",
    "lithology":         "Lithology Table",
}


def classify(heading: str) -> str:
    if not heading:
        return "NO HEADING"
    h = heading.lower()
    for pattern, ttype in KNOWN_HEADINGS.items():
        if pattern in h:
            return ttype
    return "UNMAPPED"


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def scan_folder(folder_path: str, limit: int = None):
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]

    print(f"Scanning {len(pdf_files)} PDFs...\n")

    heading_counter = Counter()        # heading text → count
    heading_to_type = {}               # heading text → classified type
    unmapped = Counter()               # unmapped heading → count
    errors = []

    for pdf_path in tqdm(pdf_files, desc="Scanning"):
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page in pdf.pages:
                    try:
                        table_objects = page.find_tables()
                    except Exception:
                        continue

                    for tobj in table_objects:
                        heading = _get_heading_above_table(page, tobj.bbox)
                        label = heading if heading else "__NO_HEADING__"
                        heading_counter[label] += 1
                        ttype = classify(heading)
                        heading_to_type[label] = ttype
                        if ttype == "UNMAPPED":
                            unmapped[label] += 1

        except Exception:
            errors.append({"file": pdf_path.name, "error": traceback.format_exc()})

    # ---------------------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------------------

    print("\n" + "="*60)
    print("ALL TABLE HEADINGS (sorted by frequency)")
    print("="*60)
    print(f"{'Count':>6}  {'Mapped To':<28}  Heading")
    print("-"*60)
    for heading, count in heading_counter.most_common():
        ttype = heading_to_type.get(heading, "?")
        print(f"{count:>6}  {ttype:<28}  {heading}")

    print("\n" + "="*60)
    print("UNMAPPED HEADINGS — not assigned to any table type")
    print("="*60)
    if unmapped:
        for heading, count in unmapped.most_common():
            print(f"  {count:>5}x  {repr(heading)}")
        print(f"\n  → Add these to HEADING_TO_TABLE_TYPE in extract_pipeline_fixed.py")
    else:
        print("  None — all headings are mapped!")

    print(f"\n{'='*60}")
    print(f"Total PDFs scanned : {len(pdf_files)}")
    print(f"Total tables found : {sum(heading_counter.values())}")
    print(f"Unmapped headings  : {len(unmapped)}")
    print(f"Errors             : {len(errors)}")

    if errors:
        print("\nFailed PDFs:")
        for e in errors[:10]:
            print(f"  {e['file']}")

    # Save full results to JSON for further analysis
    out = {
        "heading_counts": dict(heading_counter.most_common()),
        "unmapped": dict(unmapped.most_common()),
        "errors": errors,
    }
    out_path = Path(folder_path) / "_heading_scan_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan all PDF table headings")
    parser.add_argument("--input", "-i", required=True, help="Folder with PDF files")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Only scan first N PDFs (for quick testing)")
    args = parser.parse_args()
    scan_folder(args.input, args.limit)