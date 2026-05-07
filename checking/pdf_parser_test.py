import os
import re
import json
import sqlite3
import logging
import argparse
import traceback
from pathlib import Path

import pdfplumber
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def fix_double_letter_bug(text: str) -> str:
    """Fix 'SSttaarrtt' OCR artefact where every character is doubled."""
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


def clean_text(text) -> str:
    """Collapse newlines to spaces, fix doubled letters, collapse whitespace."""
    if not isinstance(text, str):
        return text
    text = text.replace("\n", " ")
    text = fix_double_letter_bug(text)
    return re.sub(r"\s+", " ", text).strip()


def make_pattern(phrase: str) -> str:
    """Allow arbitrary whitespace between every character (handles OCR spacing)."""
    return r"\s*".join(list(phrase))


# ---------------------------------------------------------------------------
# Metadata patterns — validated against real PDF sample
# ---------------------------------------------------------------------------

def _word_pat(phrase):
    """Label pattern: phrase + optional junk + colon/space separator."""
    return r"(?i)" + make_pattern(phrase) + r"[^\n:]*[:\s]+"

_TERM = r"(?=\s{1,}|\n|$)"   # stop at any whitespace, newline, or end

# Full pattern map — every pattern tested against real PDF output
# ---------------------------------------------------------------------------
# Metadata patterns — Yenilənmiş "Tənbəl" (Lazy) və "Divarlı" Regex-lər
# ---------------------------------------------------------------------------

# Bu Regex "Divar" rolunu oynayır. Gördüyü yerdə axtarışı dayandırır.
# Əgər yeni sətir (\n), və ya başqa bir parametr başlığı (Rig Name, Depth və s.) gələrsə stop edir.
# --- 1. YENİ UNIVERSAL DİVAR ---
# Bu divar istənilən Böyük hərflə başlayan və sonu ":" ilə bitən başlığı görəndə axtarışı dayandırır.
_STOP = r"(?=\s*[A-Z][a-zA-Z0-9\s\(\)\/\-\+\.]*?:|$|\n)"

METADATA_PATTERNS = {
    "Wellbore ID": r"(?i)Wellbore\s*:\s*(?:\n*\s*Wellbore\s*:\s*)?([A-Za-z0-9/\-\.]+)",
    "Report number": r"(?i)Report\s+number\s*[:\s]+(\d+)",
    "Period": r"(?i)P+e+r+i+o+d+\s*:\s*([0-9\-:\s]+)",
    "Status": r"(?i)Status\s*:\s*([A-Za-z]+)",
    
    # ".*?" mətni tənbəlcəsinə götürür, "_STOP" isə növbəti başlığı görən kimi onu saxlayır
    "Operator": r"(?i)Operator\s*:\s*(.*?)" + _STOP,
    "Rig Name": r"(?i)Rig\s+Name\s*:\s*(.*?)" + _STOP,
    "Drilling Contractor": r"(?i)Drilling\s+contractor\s*:\s*(.*?)" + _STOP,
    
    "Spud Date": r"(?i)Spud\s+Date\s*:\s*([\d\-\/\s:]+)",
    "Water depth MSL (m)": r"(?i)Water\s+depth\s+MSL[^\n:]*:\s*([\d\.]+)",
    "Elevation RKB-MSL (m)": r"(?i)Elevation\s+RKB[^\n:]*:\s*([\d\.]+)",
    "Current Depth mMD": r"(?i)Depth\s+m[Mm][Dd]\s*:\s*([\d\.]+)",
    "Current Depth mTVD": r"(?i)Depth\s+mTVD\s*:\s*([\d\.]+)",
    "Kick Off Depth mMD": r"(?i)Depth\s+at\s+[Kk]ick\s+[Oo]ff\s+mMD\s*:\s*([\d\.]+)",
    "Kick Off Depth mTVD": r"(?i)Depth\s+at\s+[Kk]ick\s+[Oo]ff\s+mTVD\s*:\s*([\d\.]+)",
    "Last Casing Depth mMD": r"(?i)Depth\s+[Aa]t\s+[Ll]ast\s+[Cc]asing\s+mMD\s*:\s*([\d\.]+)",
    "Plug Back Depth mMD": r"(?i)Plug\s+Back\s+Depth\s+mMD\s*:\s*([\d\.]+)",
    "Tight well": r"(?i)Tight\s+well\s*:\s*([YyNn])",
    "HPHT": r"(?i)HPHT\s*:\s*([YyNn])",
    "Hole Diameter (in)": r"(?i)Hole\s+Dia[^\n:]*:\s*([\d\.]+)",
    "Formation Strength (g/cm3)": r"(?i)Formation\s+strength[^\n:]*:\s*([\d\.]+)",
    "Pressure Test Type": r"(?i)Pressure\s+Test\s+Type\s*:\s*(.*?)" + _STOP,
}

# --- 2. ZİBİL DATALARI TƏMİZLƏYƏN FUNKSİYA ---
def extract_metadata(full_text: str) -> dict:
    """Extract all metadata fields using the validated patterns and clean garbage data."""
    metadata = {}
    for key, pattern in METADATA_PATTERNS.items():
        match = re.search(pattern, full_text)
        if match:
            val = match.group(1).strip()
            # Boşluqları, lazımsız -999.99 rəqəmlərini və tək qalan simvolları null edirik
            if val in ["-999.99", "", "None"] or len(val) < 1:
                metadata[key] = None
            else:
                metadata[key] = val
        else:
            metadata[key] = None
    return metadata


# ---------------------------------------------------------------------------
# Summary sections
# ---------------------------------------------------------------------------

def extract_summaries(full_text: str) -> dict:
    act_match = re.search(
        r"(?i)Summary\s+of\s+activities[^\n]*\n(.*?)"
        r"(?=\n\s*Summary\s+of\s+planned|$)",
        full_text, re.DOTALL,
    )
    plan_match = re.search(
        r"(?i)Summary\s+of\s+planned\s+activities[^\n]*\n(.*?)"
        r"(?=\n\s*(?:Operations|Survey|Casing|$))",
        full_text, re.DOTALL,
    )

    return {
        "Summary of activities":         clean_text(act_match.group(1))  if act_match  else None,
        "Summary of planned activities": clean_text(plan_match.group(1)) if plan_match else None,
    }


# ---------------------------------------------------------------------------
# Table detection & extraction
# ---------------------------------------------------------------------------

TABLE_TYPE_KEYWORDS = {
    "Operations Table":        ["activity", "remark", "mainsub", "state", "starttime", "endtime"],
    "Drilling Fluid Table":    ["fluid", "density", "viscosity", "yieldpoint", "plasticviscosity"],
    "Equipment Failure Table": ["equipment", "failure", "downtime", "system", "class"],
}


def normalise_header_str(headers: list) -> str:
    raw = "".join(str(h) for h in headers if h)
    raw = raw.lower().replace("\n", "").replace(" ", "").replace("-", "")
    raw = re.sub(r"(.)\1+", r"\1", raw)   # collapse doubled chars (OCR fix)
    return raw


def identify_table_type(headers: list) -> str:
    norm = normalise_header_str(headers)
    for table_type, keywords in TABLE_TYPE_KEYWORDS.items():
        if any(kw in norm for kw in keywords):
            return table_type
    return "Other Table"


def is_header_row(row: list, headers: list) -> bool:
    cleaned_row = [clean_text(c) for c in row if c]
    cleaned_hdr = [clean_text(h) for h in headers if h]
    if not cleaned_row or not cleaned_hdr:
        return False
    matches = sum(1 for r in cleaned_row if r in cleaned_hdr)
    return matches / max(len(cleaned_hdr), 1) > 0.5

def is_junk_table(table_data):
    """Cədvəli həm başlıq, həm də daxili strukturuna görə yoxlayır."""
    if not table_data or len(table_data) == 0:
        return True
    
    # 1. Başlıqları mətn kimi birləşdirib yoxlayırıq
    first_row = [str(c).lower() for c in table_data[0] if c]
    header_text = "".join(first_row).replace(" ", "").replace("\n", "")
    
    junk_keywords = [
        "status", "reportnumber", "operator", "rigname", "spuddate",
        "distdrilled", "penetration", "holedia", "pressuretest",
        "formationstrength", "kickoff", "plugback", "summaryreport",
        "depthmmd", "depthmtvd", "lastcasing"
    ]
    
    # Əgər başlıqda bu sözlərdən biri varsa - Zibildir
    if any(kw in header_text for kw in junk_keywords):
        return True

    # 2. Struktur yoxlanışı: Əgər sətirlərdə ":" varsa, bu formadır (cədvəl deyil)
    # Sənin JSON-da gördüyümüz "Dist Drilled (m):" kimi halları bu tutur
    for row in table_data[:5]: # İlk 5 sətri yoxlamaq kifayətdir
        for cell in row:
            if cell and isinstance(cell, str) and ":" in cell:
                # İstisna: Əgər bu bir vaxtdırsa (məs. 12:00), onu zibil sayma
                if not re.search(r"\d{2}:\d{2}", cell):
                    return True
                    
    return False


def extract_tables_from_page(page) -> dict:
    result: dict[str, list] = {}
    tables = page.extract_tables()
    if not tables:
        return result

    for table in tables:
        # Boş sətirləri təmizləyirik
        cleaned = [row for row in table if any(cell for cell in row if cell)]
        if not cleaned:
            continue

        # Başlıqları götürürük
        headers = [clean_text(h) for h in cleaned[0]]
        
        # --- YENİ FİLTR BURADADIR ---
        # Əgər bu cədvəl əslində metadatadırsa, onu emal etmirik
        if is_junk_table(headers):
            continue 
        # ----------------------------

        table_type = identify_table_type(headers)

        rows_out = []
        for row in cleaned[1:]:
            if is_header_row(row, headers):
                continue
            row_dict = {
                (headers[i] if i < len(headers) and headers[i] else f"Col_{i}"): (
                    clean_text(row[i]) if i < len(row) else None
                )
                for i in range(len(headers))
            }
            rows_out.append(row_dict)

        if rows_out:
            result.setdefault(table_type, []).extend(rows_out)

    return result


# ---------------------------------------------------------------------------
# Per-PDF orchestration
# ---------------------------------------------------------------------------

KNOWN_TABLE_TYPES = ["Operations Table", "Drilling Fluid Table", "Equipment Failure Table"]


def extract_all_data_from_pdf(pdf_path: str) -> dict:
    full_text = ""
    tables_data: dict[str, list] = {k: [] for k in KNOWN_TABLE_TYPES}
    tables_data["Other Tables"] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_text  = page.extract_text() or ""
            fixed_text = fix_double_letter_bug(raw_text)
            full_text += fixed_text + "\n"

            for ttype, rows in extract_tables_from_page(page).items():
                target = ttype if ttype in tables_data else "Other Tables"
                tables_data[target].extend(rows)

    return {
        "Metadata":  extract_metadata(full_text),
        "Summaries": extract_summaries(full_text),
        "Tables":    tables_data,
    }

# --- BU HİSSƏNİ KODUNUN ƏN SONUNA YAPIŞDIR ---

if __name__ == "__main__":
    # 1. PDF-lərin olduğu qovluğun yolunu bura yaz
    PDF_FOLDER = "D:/drilling-report-nlp/data/raw/pdf_reports/PDF_version_1000" 
    
    # 2. Qovluqdakı ilk 5 PDF faylını seçək
    pdf_path = Path(PDF_FOLDER)
    files = sorted(list(pdf_path.glob("*.pdf")))[:5] # Rəqəmi dəyişə bilərsən (məs. :10)

    if not files:
        print(f"❌ Xəta: '{PDF_FOLDER}' qovluğunda PDF faylı tapılmadı!")
    else:
        print(f"🔍 {len(files)} fayl üzərində test başlayır...\n")
        
        all_results = []

        for f in files:
            print(f"📄 Emal edilir: {f.name}")
            try:
                # Sənin yazdığın əsas funksiyanı çağırırıq
                result = extract_all_data_from_pdf(str(f))
                result["filename"] = f.name
                all_results.append(result)
            except Exception as e:
                print(f"⚠️ {f.name} emal edilərkən xəta baş verdi: {e}")

        # 3. Nəticələri terminalda səliqəli JSON formatında göstərək
        print("\n" + "="*50)
        print("📊 TEST NƏTİCƏLƏRİ")
        print("="*50)
        print(json.dumps(all_results, indent=2, ensure_ascii=False))
        
        # İstəyirsənsə nəticəni müvəqqəti fayla da yaza bilərsən
        with open("test_output.json", "w", encoding="utf-8") as out_f:
            json.dump(all_results, out_f, indent=2, ensure_ascii=False)
        print(f"\n✅ Nəticələr 'test_output.json' faylına da qeyd edildi.")