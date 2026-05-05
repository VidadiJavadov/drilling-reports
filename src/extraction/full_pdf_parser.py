"""
Task 1 – PDF Parsing & Structured Data Extraction
eiLink R&D Centre
===========================================
All metadata patterns validated against the real PDF sample text.

Fixes vs. previous version:
  FIX 1 — source_file column added to documents table (Task 2 needs it)
  FIX 2 — Wellbore ID: PDF has two "Wellbore:" lines, second has value
           Old pattern matched first empty line → returned "Wellbore"
           New pattern explicitly skips the blank first line
  FIX 3 — Report number: old _word_pat("Report") matched "Report creation time: 2018"
           → returned "2018" instead of "4". Now uses "Report number" literally.
  FIX 4 — Rig Name / Drilling Contractor: PDF field is "MÆRSK INSPIRER" (non-ASCII Æ)
           and "Drilling contractor" (lowercase c). Capture to end-of-line instead.
  FIX 5 — Current Depth: PDF label is "Depth mMd:" not "Current Depth:"
           Last Casing: PDF label is "Depth At Last Casing mMD:" not "Last Casing:"
           Kick Off: PDF label is "Depth at Kick Off mMD:"
  FIX 6 — Spud Date / Period: capture full datetime including the time part (HH:MM)
           Period also has a duplicate blank first line — skip it like Wellbore ID
  FIX 7 — process_folder passes source_file to _insert_document (was missing)
"""

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
_NEXT_FIELD = r"(?=\s*(?:Rig Name|Drilling|Spud Date|Water|Elevation|Depth|Wellbore|Status|Period|Report|Tight|HPHT|Hole|Pressure|Formation|Dia |Date )|$|\n)"

METADATA_PATTERNS = {

    # YENİ: Həm tək, həm də alt-alta ikiqat yazılmış "Wellbore:" üçün işləyir
    "Wellbore ID": (
        r"(?i)Wellbore\s*:\s*(?:\n*\s*Wellbore\s*:\s*)?([A-Za-z0-9/][A-Za-z0-9/\-\.]+)"
    ),

    "Report number": (
        r"(?i)Report\s+number\s*[:\s]+(\d+)"
    ),

    "Period": (
        r"(?i)Period\s*:\s*[\n\s]*Period\s*:\s*([\d]{4}[\d\-\/ :]+?)(?:\n|$|\s{2,})"
    ),

    "Status": (
        r"(?i)Status\s*:\s*([A-Za-z ]+?)" + _NEXT_FIELD
    ),

    # DÜZƏLİŞ: Məlumatı udmasının (sürüşmənin) qarşısı alındı
    "Operator": (
        r"(?i)Operator\s*:\s*(.+?)" + _NEXT_FIELD
    ),

    # DÜZƏLİŞ: Məlumatı udmasının (sürüşmənin) qarşısı alındı
    "Rig Name": (
        r"(?i)Rig\s+Name\s*[:\s]+(.+?)" + _NEXT_FIELD
    ),

    # DÜZƏLİŞ: Məlumatı udmasının (sürüşmənin) qarşısı alındı
    "Drilling Contractor": (
        r"(?i)Drilling\s+contractor\s*[:\s]+(.+?)" + _NEXT_FIELD
    ),

    "Spud Date": (
        r"(?i)Spud\s+Date\s*[:\s]+([\d\-\/]+ [\d:]+)"
    ),

    "Water depth MSL (m)": (
        r"(?i)Water\s+depth\s+MSL[^\n:]*[:\s]+([\d\.]+)"
    ),

    "Elevation RKB-MSL (m)": (
        r"(?i)Elevation\s+RKB[^\n:]*[:\s]+([\d\.]+)"
    ),

    "Current Depth mMD": (
        r"(?i)Depth\s+m[Mm][Dd]\s*[:\s]+([\d\.]+)"
    ),

    "Current Depth mTVD": (
        r"(?i)Depth\s+mTVD\s*[:\s]+([\d\.]+)"
    ),

    "Kick Off Depth mMD": (
        r"(?i)Depth\s+at\s+[Kk]ick\s+[Oo]ff\s+mMD\s*[:\s]*([\d\.]+)"
    ),

    "Kick Off Depth mTVD": (
        r"(?i)Depth\s+at\s+[Kk]ick\s+[Oo]ff\s+mTVD\s*[:\s]*([\d\.]+)"
    ),

    "Last Casing Depth mMD": (
        r"(?i)Depth\s+[Aa]t\s+[Ll]ast\s+[Cc]asing\s+mMD\s*[:\s]*([\d\.]+)"
    ),

    "Plug Back Depth mMD": (
        r"(?i)Plug\s+Back\s+Depth\s+mMD\s*[:\s]*([\d\.]+)"
    ),

    "Tight well": (
        r"(?i)Tight\s+well\s*[:\s]+([YyNn])"
    ),

    "HPHT": (
        r"(?i)HPHT\s*[:\s]+([YyNn])"
    ),

    "Hole Diameter (in)": (
        r"(?i)Hole\s+Dia[^\n:]*[:\s]+([\d\.]+)"
    ),

    "Formation Strength (g/cm3)": (
        r"(?i)Formation\s+strength[^\n:]*[:\s]+([\d\.]+)"
    ),

    "Pressure Test Type": (
        r"(?i)Pressure\s+Test\s+Type\s*[:\s]+([A-Za-z /]+?)" + _NEXT_FIELD
    ),
}


def extract_metadata(full_text: str) -> dict:
    """Extract all metadata fields using the validated patterns."""
    metadata = {}
    for key, pattern in METADATA_PATTERNS.items():
        match = re.search(pattern, full_text)
        metadata[key] = match.group(1).strip() if match else None
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


def extract_tables_from_page(page) -> dict:
    result: dict[str, list] = {}
    tables = page.extract_tables()
    if not tables:
        return result

    for table in tables:
        cleaned = [row for row in table if any(cell for cell in row if cell)]
        if not cleaned:
            continue

        headers = [clean_text(h) for h in cleaned[0]]
        table_type = identify_table_type(headers)

        rows_out = []
        for row in cleaned[1:]:
            if is_header_row(row, headers):
                continue
            row_dict = {
                (headers[i] if headers[i] else f"Col_{i}"): (
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


# ---------------------------------------------------------------------------
# SQLite — FIX 1: source_file column added
# ---------------------------------------------------------------------------

def _create_sqlite_schema(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            id                   TEXT PRIMARY KEY,
            source_file          TEXT,
            wellbore_id          TEXT,
            report_number        TEXT,
            period               TEXT,
            status               TEXT,
            operator             TEXT,
            rig_name             TEXT,
            drilling_contractor  TEXT,
            spud_date            TEXT,
            water_depth_m        TEXT,
            elevation_rkb        TEXT,
            current_depth_mmd    TEXT,
            current_depth_mtvd   TEXT,
            kickoff_depth_mmd    TEXT,
            kickoff_depth_mtvd   TEXT,
            last_casing_mmd      TEXT,
            plug_back_mmd        TEXT,
            tight_well           TEXT,
            hpht                 TEXT,
            hole_diameter        TEXT,
            formation_strength   TEXT,
            pressure_test_type   TEXT,
            summary_activities   TEXT,
            summary_planned      TEXT
        );

        CREATE TABLE IF NOT EXISTS operations (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id   TEXT REFERENCES documents(id),
            start_time    TEXT,
            end_time      TEXT,
            end_depth_mmd TEXT,
            main_activity TEXT,
            sub_activity  TEXT,
            state         TEXT,
            remark        TEXT
        );

        CREATE TABLE IF NOT EXISTS drilling_fluids (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id       TEXT REFERENCES documents(id),
            fluid_type        TEXT,
            density           TEXT,
            plastic_viscosity TEXT,
            yield_point       TEXT,
            sample_time       TEXT,
            sample_depth      TEXT,
            raw_json          TEXT
        );

        CREATE TABLE IF NOT EXISTS equipment_failures (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id      TEXT REFERENCES documents(id),
            start_time       TEXT,
            depth            TEXT,
            equipment_system TEXT,
            equipment_class  TEXT,
            downtime_min     TEXT,
            remark           TEXT,
            raw_json         TEXT
        );

        CREATE TABLE IF NOT EXISTS other_tables (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT REFERENCES documents(id),
            raw_json    TEXT
        );
    """)
    conn.commit()


def _insert_document(conn: sqlite3.Connection, doc_id: str,
                     record: dict, source_file: str):
    meta   = record.get("Metadata",  {})
    summ   = record.get("Summaries", {})
    tables = record.get("Tables",    {})
    cur    = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO documents VALUES
        (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        doc_id,
        source_file,
        meta.get("Wellbore ID"),
        meta.get("Report number"),
        meta.get("Period"),
        meta.get("Status"),
        meta.get("Operator"),
        meta.get("Rig Name"),
        meta.get("Drilling Contractor"),
        meta.get("Spud Date"),
        meta.get("Water depth MSL (m)"),
        meta.get("Elevation RKB-MSL (m)"),
        meta.get("Current Depth mMD"),
        meta.get("Current Depth mTVD"),
        meta.get("Kick Off Depth mMD"),
        meta.get("Kick Off Depth mTVD"),
        meta.get("Last Casing Depth mMD"),
        meta.get("Plug Back Depth mMD"),
        meta.get("Tight well"),
        meta.get("HPHT"),
        meta.get("Hole Diameter (in)"),
        meta.get("Formation Strength (g/cm3)"),
        meta.get("Pressure Test Type"),
        summ.get("Summary of activities"),
        summ.get("Summary of planned activities"),
    ))

    for row in tables.get("Operations Table", []):
        cur.execute("""
            INSERT INTO operations
            (document_id,start_time,end_time,end_depth_mmd,
             main_activity,sub_activity,state,remark)
            VALUES (?,?,?,?,?,?,?,?)""", (
            doc_id,
            row.get("Start time") or row.get("Start Time"),
            row.get("End time")   or row.get("End Time"),
            row.get("End Depth mMD") or row.get("End Depth"),
            row.get("Main Activity") or row.get("Main"),
            row.get("Sub Activity")  or row.get("Sub"),
            row.get("State"),
            row.get("Remark") or row.get("Remarks"),
        ))

    for row in tables.get("Drilling Fluid Table", []):
        cur.execute("""
            INSERT INTO drilling_fluids
            (document_id,fluid_type,density,plastic_viscosity,
             yield_point,sample_time,sample_depth,raw_json)
            VALUES (?,?,?,?,?,?,?,?)""", (
            doc_id,
            row.get("Fluid type")  or row.get("Fluid Type") or row.get("Type"),
            row.get("Density")     or row.get("density"),
            row.get("Plastic viscosity") or row.get("Plastic Viscosity"),
            row.get("Yield point") or row.get("Yield Point"),
            row.get("Sample time") or row.get("Sample Time"),
            row.get("Sample depth") or row.get("Sample Depth"),
            json.dumps(row, ensure_ascii=False),
        ))

    for row in tables.get("Equipment Failure Table", []):
        cur.execute("""
            INSERT INTO equipment_failures
            (document_id,start_time,depth,equipment_system,
             equipment_class,downtime_min,remark,raw_json)
            VALUES (?,?,?,?,?,?,?,?)""", (
            doc_id,
            row.get("Start time") or row.get("Start Time"),
            row.get("Depth")      or row.get("Depths"),
            row.get("Equipment system") or row.get("Equipment System"),
            row.get("Equipment class")  or row.get("Equipment Class"),
            row.get("Downtime")   or row.get("Downtime (min)"),
            row.get("Remark")     or row.get("Remarks"),
            json.dumps(row, ensure_ascii=False),
        ))

    for row in tables.get("Other Tables", []):
        cur.execute(
            "INSERT INTO other_tables (document_id,raw_json) VALUES (?,?)",
            (doc_id, json.dumps(row, ensure_ascii=False)),
        )

    conn.commit()


# ---------------------------------------------------------------------------
# Main loop — FIX 7: source_file passed to _insert_document
# ---------------------------------------------------------------------------

def process_folder(
    folder_path: str,
    output_json: str,
    output_db:   str,
    limit: int | None = None,
):
    folder    = Path(folder_path)
    pdf_files = sorted(folder.glob("*.pdf"))
    if limit:
        pdf_files = pdf_files[:limit]

    log.info("Found %d PDF files to process.", len(pdf_files))
    os.makedirs(Path(output_json).parent, exist_ok=True)
    os.makedirs(Path(output_db).parent,   exist_ok=True)

    conn = sqlite3.connect(output_db)
    _create_sqlite_schema(conn)

    documents, errors = [], []

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        doc_id = pdf_path.name          # e.g. "15_9_F_10_2009_04_08.pdf"
        try:
            record        = extract_all_data_from_pdf(str(pdf_path))
            record["_id"] = doc_id

            ops   = len(record["Tables"].get("Operations Table",       []))
            fluid = len(record["Tables"].get("Drilling Fluid Table",   []))
            equip = len(record["Tables"].get("Equipment Failure Table", []))
            log.info("  %-50s | ops=%3d fluid=%2d equip=%2d wellbore=%s",
                     doc_id, ops, fluid, equip,
                     record["Metadata"].get("Wellbore ID", "?"))

            # FIX 7: pass source_file explicitly
            _insert_document(conn, doc_id, record, source_file=doc_id)
            documents.append(record)

        except Exception as exc:
            log.error("Failed: %s\n%s", doc_id, traceback.format_exc())
            errors.append({"file": doc_id, "error": str(exc)})

    conn.close()

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)

    log.info("Done! %d processed, %d errors.", len(documents), len(errors))
    log.info("  JSON   -> %s", output_json)
    log.info("  SQLite -> %s", output_db)

    if errors:
        err_path = Path(output_json).with_name("extraction_errors.json")
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        log.warning("  Errors -> %s", err_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 - PDF extraction pipeline")
    parser.add_argument("--input", "-i",
        default="../../data/raw/pdf_reports/PDF_version_1000")
    parser.add_argument("--json", "-j",
        default="../../data/processed/document_database.json")
    parser.add_argument("--db", "-d",
        default="../../data/processed/document_database.sqlite")
    parser.add_argument("--limit", "-l", type=int, default=None,
        help="Process only first N PDFs (for testing)")
    args = parser.parse_args()

    process_folder(args.input, args.json, args.db, args.limit)
