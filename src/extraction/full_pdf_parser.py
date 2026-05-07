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
# Metadata patterns
# ---------------------------------------------------------------------------

_STOP = r"(?=\s*[A-Z][a-zA-Z0-9\s\(\)\/\-\+\.]*?:|$|\n)"

# For text/name fields in the single-line multi-column layout, we stop when the
# next KNOWN section label appears. This prevents grabbing adjacent column text.
_KNOWN_LABELS = (
    r"Formation|Dist|Penetration|Hole|Pressure|Plug|Depth|Dia|Date|"
    r"Report|Days|Status|Tight|HPHT|Temperature|Wellbore|Spud|Water|"
    r"Elevation|Drilling|Operator|Rig|Survey|Lithology|Gas|Summary|Operations"
)
_STOP_LABEL = r"(?=\s*(?:" + _KNOWN_LABELS + r")\b)"

METADATA_PATTERNS = {
    "Wellbore ID":              r"(?i)Wellbore\s*:\s*(?:\n*\s*Wellbore\s*:\s*)?([A-Za-z0-9/\-\.]+)",
    "Report number":            r"(?i)Report\s+number\s*[:\s]+(\d+)",
    "Period":                   r"(?i)P+e+r+i+o+d+\s*:\s*([0-9\-:\s]+)",
    "Status":                   r"(?i)Status\s*:\s*([A-Za-z]+)",
    "Operator":                 r"(?i)Operator\s*:\s*(.*?)" + _STOP_LABEL,
    "Rig Name":                 r"(?i)Rig\s+Name\s*:\s*(.*?)" + _STOP_LABEL,
    "Drilling Contractor":      r"(?i)Drilling\s+contractor\s*:\s*(.*?)" + _STOP_LABEL,
    "Spud Date":                r"(?i)Spud\s+Date\s*:\s*([\d\-\/\s:]+)",
    "Water depth MSL (m)":      r"(?i)Water\s+depth\s+MSL[^\n:]*:\s*([\d\.]+)",
    "Elevation RKB-MSL (m)":    r"(?i)Elevation\s+RKB[^\n:]*:\s*([\d\.]+)",
    "Current Depth mMD":        r"(?i)Depth\s+m[Mm][Dd]\s*:\s*([\d\.]+)",
    "Current Depth mTVD":       r"(?i)Depth\s+mTVD\s*:\s*([\d\.]+)",
    "Kick Off Depth mMD":       r"(?i)Depth\s+at\s+[Kk]ick\s+[Oo]ff\s+mMD\s*:\s*([\d\.]+)",
    "Kick Off Depth mTVD":      r"(?i)Depth\s+at\s+[Kk]ick\s+[Oo]ff\s+mTVD\s*:\s*([\d\.]+)",
    "Last Casing Depth mMD":    r"(?i)Depth\s+[Aa]t\s+[Ll]ast\s+[Cc]asing\s+mMD\s*:\s*([\d\.]+)",
    "Plug Back Depth mMD":      r"(?i)Plug\s+Back\s+Depth\s+mMD\s*:\s*([\d\.]+)",
    "Tight well":               r"(?i)Tight\s+well\s*:\s*([YyNn])",
    "HPHT":                     r"(?i)HPHT\s*:\s*([YyNn])",
    "Hole Diameter (in)":       r"(?i)Hole\s+Dia[^\n:]*:\s*([\d\.]+)",
    "Formation Strength (g/cm3)": r"(?i)Formation\s+strength[^\n:]*:\s*([\d\.]+)",
    "Pressure Test Type":       r"(?i)Pressure\s+Test\s+Type\s*:\s*(.*?)" + _STOP,
}


def extract_metadata(full_text: str) -> dict:
    """Extract all metadata fields using the validated patterns."""
    metadata = {}
    for key, pattern in METADATA_PATTERNS.items():
        match = re.search(pattern, full_text)
        if match:
            val = match.group(1).strip()
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
    """
    FIX: pdfplumber collapses multi-column pages into a single long line,
    so line-by-line extraction fails. Instead we use regex directly on the
    single-line text with precise anchors.

    Pattern:
      "Summary of activities (24 Hours)"  → capture until next section header
      "Summary of planned activities ..."  → capture until next section header

    The STOP anchor is any known section keyword that marks the next block.
    We also strip any trailing numeric/time garbage that may follow.
    """
    # These words mark the end of a summary block
    _SUMMARY_STOP = (
        r"(?=Summary\s+of\s+planned|"
        r"Operations\b|"
        r"Drilling\s+Fluid\b|"
        r"Equipment\b|"
        r"Survey\s+Station\b|"
        r"Pore\s+Pressure\b|"
        r"Lithology\b|"
        r"Gas\s+Reading\b)"
    )

    act_m = re.search(
        r"Summary\s+of\s+activities[^\n]*?"  # header (not greedy)
        r"\s+((?:[A-Z0-9\"\.\,\&\;\:\-\/\(\)\s]+?))"  # body: only ALL-CAPS sentences
        + _SUMMARY_STOP,
        full_text,
        re.IGNORECASE | re.DOTALL,
    )

    plan_m = re.search(
        r"Summary\s+of\s+planned\s+activities[^\n]*?"
        r"\s+((?:[A-Z0-9\"\.\,\&\;\:\-\/\(\)\s]+?))"
        + _SUMMARY_STOP,
        full_text,
        re.IGNORECASE | re.DOTALL,
    )

    def _clean_summary(m):
        if not m:
            return None
        text = m.group(1).strip()
        # Remove leading "(24 Hours)" or similar bracket prefix
        text = re.sub(r"^\(\d+\s+Hours?\)\s*", "", text, flags=re.IGNORECASE)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Must be at least 10 chars of real content
        return text if len(text) >= 10 else None

    return {
        "Summary of activities":         _clean_summary(act_m),
        "Summary of planned activities": _clean_summary(plan_m),
    }


# ---------------------------------------------------------------------------
# Table detection & extraction
# ---------------------------------------------------------------------------

TABLE_TYPE_KEYWORDS = {
    # Order matters — checked top to bottom; first match wins.
    # Gas Reading must come BEFORE Equipment Failure because both have "class".
    "Gas Reading Table":       ["highest gas", "lowest gas", "depth to top", "c1 (ppm)"],
    "Operations Table":        ["start time", "end time", "main", "activity", "state", "remark"],
    "Drilling Fluid Table":    ["fluid", "density", "viscosity", "yield", "sample"],
    # Equipment Failure keywords are now specific compound phrases — not single
    # generic words like "class" that also appear in Gas Reading headers.
    "Equipment Failure Table": ["equipment system", "equipment class", "downtime", "equipment failure"],
}


def normalise_header_str(headers: list) -> str:
    raw = " ".join(str(h) for h in headers if h)
    raw = fix_double_letter_bug(raw)          # fix OCR doubles first
    raw = raw.lower().replace("\n", " ").replace("-", "")
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


# ---------------------------------------------------------------------------
# FIX 1: Drilling Fluid table is TRANSPOSED (rows = properties, cols = samples)
# Detect this format and pivot it properly into one dict per sample column.
# ---------------------------------------------------------------------------

def is_transposed_fluid_table(table: list) -> bool:
    """
    A transposed fluid table has its first column as property names
    (Sample Time, Fluid Type, Density, etc.) rather than column headers.
    Detected when the first-column values match known fluid property names.
    """
    if not table or len(table) < 3:
        return False
    first_col_vals = [str(row[0]).strip().lower() for row in table if row and row[0]]
    fluid_props = {"sample time", "fluid type", "fluid density", "density",
                   "sample depth", "sample point", "funnel visc", "plastic visc",
                   "yield point", "filtration"}
    hits = sum(1 for v in first_col_vals if any(fp in v for fp in fluid_props))
    return hits >= 3


def extract_transposed_fluid_table(table: list) -> list[dict]:
    """
    Convert a transposed table (prop_name | val1 | val2 | ...) into a list
    of dicts — one per sample column.
    """
    if not table:
        return []

    # How many sample columns are there? (all columns after the first)
    num_samples = max(len(row) for row in table) - 1
    if num_samples < 1:
        return []

    samples = [{} for _ in range(num_samples)]

    for row in table:
        if not row or not row[0]:
            continue
        prop_name = clean_text(row[0])
        if not prop_name:
            continue
        for col_idx in range(num_samples):
            val_idx = col_idx + 1
            val = clean_text(row[val_idx]) if val_idx < len(row) else None
            # Skip empty / placeholder values
            if val in (None, "", "-999.99", "-"):
                val = None
            samples[col_idx][prop_name] = val

    # Drop fully-empty sample dicts
    return [s for s in samples if any(v for v in s.values())]


# ---------------------------------------------------------------------------
# FIX 2: Smarter junk detection — do NOT kill Drilling Fluid tables.
# Only reject tables that are clearly metadata forms (key:value with colons),
# not the ones that contain known table types.
# ---------------------------------------------------------------------------

def is_junk_table(table_data: list) -> bool:
    """Return True only for tables that are definitely metadata forms, not data tables."""
    if not table_data or len(table_data) == 0:
        return True

    # If it looks like a transposed fluid table, never junk it
    if is_transposed_fluid_table(table_data):
        return False

    # Build flat text from first column only (to detect key:value form tables)
    first_col_text = ""
    for row in table_data[:10]:
        if row and row[0]:
            first_col_text += str(row[0]).lower() + " "

    # Metadata form tables: first column cells end with ":"
    # Count how many rows have a colon in the first cell (ignoring time HH:MM)
    colon_count = 0
    for row in table_data[:8]:
        if not row or not row[0]:
            continue
        cell = str(row[0])
        # Ignore time-like colons e.g. "00:00"
        stripped = re.sub(r"\d{2}:\d{2}", "", cell)
        if ":" in stripped:
            colon_count += 1

    if colon_count >= 3:
        return True

    # Tiny tables with no real data
    non_empty_rows = [r for r in table_data if r and any(c for c in r if c and str(c).strip())]
    if len(non_empty_rows) <= 1:
        return True

    return False


def extract_tables_from_page(page) -> dict:
    result: dict[str, list] = {}
    tables = page.extract_tables()
    if not tables:
        return result

    for table in tables:
        # Remove fully-empty rows
        cleaned = [row for row in table if row and any(cell for cell in row if cell)]
        if not cleaned:
            continue

        # FIX 2: use the improved junk filter
        if is_junk_table(cleaned):
            continue

        # FIX 1: handle transposed Drilling Fluid table specially
        if is_transposed_fluid_table(cleaned):
            rows_out = extract_transposed_fluid_table(cleaned)
            if rows_out:
                result.setdefault("Drilling Fluid Table", []).extend(rows_out)
            continue

        # Normal table: first row = headers
        headers = [clean_text(h) for h in cleaned[0]]
        table_type = identify_table_type(headers)

        rows_out = []
        for row in cleaned[1:]:
            if is_header_row(row, headers):
                continue
            row_dict = {}
            for i in range(len(headers)):
                header_name = headers[i] if (i < len(headers) and headers[i]) else f"Col_{i}"
                cell_value  = clean_text(row[i]) if (i < len(row) and row[i] is not None) else None
                row_dict[header_name] = cell_value
            rows_out.append(row_dict)

        if rows_out:
            result.setdefault(table_type, []).extend(rows_out)

    return result


# ---------------------------------------------------------------------------
# Per-PDF orchestration
# ---------------------------------------------------------------------------

KNOWN_TABLE_TYPES = ["Operations Table", "Drilling Fluid Table", "Equipment Failure Table", "Gas Reading Table"]


def extract_all_data_from_pdf(pdf_path: str) -> dict:
    full_text = ""
    tables_data: dict[str, list] = {k: [] for k in KNOWN_TABLE_TYPES}
    tables_data["Other Tables"] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw_text   = page.extract_text() or ""
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
# SQLite schema
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

        CREATE TABLE IF NOT EXISTS gas_readings (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id          TEXT REFERENCES documents(id),
            time                 TEXT,
            class                TEXT,
            depth_top_mmd        TEXT,
            depth_bottom_md      TEXT,
            depth_top_mtvd       TEXT,
            depth_bottom_tvd     TEXT,
            highest_gas_pct      TEXT,
            lowest_gas           TEXT,
            c1_ppm               TEXT,
            c2_ppm               TEXT,
            c3_ppm               TEXT,
            ic4_ppm              TEXT,
            ic5_ppm              TEXT,
            raw_json             TEXT
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
        doc_id, source_file,
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

    # -------------------------------------------------------------------------
    # FIX 3: Operations — column names now match actual PDF headers
    # Actual header after OCR fix: "Main -- Sub Activity" (not "Main" / "Sub")
    # -------------------------------------------------------------------------
    for row in tables.get("Operations Table", []):
        main_sub = (row.get("Main -- Sub Activity")
                    or row.get("Main - Sub Activity")
                    or row.get("Main Activity")
                    or row.get("Main"))
        # Split "drilling -- trip" into main / sub if combined
        main_act, sub_act = None, None
        if main_sub:
            parts = re.split(r"\s+--\s+|\s+-\s+", main_sub, maxsplit=1)
            main_act = parts[0].strip() if parts else main_sub
            sub_act  = parts[1].strip() if len(parts) > 1 else None

        cur.execute("""
            INSERT INTO operations
            (document_id,start_time,end_time,end_depth_mmd,
             main_activity,sub_activity,state,remark)
            VALUES (?,?,?,?,?,?,?,?)""", (
            doc_id,
            row.get("Start time") or row.get("Start Time"),
            row.get("End time")   or row.get("End Time"),
            row.get("End Depth mMD") or row.get("End Depth"),
            main_act,
            sub_act,
            row.get("State"),
            row.get("Remark") or row.get("Remarks"),
        ))

    # -------------------------------------------------------------------------
    # FIX 4: Drilling Fluid — column names match transposed-pivot output
    # Keys are the property names (first column of original table), e.g.
    # "Sample Time", "Fluid Type", "Fluid Density (g/cm3)", etc.
    # -------------------------------------------------------------------------
    for row in tables.get("Drilling Fluid Table", []):
        cur.execute("""
            INSERT INTO drilling_fluids
            (document_id,fluid_type,density,plastic_viscosity,
             yield_point,sample_time,sample_depth,raw_json)
            VALUES (?,?,?,?,?,?,?,?)""", (
            doc_id,
            row.get("Fluid Type")  or row.get("Fluid type") or row.get("Type"),
            row.get("Fluid Density (g/cm3)") or row.get("Density") or row.get("density"),
            row.get("Plastic visc. (mPa.s)") or row.get("Plastic Viscosity") or row.get("Plastic viscosity"),
            row.get("Yield point (Pa)") or row.get("Yield Point") or row.get("Yield point"),
            row.get("Sample Time") or row.get("Sample time"),
            row.get("Sample Depth mMD") or row.get("Sample Depth") or row.get("Sample depth"),
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

    for row in tables.get("Gas Reading Table", []):
        # Normalise -999.99 sentinel values to None
        def _gas_val(v):
            return None if v in (None, "", "-999.99") else v
        cur.execute("""
            INSERT INTO gas_readings
            (document_id,time,class,depth_top_mmd,depth_bottom_md,
             depth_top_mtvd,depth_bottom_tvd,highest_gas_pct,lowest_gas,
             c1_ppm,c2_ppm,c3_ppm,ic4_ppm,ic5_ppm,raw_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            doc_id,
            _gas_val(row.get("Time")),
            _gas_val(row.get("Class")),
            _gas_val(row.get("Depth to Top mMD") or row.get("Depth to Top")),
            _gas_val(row.get("Depth to Bottom MD") or row.get("Depth to Bottom")),
            _gas_val(row.get("Depth to Top mTVD")),
            _gas_val(row.get("Depth to Bottom TVD")),
            _gas_val(row.get("Highest Gas (%)") or row.get("Highest Gas")),
            _gas_val(row.get("Lowest Gas ()") or row.get("Lowest Gas")),
            _gas_val(row.get("C1 (ppm)")),
            _gas_val(row.get("C2 (ppm)")),
            _gas_val(row.get("C3 (ppm)")),
            _gas_val(row.get("IC4 (ppm)")),
            _gas_val(row.get("IC5 (ppm)")),
            json.dumps(row, ensure_ascii=False),
        ))

    conn.commit()


# ---------------------------------------------------------------------------
# Main loop
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
        doc_id = pdf_path.name
        try:
            record        = extract_all_data_from_pdf(str(pdf_path))
            record["_id"] = doc_id

            ops   = len(record["Tables"].get("Operations Table",       []))
            fluid = len(record["Tables"].get("Drilling Fluid Table",   []))
            equip = len(record["Tables"].get("Equipment Failure Table", []))
            log.info("  %-50s | ops=%3d fluid=%2d equip=%2d wellbore=%s",
                     doc_id, ops, fluid, equip,
                     record["Metadata"].get("Wellbore ID", "?"))

            _insert_document(conn, doc_id, record, source_file=doc_id)
            documents.append(record)

        except Exception:
            log.error("Failed: %s\n%s", doc_id, traceback.format_exc())
            errors.append({"file": doc_id, "error": traceback.format_exc()})

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
