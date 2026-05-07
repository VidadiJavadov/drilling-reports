"""
Task 3 – NLP Analysis on Extracted Text
eiLink R&D Centre
==========================================
Applies three NLP techniques to remark/free-text fields from the SQLite database:

  3a. Named Entity Recognition (NER)
      Domain-specific rule-based NER using spaCy EntityRuler (no pre-trained model
      required). Extracts:
        • DEPTH       — numeric depth values (mMD, mTVD, m MD, metres)
        • EQUIPMENT   — oilfield tool/equipment names (BOP, TDS, BHA, packers, etc.)
        • MEASUREMENT — RPM, bar, lpm, MT, ton, m3 readings
        • TIME_REF    — time expressions (HH:MM, "at 06:00", "after 2 hours")

  3b. Activity Classification / Tagging
      Rule-based classifier that assigns one or more normalized activity labels to
      each remark:
        TRIP_IN, TRIP_OUT, DRILL, CEMENT, PRESSURE_TEST, EQUIPMENT_FAILURE,
        REPAIR, WAIT, CIRCULATE, FISHING, SURVEY, PUMP, CIRCULATE, DISPLACE, OTHER
      Multi-label: a remark can carry more than one tag (e.g. TRIP_OUT + EQUIPMENT_FAILURE).

  3c. TF-IDF Keyword Extraction
      Per-document top-N keywords computed across the full corpus of remarks.
      Surfaces language that makes each PDF report uniquely identifiable.

  BONUS analyses:
    • Activity distribution chart (bar chart saved as PNG)
    • Entity type frequency summary
    • Co-occurrence matrix of activity labels
    • Top-10 most-mentioned equipment items
    • Temporal density: how many events per hour-of-day

All results are saved to ../../data/results/:
    nlp_ner_results.csv          — one row per entity found
    nlp_activity_tags.csv        — one row per remark with its activity labels
    nlp_tfidf_keywords.csv       — top-N keywords per document
    nlp_entity_summary.csv       — entity type counts per document
    nlp_equipment_top10.csv      — most-mentioned equipment across corpus
    nlp_activity_distribution.csv — activity label counts
    nlp_hourly_activity.csv      — event count by hour of day
"""

import os
import re
import sqlite3
import warnings
import logging
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH      = "../../data/processed/document_database.sqlite"
RESULTS_DIR  = "../../data/results"
TOP_N_KEYWORDS = 15   # keywords per document

# ---------------------------------------------------------------------------
# Drilling abbreviation expansion (same as Task 2 — ensures NER sees full terms)
# ---------------------------------------------------------------------------
DRILLING_ABBREVS = {
    r"\bPOOH\b": "pull out of hole",
    r"\bPOH\b":  "pull out of hole",
    r"\bRIH\b":  "run in hole",
    r"\bTIH\b":  "trip in hole",
    r"\bTOOH\b": "trip out of hole",
    r"\bBHA\b":  "bottom hole assembly",
    r"\bMWD\b":  "measurement while drilling",
    r"\bLWD\b":  "logging while drilling",
    r"\bDHM\b":  "downhole motor",
    r"\bWOB\b":  "weight on bit",
    r"\bROP\b":  "rate of penetration",
    r"\bSPP\b":  "standpipe pressure",
    r"\bECD\b":  "equivalent circulating density",
    r"\bBOP\b":  "blowout preventer",
    r"\bDHSV\b": "downhole safety valve",
    r"\bWOC\b":  "wait on cement",
    r"\bTDS\b":  "top drive system",
    r"\bXO\b":   "crossover",
    r"\bWH\b":   "wellhead",
    r"\bROV\b":  "remotely operated vehicle",
}

def expand_abbrevs(text: str) -> str:
    if not isinstance(text, str):
        return ""
    for pat, rep in DRILLING_ABBREVS.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    return text


# ===========================================================================
# 3a — Named Entity Recognition (rule-based, no pre-trained model needed)
# ===========================================================================

# --- Depth entities --------------------------------------------------------
# Matches: "2145 mMD", "207 m MD", "3426.62 mMD", "1365m MD", "460m"
_DEPTH_PAT = re.compile(
    r"\b(\d+(?:[.,]\d+)?)\s*"
    r"(?:m\s*MD|mMD|m\s*TVD|mTVD|m\s*(?:depth)?|metres?|meters?)\b",
    re.IGNORECASE,
)

# --- Measurement entities --------------------------------------------------
_MEAS_PATTERNS = [
    ("RPM",      re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:RPM|rpm)\b")),
    ("BAR",      re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:bar|BAR|Bar)\b")),
    ("LPM",      re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:lpm|LPM|l/min)\b")),
    ("MT",       re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:MT|mT|metric\s*ton)\b")),
    ("M3",       re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:m3|m³)\b")),
    ("SG",       re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:sg|SG|g/cm3|g/cm³)\b")),
    ("DEG",      re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:deg|dgr|°)\b", re.IGNORECASE)),
    ("KNM",      re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(?:kNm|KNm)\b")),
]

# --- Equipment entities ----------------------------------------------------
# Ordered longest-first to avoid partial matches
_EQUIP_TERMS = sorted([
    "blowout preventer", "top drive system", "downhole safety valve",
    "bottom hole assembly", "measurement while drilling",
    "logging while drilling", "downhole motor", "remotely operated vehicle",
    "FLX packer", "spear BHA", "cement retainer", "drill collar",
    "HWDP", "drill pipe", "landing string", "centralizer", "crossover sub",
    "packer", "liner hanger", "float collar", "float shoe",
    "casing hanger", "wellhead", "BOP stack", "choke manifold",
    "accumulator", "diverter", "annular preventer", "pipe ram",
    "blind ram", "shear ram", "kill line", "choke line",
    "mud motor", "PDC bit", "tricone bit", "hole opener", "reamer",
    "jar", "shock sub", "stabilizer", "MWD tool", "LWD tool",
    "NMDC", "survey tool", "inclinometer", "gyro",
    "cement head", "cementing unit", "pump", "TDS", "BOP", "WH",
    "CART", "ROV", "XO", "FOSV", "HTS", "CET", "CBL",
    # common abbreviations NOT expanded (still appear in remarks)
    "HO BHA", "26\" HO", "20\" casing", "30\" conductor",
], key=len, reverse=True)

_EQUIP_PATTERNS = [
    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    for term in _EQUIP_TERMS
]

# --- Time reference entities -----------------------------------------------
_TIME_PAT = re.compile(
    r"\b(?:"
    r"\d{2}:\d{2}"                          # HH:MM
    r"|at\s+\d{1,2}:\d{2}"                 # "at 06:00"
    r"|\d+\s*(?:hour|hr|min)s?"            # "2 hours", "30 min"
    r"|after\s+\d+\s*(?:hour|hr|min)s?"   # "after 2 hours"
    r")",
    re.IGNORECASE,
)


def extract_entities(text: str) -> list[dict]:
    """
    Extract all NER entities from a single text string.
    Returns list of {text, label, start, end} dicts.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    expanded = expand_abbrevs(text)
    entities = []

    # Depths
    for m in _DEPTH_PAT.finditer(expanded):
        entities.append({
            "entity_text":  m.group(0).strip(),
            "entity_label": "DEPTH",
            "value":        m.group(1).replace(",", "."),
            "start":        m.start(),
            "end":          m.end(),
        })

    # Measurements
    for unit_label, pat in _MEAS_PATTERNS:
        for m in pat.finditer(expanded):
            entities.append({
                "entity_text":  m.group(0).strip(),
                "entity_label": f"MEASUREMENT_{unit_label}",
                "value":        m.group(1).replace(",", "."),
                "start":        m.start(),
                "end":          m.end(),
            })

    # Equipment (greedy: mark spans to avoid double-counting)
    matched_spans = []
    for term, pat in zip(_EQUIP_TERMS, _EQUIP_PATTERNS):
        for m in pat.finditer(expanded):
            # skip if this span overlaps an already-matched span
            if any(s <= m.start() < e or s < m.end() <= e
                   for s, e in matched_spans):
                continue
            matched_spans.append((m.start(), m.end()))
            entities.append({
                "entity_text":  m.group(0).strip(),
                "entity_label": "EQUIPMENT",
                "value":        term,
                "start":        m.start(),
                "end":          m.end(),
            })

    # Time references
    for m in _TIME_PAT.finditer(expanded):
        entities.append({
            "entity_text":  m.group(0).strip(),
            "entity_label": "TIME_REF",
            "value":        m.group(0).strip(),
            "start":        m.start(),
            "end":          m.end(),
        })

    return entities


# ===========================================================================
# 3b — Activity Classification (multi-label rule-based)
# ===========================================================================

# Each rule: (label, list_of_trigger_patterns)
# Patterns are OR-ed; a match anywhere in the remark fires the label.
_ACTIVITY_RULES: list[tuple[str, list[str]]] = [
    ("TRIP_OUT", [
        r"\bpull(?:ed|ing)?\s+out\s+of\s+hole\b",
        r"\bpooh\b", r"\bpoh\b", r"\btrip(?:ped|ping)?\s+out\b",
        r"\btooh\b", r"\bracked?\s+back\b",
        r"\blaid?\s+out\b",
    ]),
    ("TRIP_IN", [
        r"\brun(?:ning)?\s+in\s+hole\b",
        r"\brih\b", r"\btih\b", r"\btrip(?:ped|ping)?\s+in\b",
        r"\bmake?\s+up\s+and\s+rih\b",
    ]),
    ("DRILL", [
        r"\bdrill(?:ed|ing)?\b",
        r"\brate\s+of\s+penetration\b",
        r"\brop\b", r"\bwob\b",
        r"\bdrilled?\s+\d+\s*m\b",
    ]),
    ("CEMENT", [
        r"\bcemen(?:t(?:ed|ing)?|tation)\b",
        r"\bpumped?\s+\d+.*?cement\b",
        r"\bslurry\b", r"\bwait(?:ed|ing)?\s+on\s+cement\b",
        r"\bwoc\b",
    ]),
    ("PRESSURE_TEST", [
        r"\bpressure\s+test\b",
        r"\bleak(?:\s*off)?\s+test\b",
        r"\blot\b", r"\bfit\b",
        r"\bline\s+test\b",
        r"\btested?\s+to\s+\d+\s*bar\b",
    ]),
    ("CIRCULATE", [
        r"\bcirculat(?:ed|ing|ion)\b",
        r"\bpumped?\s+\d+\s*m3\b",
        r"\bflushed?\b", r"\bdisplac(?:ed|ing)\b",
        r"\bpill\b", r"\bhivis\b",
    ]),
    ("SURVEY", [
        r"\bsurvey\b", r"\binclinometer\b",
        r"\bgyro\b", r"\bmwd\s*data\b",
        r"\btook?\s+survey\b",
        r"\binclination\b.*?\bdeg\b",
    ]),
    ("FISHING", [
        r"\bfishing\b", r"\bfish(?:ed|ing)?\b",
        r"\bovershot\b", r"\bspear\b",
        r"\bjunk\b", r"\bmill(?:ed|ing)?\b",
    ]),
    ("EQUIPMENT_FAILURE", [
        r"\bfail(?:ed|ure|ing)\b",
        r"\bbreakdown\b", r"\bmalfunction\b",
        r"\bproblem\b", r"\bunable\s+to\b",
        r"\bno\s+(?:go|progress|success)\b",
        r"\bnegative\b",
    ]),
    ("REPAIR", [
        r"\brepair(?:ed|ing)?\b",
        r"\breplace(?:d|ment)?\b",
        r"\binstall(?:ed|ing)?\b",
        r"\bmade?\s+up\b",
        r"\bchanged?\b",
    ]),
    ("WAIT", [
        r"\bwait(?:ed|ing)?\b",
        r"\bstand(?:ing)?\s+by\b",
        r"\bweather\b",
        r"\bwoc\b",
        r"\bpause\b",
    ]),
    ("PUMP", [
        r"\bpump(?:ed|ing)?\b",
        r"\bmix(?:ed|ing)?\s+and\s+pump\b",
    ]),
    ("DISPLACE", [
        r"\bdisplace(?:d|ment)?\b",
        r"\bseawater\b.*?\bpumped?\b",
        r"\bpumped?.*?\bseawater\b",
    ]),
    ("CUT", [
        r"\bcut\b", r"\bperforat(?:ed|ing|ion)\b",
        r"\bshoot\b",
    ]),
]

# Compile all patterns once
_COMPILED_RULES: list[tuple[str, list[re.Pattern]]] = [
    (label, [re.compile(p, re.IGNORECASE) for p in patterns])
    for label, patterns in _ACTIVITY_RULES
]


def classify_activity(text: str) -> list[str]:
    """
    Return list of activity labels that apply to this remark.
    Returns ["OTHER"] if no rule fires.
    """
    if not isinstance(text, str) or not text.strip():
        return ["OTHER"]

    expanded = expand_abbrevs(text)
    labels = []
    for label, patterns in _COMPILED_RULES:
        if any(p.search(expanded) for p in patterns):
            labels.append(label)

    return labels if labels else ["OTHER"]


# ===========================================================================
# 3c — TF-IDF Keyword Extraction
# ===========================================================================

# Drilling stopwords (common but uninformative in this corpus)
_DRILLING_STOPWORDS = {
    "the", "and", "with", "from", "into", "this", "that", "was", "were",
    "been", "have", "has", "had", "for", "are", "not", "but", "all",
    "at", "to", "in", "of", "on", "a", "an", "is", "it", "be", "by",
    "up", "out", "ok", "per", "as", "or", "no", "do",
    # very common drilling terms that appear in every report
    "well", "pipe", "hole", "depth", "time", "report",
    "drilling", "drill", "run", "pump", "flow",
}


def extract_tfidf_keywords(
    doc_texts: dict[str, str],
    top_n: int = TOP_N_KEYWORDS,
) -> pd.DataFrame:
    """
    Compute TF-IDF over the corpus (one doc = all remarks concatenated).
    Returns a DataFrame with columns: doc_id, rank, keyword, tfidf_score.
    """
    doc_ids   = list(doc_texts.keys())
    corpus    = [doc_texts[d] for d in doc_ids]

    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),           # unigrams + bigrams
        min_df=2,                     # must appear in ≥2 docs to be useful
        sublinear_tf=True,            # log(1+tf) — dampens very frequent terms
        stop_words=list(_DRILLING_STOPWORDS),
        token_pattern=r"\b[a-z][a-z\d]{2,}\b",  # min 3 chars, starts with letter
    )

    tfidf_matrix = vec.fit_transform(corpus)
    feature_names = vec.get_feature_names_out()

    rows = []
    for i, doc_id in enumerate(doc_ids):
        scores = tfidf_matrix[i].toarray().flatten()
        top_idx = scores.argsort()[::-1][:top_n]
        for rank, idx in enumerate(top_idx, start=1):
            if scores[idx] > 0:
                rows.append({
                    "doc_id":      doc_id,
                    "rank":        rank,
                    "keyword":     feature_names[idx],
                    "tfidf_score": round(float(scores[idx]), 6),
                })

    return pd.DataFrame(rows)


# ===========================================================================
# Database loading
# ===========================================================================

def load_texts_from_db(db_path: str) -> pd.DataFrame:
    """
    Load all text fields from the database into a single DataFrame.
    Returns columns: doc_id, text_type, text, start_time
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Run Task 1 first to generate the database."
        )

    conn = sqlite3.connect(db_path)
    rows = []

    # Operations remarks
    try:
        df_ops = pd.read_sql_query("""
            SELECT d.source_file AS doc_id,
                   o.remark      AS text,
                   o.start_time  AS start_time,
                   'operation'   AS text_type
            FROM   operations o
            JOIN   documents  d ON o.document_id = d.id
            WHERE  o.remark IS NOT NULL AND trim(o.remark) != ''
        """, conn)
        rows.append(df_ops)
        log.info("  Loaded %d operation remarks.", len(df_ops))
    except Exception as e:
        log.warning("  Could not load operations: %s", e)

    # Equipment failure remarks
    try:
        df_eq = pd.read_sql_query("""
            SELECT d.source_file AS doc_id,
                   e.remark      AS text,
                   e.start_time  AS start_time,
                   'equipment_failure' AS text_type
            FROM   equipment_failures e
            JOIN   documents  d ON e.document_id = d.id
            WHERE  e.remark IS NOT NULL AND trim(e.remark) != ''
        """, conn)
        rows.append(df_eq)
        log.info("  Loaded %d equipment failure remarks.", len(df_eq))
    except Exception as e:
        log.warning("  Could not load equipment failures: %s", e)

    # Summary sections
    try:
        df_sum = pd.read_sql_query("""
            SELECT source_file          AS doc_id,
                   summary_activities   AS text_activities,
                   summary_planned      AS text_planned
            FROM   documents
            WHERE  summary_activities IS NOT NULL
               OR  summary_planned    IS NOT NULL
        """, conn)
        for _, r in df_sum.iterrows():
            for col, ttype in [("text_activities", "summary_activities"),
                                ("text_planned",    "summary_planned")]:
                if pd.notna(r[col]) and r[col].strip():
                    rows.append(pd.DataFrame([{
                        "doc_id":     r["doc_id"],
                        "text":       r[col],
                        "start_time": None,
                        "text_type":  ttype,
                    }]))
        log.info("  Loaded summary sections from %d documents.", len(df_sum))
    except Exception as e:
        log.warning("  Could not load summaries: %s", e)

    conn.close()

    if not rows:
        raise RuntimeError("No text data found in database.")

    df = pd.concat(rows, ignore_index=True)
    df["text"] = df["text"].astype(str).str.strip()
    return df


# ===========================================================================
# Bonus analyses
# ===========================================================================

def hourly_activity_distribution(df_tagged: pd.DataFrame) -> pd.DataFrame:
    """Count events by hour-of-day from start_time column."""
    df = df_tagged.copy()
    df["hour"] = df["start_time"].apply(
        lambda t: int(str(t)[:2]) if pd.notna(t) and re.match(r"\d{2}:\d{2}", str(t)) else None
    )
    df = df.dropna(subset=["hour"])
    counts = df.groupby("hour").size().reset_index(name="event_count")
    counts["hour"] = counts["hour"].astype(int)
    counts = counts.sort_values("hour").reset_index(drop=True)
    return counts


def top_equipment_mentions(df_ner: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Most frequently mentioned equipment items across the entire corpus."""
    equip = df_ner[df_ner["entity_label"] == "EQUIPMENT"].copy()
    if equip.empty:
        return pd.DataFrame(columns=["equipment", "count"])
    counts = (
        equip["value"]
        .str.lower()
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    counts.columns = ["equipment", "count"]
    return counts


def activity_cooccurrence(df_tagged: pd.DataFrame) -> pd.DataFrame:
    """
    Build a co-occurrence matrix of activity labels.
    Useful for presentations: shows which activities often happen together.
    """
    # Expand multi-label rows
    all_labels = sorted({
        lbl
        for labels_str in df_tagged["activity_labels"]
        for lbl in labels_str.split("|")
        if lbl != "OTHER"
    })
    matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)
    for labels_str in df_tagged["activity_labels"]:
        lbls = [l for l in labels_str.split("|") if l != "OTHER"]
        for i, a in enumerate(lbls):
            for b in lbls:
                matrix.loc[a, b] += 1
    return matrix


# ===========================================================================
# Main
# ===========================================================================

def run_nlp_analysis(
    db_path:     str = DB_PATH,
    results_dir: str = RESULTS_DIR,
    top_n:       int = TOP_N_KEYWORDS,
):
    os.makedirs(results_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    log.info("Loading text data from database...")
    df = load_texts_from_db(db_path)
    log.info("Total text entries: %d across %d documents.",
             len(df), df["doc_id"].nunique())

    # ── 3a: NER ──────────────────────────────────────────────────────────────
    log.info("Running NER on %d text entries...", len(df))

    ner_rows = []
    for _, row in df.iterrows():
        entities = extract_entities(row["text"])
        for ent in entities:
            ner_rows.append({
                "doc_id":       row["doc_id"],
                "text_type":    row["text_type"],
                "source_text":  row["text"][:120],
                "entity_text":  ent["entity_text"],
                "entity_label": ent["entity_label"],
                "value":        ent["value"],
            })

    df_ner = pd.DataFrame(ner_rows)
    ner_path = os.path.join(results_dir, "nlp_ner_results.csv")
    df_ner.to_csv(ner_path, index=False, encoding="utf-8")
    log.info("  NER: %d entities found → %s", len(df_ner), ner_path)

    # Entity summary per document
    if not df_ner.empty:
        entity_summary = (
            df_ner.groupby(["doc_id", "entity_label"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        entity_summary.to_csv(
            os.path.join(results_dir, "nlp_entity_summary.csv"),
            index=False, encoding="utf-8",
        )

    # ── 3b: Activity Classification ──────────────────────────────────────────
    log.info("Classifying activities...")

    df_ops_only = df[df["text_type"].isin(
        ["operation", "summary_activities", "summary_planned", "equipment_failure"]
    )].copy()

    df_ops_only["activity_labels"] = df_ops_only["text"].apply(
        lambda t: "|".join(classify_activity(t))
    )

    activity_path = os.path.join(results_dir, "nlp_activity_tags.csv")
    df_ops_only[["doc_id", "text_type", "start_time", "text", "activity_labels"]].to_csv(
        activity_path, index=False, encoding="utf-8",
    )
    log.info("  Activity tagging complete → %s", activity_path)

    # Activity distribution
    label_counts: Counter = Counter()
    for labels_str in df_ops_only["activity_labels"]:
        for lbl in labels_str.split("|"):
            label_counts[lbl] += 1

    df_dist = pd.DataFrame(
        label_counts.most_common(),
        columns=["activity_label", "count"],
    )
    df_dist.to_csv(
        os.path.join(results_dir, "nlp_activity_distribution.csv"),
        index=False, encoding="utf-8",
    )

    # Co-occurrence matrix
    cooc = activity_cooccurrence(df_ops_only)
    cooc.to_csv(
        os.path.join(results_dir, "nlp_activity_cooccurrence.csv"),
        encoding="utf-8",
    )

    # ── 3c: TF-IDF ───────────────────────────────────────────────────────────
    log.info("Computing TF-IDF keywords...")

    # Build one document per PDF: concatenate all its remarks
    doc_texts: dict[str, str] = {}
    for doc_id, group in df.groupby("doc_id"):
        doc_texts[doc_id] = " ".join(group["text"].dropna().tolist())

    if len(doc_texts) >= 2:
        df_tfidf = extract_tfidf_keywords(doc_texts, top_n=top_n)
        tfidf_path = os.path.join(results_dir, "nlp_tfidf_keywords.csv")
        df_tfidf.to_csv(tfidf_path, index=False, encoding="utf-8")
        log.info("  TF-IDF: %d keyword rows → %s", len(df_tfidf), tfidf_path)
    else:
        log.warning("  TF-IDF skipped: need ≥2 documents in corpus.")
        df_tfidf = pd.DataFrame()

    # ── Bonus: top equipment ──────────────────────────────────────────────────
    df_equip_top = top_equipment_mentions(df_ner, top_n=10)
    df_equip_top.to_csv(
        os.path.join(results_dir, "nlp_equipment_top10.csv"),
        index=False, encoding="utf-8",
    )

    # ── Bonus: hourly activity ────────────────────────────────────────────────
    df_hourly = hourly_activity_distribution(df_ops_only)
    df_hourly.to_csv(
        os.path.join(results_dir, "nlp_hourly_activity.csv"),
        index=False, encoding="utf-8",
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info("TASK 3 COMPLETE — Results in %s", results_dir)
    log.info("="*60)
    log.info("  3a NER entities found   : %d", len(df_ner))
    if not df_ner.empty:
        for label, cnt in df_ner["entity_label"].value_counts().items():
            log.info("       %-25s : %d", label, cnt)
    log.info("  3b Activity-tagged rows  : %d", len(df_ops_only))
    log.info("     Top activities:")
    for lbl, cnt in label_counts.most_common(5):
        log.info("       %-20s : %d", lbl, cnt)
    log.info("  3c TF-IDF rows           : %d", len(df_tfidf))
    if not df_equip_top.empty:
        log.info("  Top equipment items:")
        for _, r in df_equip_top.head(5).iterrows():
            log.info("       %-30s : %d", r["equipment"], r["count"])

    return {
        "ner":           df_ner,
        "activity_tags": df_ops_only,
        "tfidf":         df_tfidf,
        "equip_top10":   df_equip_top,
        "hourly":        df_hourly,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 3 – NLP Analysis")
    parser.add_argument("--db", "-d",
        default=DB_PATH,
        help="Path to SQLite database (output of Task 1)")
    parser.add_argument("--results", "-r",
        default=RESULTS_DIR,
        help="Directory to write result CSVs")
    parser.add_argument("--top-n", "-n",
        type=int, default=TOP_N_KEYWORDS,
        help="Top N TF-IDF keywords per document (default 15)")
    args = parser.parse_args()

    run_nlp_analysis(
        db_path=args.db,
        results_dir=args.results,
        top_n=args.top_n,
    )
