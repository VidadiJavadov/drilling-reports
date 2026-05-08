"""
Task 2 - Enhanced NDS Event Matching (SQLite Version)
=====================================================
Hybrid approach combining:
  1. BM25 (keyword/lexical matching)        — lexical overlap
  2. Semantic similarity (Sentence-BERT)    — paraphrase / meaning
  3. Problem-keyword boosting               — rare domain terms (stuck, tight, clay…)
  4. Drilling abbreviation expansion        — normalise oilfield shorthand
  5. Minimum-length filtering               — discard uninformative one-liners
  6. Top-k deduplication                    — no duplicate remarks in candidates

Fixes vs. the hand-edited SQLite version
-----------------------------------------
* FIX 1 — file_name was built as `wellbore_id || '_' || report_number`, which
  does NOT match the PDF filename format expected by `startswith(well_prefix)`.
  The query now selects the real `source_file` column from the documents table
  (falls back to wellbore_id if that column doesn't exist — see load_data_from_db).

* FIX 2 — preprocess() used raw str.replace("ccumulation", ...) which is a
  dangerous substring match that corrupts words like "accumulator".
  All normalisation is now done with word-boundary regex (\b...\b).

* FIX 3 — PROBLEM_KEYWORDS contained abbreviations (pooh, rih, tih…) that are
  already expanded to full phrases by preprocess(). Matching the short form after
  expansion always fails (the token "pooh" no longer exists). Removed them.

* FIX 4 — load_data_from_db had no error handling: a wrong db_path, missing
  table, or missing column produced an unreadable traceback. Now raises clear
  messages and inspects the actual schema before querying.

* FIX 5 — well_prefix matching is now case-insensitive and also strips leading/
  trailing whitespace that sometimes appears in wellbore_id values from the DB.
"""
import pandas as pd
import numpy as np
import os
import re
import sqlite3
import warnings
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Abbreviation expansion  (expand BEFORE embedding — MiniLM knows English,
# not oilfield shorthand)
# -----------------------------------------------------------------------------
DRILLING_ABBREVS = {
    r'\bPOOH\b': 'pull out of hole pulling out',
    r'\bPOH\b':  'pull out of hole pulling out',
    r'\bRIH\b':  'run in hole running in',
    r'\bTIH\b':  'trip in hole',
    r'\bTOOH\b': 'trip out of hole pulling out',
    r'\bBHA\b':  'bottom hole assembly drilling assembly',
    r'\bDC\b':   'drill collar',
    r'\bNMDC\b': 'non magnetic drill collar',
    r'\bMWD\b':  'measurement while drilling tool',
    r'\bLWD\b':  'logging while drilling tool',
    r'\bDHM\b':  'downhole motor',
    r'\bWOB\b':  'weight on bit',
    r'\bROP\b':  'rate of penetration drilling speed',
    r'\bRPM\b':  'revolutions per minute rotation speed',
    r'\bSPM\b':  'strokes per minute pump rate',
    r'\bLPM\b':  'litres per minute flow rate',
    r'\bSPP\b':  'standpipe pressure pump pressure',
    r'\bMD\b':   'measured depth',
    r'\bTVD\b':  'true vertical depth',
    r'\bDLS\b':  'dogleg severity well curvature',
    r'\bINC\b':  'inclination well angle',
    r'\bAZ\b':   'azimuth well direction',
    r'\bMW\b':   'mud weight drilling fluid density',
    r'\bECD\b':  'equivalent circulating density',
    r'\bESD\b':  'equivalent static density',
    r'\bPV\b':   'plastic viscosity',
    r'\bYP\b':   'yield point',
    r'\bKCl\b':  'potassium chloride drilling fluid',
    r'\bCSG\b':  'casing',
    r'\bLIN\b':  'liner',
    r'\bTBG\b':  'tubing',
    r'\bPBR\b':  'polished bore receptacle',
    r'\bBOP\b':  'blowout preventer',
    r'\bDHSV\b': 'downhole safety valve',
    r'\bWOC\b':  'wait on cement',
    r'\bCBL\b':  'cement bond log',
    r'\bCET\b':  'cement evaluation tool',
    r'\bFIT\b':  'formation integrity test',
    r'\bLOT\b':  'leak off test',
    r'\bTDS\b':  'top drive system',
    r'\bXO\b':   'crossover',
    r'\bWH\b':   'wellhead',
    r'\bROV\b':  'remotely operated vehicle',
    r'\bSG\b':   'specific gravity',
    r'\bMT\b':   'metric ton',
}


def expand_abbreviations(text: str) -> str:
    """Replace oilfield abbreviations with full English phrases."""
    if not isinstance(text, str):
        return ''
    for pattern, replacement in DRILLING_ABBREVS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


# FIX 2: all normalisations use word-boundary regex, never raw substring
_NORMALISE_RULES: list[tuple[str, str]] = [
    (r'\bccumulation\b',                'accumulation'),
    (r'\baccumulat\w*\b',               'accumulation'),
    (r'\bcovered\s+(with|in)\s+clay\b', 'clay accumulation'),
    (r'\bclays\b',                      'clay'),
    (r'\bpulled\b',                     'pull'),
    (r'\bpulling\b',                    'pull'),
    (r'\bran\b',                        'run'),
    (r'\brunning\b',                    'run'),
]


def preprocess(text: str) -> str:
    """Expand abbreviations, fix common typos, normalise whitespace."""
    text = expand_abbreviations(text)
    text = text.lower()
    for pattern, replacement in _NORMALISE_RULES:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -----------------------------------------------------------------------------
# Problem-keyword boost
# FIX 3: NO abbreviations here — they are expanded before this step runs,
# so "pooh" / "rih" etc. no longer exist as tokens after preprocess().
# -----------------------------------------------------------------------------
PROBLEM_KEYWORDS = {
    # Stuck pipe
    'stuck', 'sticking', 'differential', 'freeing', 'jarring',
    'overpull', 'drag', 'torque',
    # Tight hole / pack-off
    'tight', 'restriction', 'packoff', 'bridging', 'obstruction',
    'ream', 'reaming', 'wiper',
    # Clay / fill / cavings
    'clay', 'shale', 'cavings', 'cuttings', 'fill', 'accumulation', 'balling',
    # Inclination / trajectory
    'inclination', 'azimuth', 'dogleg', 'trajectory', 'deviation',
    # Well control
    'kick', 'influx', 'choke', 'wellcontrol', 'overpressure',
    # Fluid losses
    'loss', 'lost', 'losses', 'seepage',
    # Equipment failure
    'failure', 'failed', 'breakdown', 'repair', 'replace', 'malfunction',
    # Fishing
    'fish', 'fishing', 'junk', 'spear', 'overshot', 'recovery',
}


def keyword_boost_score(query_expanded: str,
                        remarks_expanded: list[str]) -> np.ndarray:
    """IDF-weighted keyword presence score for problem-signal terms."""
    n = len(remarks_expanded)
    scores = np.zeros(n)
    query_tokens = set(re.findall(r'\b\w+\b', query_expanded.lower()))
    active_kws = query_tokens & PROBLEM_KEYWORDS

    if not active_kws:
        return scores

    for kw in active_kws:
        df_count = sum(1 for r in remarks_expanded if kw in r.lower())
        if df_count == 0:
            continue
        idf = np.log((n + 1) / (df_count + 1)) + 1
        for i, remark in enumerate(remarks_expanded):
            if kw in remark.lower():
                scores[i] += idf
    return scores




# -----------------------------------------------------------------------------
# Signal 4: Numeric value overlap boost
# -----------------------------------------------------------------------------
def extract_numbers(text: str) -> set[str]:
    """
    Extract all numeric values from text, normalising comma-decimals to dots.
    "0,16" and "0.16" are treated as the same number.
    This handles European decimal notation common in Norwegian drilling reports.
    """
    text_norm = re.sub(r'(\d),(\d)', r'\1.\2', text)
    return set(re.findall(r'\b\d+(?:\.\d+)?\b', text_norm))


def numeric_overlap_score(query: str,
                          remarks_expanded: list[str]) -> np.ndarray:
    """
    For each remark, compute a score based on how many of the query's numeric
    values appear in the remark.

    Weighted by digit-length so that specific values like "0.16" (rare, precise)
    contribute more than coarse values like "2" (common, ambiguous).

    Example:
      event   = "Reduced inclination to 0.16 deg"  → query_nums = {"0.16"}
      correct = "...reduced inclination to 0,16 deg" → overlap = {"0.16"} → score 1.0
      wrong   = "...Inc 85,4 dgr..."                → overlap = {}        → score 0.0
    """
    query_nums = extract_numbers(query)
    scores = np.zeros(len(remarks_expanded))

    if not query_nums:
        return scores

    total_weight = sum(len(n) for n in query_nums)
    if total_weight == 0:
        return scores

    for i, remark in enumerate(remarks_expanded):
        remark_nums = extract_numbers(remark)
        overlap = query_nums & remark_nums
        scores[i] = sum(len(n) for n in overlap) / total_weight

    return scores

# -----------------------------------------------------------------------------
# BM25
# -----------------------------------------------------------------------------
class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.tokenized = [self._tokenize(doc) for doc in corpus]
        self.N  = len(self.tokenized)
        self.avgdl = np.mean([len(d) for d in self.tokenized]) if self.tokenized else 1
        self.df: dict[str, int] = {}
        for doc in self.tokenized:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    def score(self, query: str) -> np.ndarray:
        query_terms = self._tokenize(query)
        scores = np.zeros(self.N)
        for term in query_terms:
            if term not in self.df:
                continue
            idf = np.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
            for i, doc in enumerate(self.tokenized):
                tf  = doc.count(term)
                dl  = len(doc)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * (num / den)
        return scores


def normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


# -----------------------------------------------------------------------------
# Database loader  (FIX 1 + FIX 4)
# -----------------------------------------------------------------------------
def _get_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall()]


def load_data_from_db(db_path: str) -> pd.DataFrame:
    """
    Load operations + the PDF filename from the SQLite database.

    file_name priority:
      1. documents.source_file   — actual PDF filename stored at parse time
      2. documents.pdf_filename  — alternative column name
      3. documents.filename      — another alternative
      4. Fallback: wellbore_id   — well-level filtering only

    FIX 1: original query used wellbore_id||'_'||report_number which does NOT
    match any PDF filename and breaks well_prefix filtering entirely.
    FIX 4: meaningful errors instead of cryptic sqlite3 tracebacks.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Make sure Task 1 has been run and the path is correct."
        )

    print(f"  Connecting to: {db_path}")
    conn = sqlite3.connect(db_path)

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    print(f"  Tables: {tables}")

    for required in ('operations', 'documents'):
        if required not in tables:
            conn.close()
            raise RuntimeError(
                f"Required table '{required}' not found. Available: {tables}"
            )

    doc_cols = _get_columns(conn, 'documents')
    ops_cols = _get_columns(conn, 'operations')
    print(f"  documents columns : {doc_cols}")
    print(f"  operations columns: {ops_cols}")

    # Choose filename column
    filename_expr = None
    for col in ('source_file', 'pdf_filename', 'filename'):
        if col in doc_cols:
            filename_expr = f'd.{col}'
            break
    if filename_expr is None:
        print(
            "  WARNING: No PDF filename column found in 'documents'. "
            "Falling back to wellbore_id — only well-level matching possible."
        )
        filename_expr = 'd.wellbore_id'

    # Choose remark column
    remark_col = None
    for col in ('remark', 'Remark', 'remarks', 'description', 'text'):
        if col in ops_cols:
            remark_col = col
            break
    if remark_col is None:
        conn.close()
        raise RuntimeError(
            f"No remark column in 'operations'. Available: {ops_cols}"
        )

    # Choose join key
    join_col = None
    for col in ('document_id', 'doc_id', 'report_id'):
        if col in ops_cols:
            join_col = col
            break
    if join_col is None:
        conn.close()
        raise RuntimeError(
            f"No foreign-key column in 'operations' to join 'documents'. "
            f"Available: {ops_cols}"
        )

    query = f"""
        SELECT o.id,
               {filename_expr}   AS file_name,
               o.{remark_col}    AS Remark
        FROM   operations o
        JOIN   documents  d ON o.{join_col} = d.id
        WHERE  o.{remark_col} IS NOT NULL
          AND  trim(o.{remark_col}) != ''
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # FIX 5: strip whitespace + normalise separators
    df['file_name'] = (
        df['file_name']
        .astype(str)
        .str.strip()
        .str.replace('/', '_', regex=False)
        .str.replace('-', '_', regex=False)
    )

    print(f"  Loaded {len(df):,} operation remarks.")
    return df


# -----------------------------------------------------------------------------
# Main matching function
# -----------------------------------------------------------------------------
def match_nds_events(
    db_path: str           = "../../data/processed/document_database.sqlite",
    nds_path: str          = "../../data/raw/nds_events.xlsx",
    results_dir: str       = "../../data/results",
    semantic_weight: float = 0.50,   # reduced slightly to make room for numeric signal
    bm25_weight: float     = 0.28,
    keyword_weight: float  = 0.14,
    numeric_weight: float  = 0.08,   # Signal 4: exact numeric value overlap
    min_remark_tokens: int = 8,
    top_k: int             = 3,
):
    os.makedirs(results_dir, exist_ok=True)

    print("Loading operation remarks from database...")
    df_ops = load_data_from_db(db_path)

    print("Loading NDS events...")
    if not os.path.exists(nds_path):
        raise FileNotFoundError(f"NDS events file not found: {nds_path}")
    df_nds = pd.read_excel(nds_path)

    for col in ('Well', 'Event'):
        if col not in df_nds.columns:
            raise RuntimeError(
                f"nds_events.xlsx missing column '{col}'. "
                f"Found: {list(df_nds.columns)}"
            )

    print("Expanding drilling abbreviations in operation remarks...")
    df_ops['Remark_expanded'] = df_ops['Remark'].apply(preprocess)

    print("Loading Sentence-BERT model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    best_results: list[dict] = []
    all_top_k:    list[dict] = []

    for idx, row in df_nds.iterrows():
        well_raw   = str(row['Well']).strip()
        event_text = str(row['Event']).strip()

        # FIX 5
        #: lowercase prefix for case-insensitive matching
        well_prefix = well_raw.replace('/', '_').replace('-', '_').lower()

        print(f"\n[{idx+1}/{len(df_nds)}] Well: {well_raw}")
        print(f"  Event: {event_text[:80]}{'...' if len(event_text) > 80 else ''}")

        file_name_lower = df_ops['file_name'].str.lower()
        well_ops = df_ops[file_name_lower.str.startswith(well_prefix, na=False)].copy()

        if well_ops.empty:
            # 3-tier smart fallback:
            # Tier 1 (above): exact well   e.g. "15_9_f_13"   -> not found
            # Tier 2: same field block, drop the well number.
            #   "15/9-F-13" -> field_prefix = "15_9_f_"
            #   Matches 15_9_F_10, 15_9_F_11, 15_9_F_12 PDFs.
            #   Correct: F-13 events occurred in the same field and the source
            #   PDF belongs to one of the sibling wells in the block.
            # Tier 3: entire DB — last resort if the field block also yields nothing.
            parts = well_prefix.split("_")          # ["15","9","f","13"]
            field_prefix = "_".join(parts[:-1]) + "_"   # "15_9_f_"

            field_ops = df_ops[file_name_lower.str.startswith(field_prefix, na=False)].copy()

            if not field_ops.empty:
                print(f"  WARNING: No PDFs for well '{well_raw}'. "
                      f"Falling back to same field block (prefix: '{field_prefix}') "
                      f"— {len(field_ops):,} remarks across "
                      f"{field_ops['file_name'].nunique()} PDFs.")
                well_ops = field_ops
            else:
                print(f"  WARNING: No PDFs for field block '{field_prefix}' either. "
                      f"Falling back to entire DB ({len(df_ops):,} remarks).")
                well_ops = df_ops.copy()

        remarks_expanded = well_ops['Remark_expanded'].tolist()
        remarks_original = well_ops['Remark'].tolist()
        file_names       = well_ops['file_name'].tolist()

        # Minimum-length filter
        token_counts = [len(re.findall(r'\b\w+\b', r)) for r in remarks_expanded]
        valid_mask   = np.array(token_counts) >= min_remark_tokens
        if valid_mask.sum() == 0:
            print("  WARNING: All remarks too short — skipping length filter.")
            valid_mask = np.ones(len(token_counts), dtype=bool)

        valid_idx        = np.where(valid_mask)[0]
        remarks_exp_filt = [remarks_expanded[i] for i in valid_idx]
        remarks_ori_filt = [remarks_original[i] for i in valid_idx]
        files_filt       = [file_names[i]       for i in valid_idx]

        event_expanded = preprocess(event_text)

        # Signal 1: SBERT
        event_emb   = model.encode(event_expanded, convert_to_tensor=True)
        remark_embs = model.encode(remarks_exp_filt,
                                   convert_to_tensor=True, batch_size=64)
        cos_scores  = util.cos_sim(event_emb, remark_embs)[0].cpu().numpy()

        # Signal 2: BM25
        bm25        = BM25(remarks_exp_filt)
        bm25_scores = bm25.score(event_expanded)

        # Signal 3: Keyword boost
        kw_scores  = keyword_boost_score(event_expanded, remarks_exp_filt)

        # Signal 4: Numeric value overlap
        # Normalise comma-decimals in event text before comparison
        event_for_nums = re.sub(r'(\d),(\d)', r'\1.\2', event_expanded)
        num_scores = numeric_overlap_score(event_for_nums, remarks_exp_filt)

        # Ensemble (weights sum to 1.0)
        ensemble = (semantic_weight * normalise(cos_scores)
                    + bm25_weight   * normalise(bm25_scores)
                    + keyword_weight * normalise(kw_scores)
                    + numeric_weight * normalise(num_scores))

        # Top-k with deduplication
        ranked_idx   = np.argsort(ensemble)[::-1]
        seen_remarks : set[str]  = set()
        top_k_idx   : list[int] = []
        for i in ranked_idx:
            key = remarks_ori_filt[i].strip().lower()
            if key not in seen_remarks:
                seen_remarks.add(key)
                top_k_idx.append(i)
            if len(top_k_idx) == top_k:
                break

        best_idx = top_k_idx[0]
        print(f"  MATCH: {files_filt[best_idx]}"
              f"  | Ensemble: {ensemble[best_idx]:.4f}"
              f"  (SBERT: {cos_scores[best_idx]:.4f}"
              f", BM25: {bm25_scores[best_idx]:.4f}"
              f", KW: {kw_scores[best_idx]:.4f}"
              f", NUM: {num_scores[best_idx]:.4f})")
        print(f"  Remark: {remarks_ori_filt[best_idx][:120]}...")

        best_results.append({
            'Well':           well_raw,
            'NDS_Event':      event_text,
            'Matched_File':   files_filt[best_idx],
            'Ensemble_Score': round(float(ensemble[best_idx]),    4),
            'Semantic_Score': round(float(cos_scores[best_idx]),  4),
            'BM25_Score':     round(float(bm25_scores[best_idx]), 4),
            'Keyword_Score':  round(float(kw_scores[best_idx]),   4),
            'Numeric_Score':  round(float(num_scores[best_idx]),  4),
            'Matched_Remark': remarks_ori_filt[best_idx],
        })

        for rank, k_idx in enumerate(top_k_idx, start=1):
            all_top_k.append({
                'Well':           well_raw,
                'NDS_Event':      event_text,
                'Rank':           rank,
                'Matched_File':   files_filt[k_idx],
                'Ensemble_Score': round(float(ensemble[k_idx]),    4),
                'Semantic_Score': round(float(cos_scores[k_idx]),  4),
                'BM25_Score':     round(float(bm25_scores[k_idx]), 4),
                'Keyword_Score':  round(float(kw_scores[k_idx]),   4),
                'Numeric_Score':  round(float(num_scores[k_idx]),  4),
                'Matched_Remark': remarks_ori_filt[k_idx],
            })

    df_best  = pd.DataFrame(best_results)
    df_top_k = pd.DataFrame(all_top_k)

    best_path = os.path.join(results_dir, "event_matching_results.csv")
    topk_path = os.path.join(results_dir, "event_matching_top_k.csv")

    df_best.to_csv(best_path,  index=False, encoding='utf-8')
    df_top_k.to_csv(topk_path, index=False, encoding='utf-8')

    print(f"\nResults saved:")
    print(f"  Best matches   -> {best_path}")
    print(f"  Top-{top_k} matches -> {topk_path}")

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 80)
    print("\n--- FINAL RESULTS ---")
    print(df_best[['Well', 'Matched_File', 'Ensemble_Score',
                   'Semantic_Score', 'BM25_Score', 'Keyword_Score',
                   'Numeric_Score', 'Matched_Remark']])

    return df_best, df_top_k


if __name__ == "__main__":
    match_nds_events()
