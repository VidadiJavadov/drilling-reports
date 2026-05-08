# Drilling Daily Report Pipeline — eiLink R&D Centre Technical Task

A three-part pipeline that parses drilling daily report PDFs, stores structured data in SQLite, matches NDS events to their source documents, and runs NLP analysis on extracted free-text fields.

---

## What this does

Drilling daily reports come as dense, inconsistently formatted PDFs. This project turns them into something you can actually query and analyse. Roughly speaking it does three things:

1. **Reads every PDF** in a folder, pulls out the header metadata, operations table, drilling fluid samples, equipment failures, gas readings, and the summary sections — then writes everything into a SQLite database and a JSON file.
2. **Matches NDS events** from an Excel sheet to the operation remarks that most likely describe the same incident, using a combination of semantic similarity, BM25, and a few domain-specific boosts.
3. **Runs NLP** on all the free text — named entity recognition for depths/equipment/measurements, activity tagging (TRIP\_IN, CEMENT, etc.), and TF-IDF keyword extraction per report.

---

## Project structure

DRILLING-REPORT-NLP/
├── data/
│   ├── raw/
│   │   ├── pdf_reports/
│   │   │   └── PDF_version_1000/        # source PDFs go here
│   │   └── nds_events.xlsx
│   └── processed/
│       ├── document_database.sqlite
│       └── document_database.json
├── results/
│   ├── event_matching_results.csv
│   ├── event_matching_top_k.csv
│   ├── nlp_activity_cooccurrence.csv
│   ├── nlp_activity_distribution.csv
│   ├── nlp_activity_tags.csv
│   ├── nlp_entity_summary.csv
│   ├── nlp_equipment_top10.csv
│   ├── nlp_hourly_activity.csv
│   ├── nlp_ner_results.csv
│   └── nlp_tfidf_keywords.csv
├── src/
│   ├── extraction/
│   │   └── full_pdf_parser.py           # Task 1 — PDF parsing & database storage
│   ├── matching/
│   │   └── embedding_matcher.py         # Task 2 — NDS event matching
│   └── nlp/
│       └── nlp_pipeline.py              # Task 3 — NER, activity tagging, TF-IDF
├── venv/
└── README.md

---

## Setup

Python 3.10+ is recommended. Install dependencies with:

```bash
pip install pdfplumber sentence-transformers scikit-learn pandas openpyxl tqdm
```

That covers everything all three tasks need. No GPU required — the sentence-transformer model (`all-MiniLM-L6-v2`) runs fine on CPU, just a bit slower on large corpora.

---

## Running the pipeline

### Task 1 — Extract PDFs into the database

```bash
python src/task1_extraction.py \
  --input  data/raw/pdf_reports/PDF_version_1000 \
  --json   data/processed/document_database.json \
  --db     data/processed/document_database.sqlite
```

Add `--limit 10` if you just want to test on a handful of files first.

What gets extracted from each PDF:

- **Header metadata** — wellbore ID, report number, period, operator, rig name, spud date, water depth, current/kick-off/casing depths, hole diameter, formation strength, HPHT flag, etc.
- **Operations table** — start/end time, depth, main activity, sub-activity, state, remark
- **Drilling fluid table** — handles both normal and transposed layouts
- **Equipment failure table** — start time, depth, equipment system/class, downtime, remark
- **Gas reading table** — time, class, depth ranges, component readings (C1–IC5)
- **Summary sections** — the 24h activity summary and planned activities free text

One known quirk: some PDFs have an OCR artefact where every character is doubled ("SSttaarrtt"). The `fix_double_letter_bug` function in task1 handles this automatically.

---

### Task 2 — Match NDS events to operation remarks

```bash
python src/task2_nds_matching.py
```

Paths default to the locations above; edit the constants at the top of the file if your layout differs.

For each row in `nds_events.xlsx`, the script:
1. Filters operation remarks down to PDFs from the same well (or falls back to the same field block, then the whole DB)
2. Scores each candidate remark on four signals — semantic similarity (SBERT), BM25, problem-keyword IDF boost, and numeric value overlap
3. Writes the top-3 matches per event to CSV

**Output files:**
- `results/event_matching_results.csv` — best match per NDS event
- `results/event_matching_top_k.csv` — top 3 candidates with individual signal scores

The default ensemble weights are `semantic=0.50 / BM25=0.28 / keyword=0.14 / numeric=0.08`. These worked well on the 15/9-F-10 well set but you can tune them depending on your data.

---

### Task 3 — NLP analysis

```bash
python src/task3_nlp_analysis.py
```

Runs three analyses and saves results to `data/results/`:

**3a. Named Entity Recognition** — rule-based, no pre-trained model needed. Extracts:
- `DEPTH` — numeric depth values with units (mMD, mTVD, metres)
- `EQUIPMENT` — tool names like BOP, TDS, packer, spear BHA
- `MEASUREMENT_*` — RPM, bar, lpm, MT, m³, SG, degrees
- `TIME_REF` — HH:MM timestamps and duration expressions

**3b. Activity tagging** — assigns one or more normalised labels to each remark (TRIP\_IN, TRIP\_OUT, DRILL, CEMENT, PRESSURE\_TEST, CIRCULATE, SURVEY, FISHING, EQUIPMENT\_FAILURE, REPAIR, WAIT, PUMP, DISPLACE, CUT). Multi-label, so a remark can be both TRIP\_OUT and EQUIPMENT\_FAILURE.

**3c. TF-IDF keywords** — top 15 keywords per report using bigrams and sublinear TF weighting, with a custom drilling stopword list so generic terms like "well", "pipe", "depth" don't drown everything out.

**Bonus outputs:**
- `nlp_activity_distribution.csv` — how often each activity label appears
- `nlp_activity_cooccurrence.csv` — which labels tend to co-occur
- `nlp_equipment_top10.csv` — most mentioned equipment items across the corpus
- `nlp_hourly_activity.csv` — event counts by hour of day

---

## Output files at a glance

| File | Description |
|---|---|
| `document_database.sqlite` | Main database with all extracted tables |
| `document_database.json` | Same data, JSON format |
| `event_matching_results.csv` | Best NDS match per event |
| `event_matching_top_k.csv` | Top 3 candidates with signal scores |
| `nlp_ner_results.csv` | All entities found across the corpus |
| `nlp_entity_summary.csv` | Entity type counts per document |
| `nlp_activity_tags.csv` | Activity labels per remark |
| `nlp_activity_distribution.csv` | Label frequency counts |
| `nlp_activity_cooccurrence.csv` | Label co-occurrence matrix |
| `nlp_tfidf_keywords.csv` | Top keywords per report |
| `nlp_equipment_top10.csv` | Most mentioned equipment |
| `nlp_hourly_activity.csv` | Events by hour of day |

---

## Notes and known limitations

- The PDF parser uses `pdfplumber` directly — no external OCR service needed. That said, heavily scanned or image-only PDFs will produce empty or garbled text. The doubled-character fix covers the most common artefact seen in this dataset.
- Table classification works by finding the section heading printed above each table on the page. If a table spans a page break and its heading is on the previous page, it may fall through to the column-keyword fallback classifier, which is less reliable.
- The NDS matching falls back gracefully when no PDFs are found for the target well — first to the same field block (e.g. all 15/9-F-* wells), then to the full database. The matched file name in the output will tell you which tier was used.
- TF-IDF requires at least two documents in the corpus to compute meaningful IDF weights. Running task3 on a single PDF will skip that step with a warning.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pdfplumber` | PDF text and table extraction |
| `sentence-transformers` | SBERT semantic similarity (task 2) |
| `scikit-learn` | TF-IDF vectoriser (task 3) |
| `pandas` | Data handling throughout |
| `openpyxl` | Reading nds_events.xlsx |
| `tqdm` | Progress bars in task 1 |
