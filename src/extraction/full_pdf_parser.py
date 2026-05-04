import pdfplumber
import os
import re
import json
from tqdm import tqdm

def clean_text(text):
    """
    Cleans extracted text by removing redundant newlines and multiple spaces.
    Detects and fixes the 'SSttaarrtt' double-letter PDF OCR bug without breaking natural double letters.
    """
    if not isinstance(text, str): return text
    
    # 1. Remove newlines
    text = text.replace('\n', ' ')
    
    # 2. Fix the Double Letter Bug (e.g., SSttaarrtt ttiimmee -> Start time)
    words = text.split()
    fixed_words = []
    for w in words:
        # Check if the word consists entirely of perfectly repeating pairs
        # e.g. length is even, and w[0]==w[1], w[2]==w[3], etc.
        if len(w) >= 4 and len(w) % 2 == 0 and all(w[i] == w[i+1] for i in range(0, len(w), 2)):
            fixed_words.append(w[::2]) # Take every second character
        else:
            fixed_words.append(w)
            
    text = " ".join(fixed_words)
    
    # 3. Remove extra spaces
    return re.sub(r'\s+', ' ', text).strip()

def make_pattern(word):
    """
    Safely creates a regex pattern that allows spaces between letters.
    E.g., "Wellbore" -> r'W\s*e\s*l\s*l\s*b\s*o\s*r\s*e'
    This prevents the empty-string matching bug caused by W?e?l?l?
    """
    return r'\s*'.join(list(word))

def extract_metadata_and_summaries(full_text):
    """
    Task 1: Uses robust Regex to extract Header/Metadata and Summaries.
    """
    doc_data = {
        "Metadata": {},
        "Summaries": {}
    }
    
    # (?=\s{2,}|\n|$) ensures that the capture stops if there are 2+ spaces (next column) or a newline.
    terminator = r'(?=\s{2,}|\n|$)'
    
    patterns = {
        "Wellbore ID": r'(?i)' + make_pattern("Wellbore") + r'[^\n:]*[:\s]+([A-Za-z0-9/\-\s]+?)' + terminator,
        "Report number": r'(?i)' + make_pattern("Report") + r'[^\n:]*[:\s]+(\d+)',
        "Period": r'(?i)' + make_pattern("Period") + r'[^\n:]*[:\s]+([\d\-\s:]+?)' + terminator,
        "Status": r'(?i)' + make_pattern("Status") + r'[^\n:]*[:\s]+([A-Za-z]+?)' + terminator,
        "Operator": r'(?i)' + make_pattern("Operator") + r'[^\n:]*[:\s]+([A-Za-z0-9\.\-\s&]+?)' + terminator,
        "Rig Name": r'(?i)' + make_pattern("Rig Name") + r'[^\n:]*[:\s]+([A-Za-z0-9\.\-\s]+?)' + terminator,
        "Spud Date": r'(?i)' + make_pattern("Spud Date") + r'[^\n:]*[:\s]+([\d\-]+?)' + terminator,
        "Water depth MSL (m)": r'(?i)' + make_pattern("Water depth") + r'[^\n\d]*[:\s]+(\d{1,4}(?:\.\d+)?)' + terminator,
        "Tight well": r'(?i)' + make_pattern("Tight well") + r'[^\n:]*[:\s]+([YyNn])',
        "HPHT": r'(?i)H\s*P\s*H\s*T[^\n:]*[:\s]+([YyNn])',
        "Depth at Kick Off mMD": r'(?i)' + make_pattern("Kick Off") + r'[^\n\d]*[:\s]+([\d\.]+?)' + terminator
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, full_text)
        doc_data["Metadata"][key] = match.group(1).strip() if match else None

    # --- SUMMARY SECTION EXTRACTION ---
    # Safely matches summaries avoiding optional character bugs
    act_pattern = r'(?i)' + make_pattern("Summary of activities") + r'[^\n]*\n(.*?)(?=\n\s*' + make_pattern("Summary of plan") + r'|\n\s*Survey|$)'
    act_match = re.search(act_pattern, full_text, re.DOTALL)
    doc_data["Summaries"]["Summary of activities"] = clean_text(act_match.group(1)) if act_match else None

    plan_pattern = r'(?i)' + make_pattern("Summary of planned activities") + r'[^\n]*\n(.*?)(?=\n\s*Survey|\n\s*Operations|$)'
    plan_match = re.search(plan_pattern, full_text, re.DOTALL)
    doc_data["Summaries"]["Summary of planned activities"] = clean_text(plan_match.group(1)) if plan_match else None

    return doc_data

def identify_table_type(headers):
    """
    Identifies the type of table based on header keywords.
    """
    if not headers: return "Other"
    raw_str = "".join([str(h) for h in headers if h]).lower()
    cleaned_str = re.sub(r'(.)\1+', r'\1', raw_str.replace('\n', '').replace(' ', '').replace('-', ''))
    
    if "activity" in cleaned_str or "remark" in cleaned_str or "mainsub" in cleaned_str:
        return "Operations Table"
    elif "fluid" in cleaned_str or "density" in cleaned_str or "viscosity" in cleaned_str:
        return "Drilling Fluid Table"
    elif "equipment" in cleaned_str or "failure" in cleaned_str or "downtime" in cleaned_str:
        return "Equipment Failure Table"
    return "Other"

def extract_all_data_from_pdf(pdf_path):
    """
    Orchestrates extraction of text and tables from a single PDF.
    """
    full_text = ""
    tables_data = {
        "Operations Table": [],
        "Equipment Failure Table": [],
        "Drilling Fluid Table": []
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t: full_text += t + "\n"
            
            tables = page.extract_tables()
            for table in tables:
                cleaned_table = [row for row in table if any(cell for cell in row)]
                if not cleaned_table: continue
                
                # First row is header, clean headers early
                headers = [clean_text(h) for h in cleaned_table[0]]
                table_type = identify_table_type(headers)
                
                if table_type in tables_data:
                    for row in cleaned_table[1:]:
                        row_dict = {}
                        for i, header in enumerate(headers):
                            val = clean_text(row[i]) if i < len(row) else None
                            # Use Col_i if header is somehow empty
                            valid_header = header if (header and len(header) > 0) else f"Col_{i}"
                            row_dict[valid_header] = val
                        tables_data[table_type].append(row_dict)

    document_record = extract_metadata_and_summaries(full_text)
    document_record["Tables"] = tables_data
    return document_record

def process_and_save_to_db(folder_path, output_db_path, limit=None):
    """
    Main execution loop.
    """
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
    if limit: pdf_files = pdf_files[:limit]
    
    database_documents = []

    print(f"Starting Task 1: Full Extraction on {len(pdf_files)} files...")
    for file in tqdm(pdf_files, desc="Parsing all PDF sections"):
        full_path = os.path.join(folder_path, file)
        try:
            doc_data = extract_all_data_from_pdf(full_path)
            doc_data["_id"] = file
            database_documents.append(doc_data)
        except Exception as e:
            print(f"Error parsing {file}: {e}")

    os.makedirs(os.path.dirname(output_db_path), exist_ok=True)
    with open(output_db_path, 'w', encoding='utf-8') as f:
        json.dump(database_documents, f, indent=4, ensure_ascii=False)
        
    print(f"\n✅ Task 1 Completed! All data saved to: {output_db_path}")

if __name__ == "__main__":
    RAW_FOLDER = "../../data/raw/pdf_reports/PDF_version_1000" 
    OUTPUT_JSON = "../../data/processed/document_database.json"
    
    process_and_save_to_db(RAW_FOLDER, OUTPUT_JSON, limit=None)
