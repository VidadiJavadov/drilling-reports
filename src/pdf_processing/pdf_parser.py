import pdfplumber
import os
import re
import pandas as pd
from tqdm import tqdm

def clean_doubled_text(text):
    """
    Fixes double letters (RReemmaarrkk -> Remark).
    At the same time, it replaces unnecessary \n characters in the text with spaces.
    """
    if not isinstance(text, str): return text
    
    # 1. replace \n with ' ' (SSttaarrtt\nttiimmee -> SSttaarrtt ttiimmee)
    text = text.replace('\n', ' ')
    
    # 2. fix double letters
    cleaned = re.sub(r'(.)\1', r'\1', text)
    
    return cleaned.strip()

def clean_header_for_identification(headers):
    """
    It does a cleanup (no spaces, no special characters) to check headers.
    """
    if not headers: return ""
    raw_str = "".join([str(h) for h in headers if h]).lower()
    # removes all spaces and \n's
    raw_str = raw_str.replace('\n', '').replace(' ', '').replace('-', '')
    # We reduce all consecutive identical letters to a single letter (aaccttiivviittyy -> activity)
    cleaned_str = re.sub(r'(.)\1+', r'\1', raw_str)
    return cleaned_str

def identify_table_type(headers):
    cleaned_header_str = clean_header_for_identification(headers)
    
    if "activity" in cleaned_header_str or "remark" in cleaned_header_str or "mainsub" in cleaned_header_str:
        return "Operations Table"
    elif "fluid" in cleaned_header_str or "density" in cleaned_header_str:
        return "Drilling Fluid Table"
    elif "equipment" in cleaned_header_str or "failure" in cleaned_header_str:
        return "Equipment Failure Table"
    
    return "Other"

def extract_data_from_pdf(pdf_path):
    operations_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                cleaned_table = [row for row in table if any(cell for cell in row)]
                if not cleaned_table: continue
                
                headers = cleaned_table[0]
                table_type = identify_table_type(headers)
                
                if table_type == "Operations Table":
                    # If we find the Operations Table, we clear all its rows and cells
                    cleaned_operations_table = []
                    for row in cleaned_table:
                        cleaned_row = [clean_doubled_text(cell) for cell in row]
                        cleaned_operations_table.append(cleaned_row)
                        
                    operations_tables.append(cleaned_operations_table)

    return {"operations_tables": operations_tables}

def process_folder(folder_path, limit=5):
    all_extracted_data = []
    pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
    
    for file in tqdm(pdf_files[:limit], desc="PDF-lər oxunur"):
        full_path = os.path.join(folder_path, file)
        extracted_data = extract_data_from_pdf(full_path)
        
        all_extracted_data.append({
            "file_name": file,
            "data": extracted_data
        })

    return all_extracted_data

def save_operations_to_csv(data, output_path):
    """
    It combines all extracted tables into a Pandas DataFrame and saves it as CSV.
    """
    all_rows = []
    
    for item in data:
        file_name = item['file_name']
        ops_tables = item['data'].get('operations_tables', [])
        
        for table in ops_tables:
            # The table must contain at least a title and 1 row of data.
            if not table or len(table) < 2: 
                continue
                
            headers = table[0] # Our cleaned headers
            
            for row in table[1:]:
                # We create a dictionary for each line
                row_dict = {"file_name": file_name}
                
                for i, header in enumerate(headers):
                    if i < len(row):
                        row_dict[header] = row[i]
                    else:
                        row_dict[header] = None
                        
                all_rows.append(row_dict)
                
    # Convert the list of dictionaries to a Pandas DataFrame
    df = pd.DataFrame(all_rows)
    
    # Create the folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # We save as CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✅ Data saved as CSV: {output_path}")
    print(f"✅ Total number of lines: {len(df)}")
    
    return df

if __name__ == "__main__":
    folder = "../../data/raw/pdf_reports/PDF_version_1000" 
    
    if os.path.exists(folder) and len(os.listdir(folder)) > 0:
        # 10 PDF-də yoxlayırıq
        data = process_folder(folder, limit=None)

        print("\n--- NEW ANALYSIS RESULT ---")
        operations_found = 0
        
        for item in data:
            ops_tables = item['data']['operations_tables']
            if len(ops_tables) > 0:
                operations_found += 1
                print(f"✅ SUCCESS! {item['file_name']} -> Operations Table found.")
                print(f"  Cleaned Headlines: {ops_tables[0][0]}")
            else:
                print(f"❌ NO - {item['file_name']}")
                
        print(f"\nFinal: {operations_found} found in the Operations Table.")
        
        # We write the data as CSV to the processed folder
        output_csv = "../../data/processed/operations_table.csv"
        df = save_operations_to_csv(data, output_csv)
        
        # Let's look at the first 3 rows of the DataFrame
        if not df.empty:
            print("\n--- DATAFRAME FIRST 3 ROWS ---")
            print(df.head(3).to_string())
        else:
            print("\n❌ DataFrame is empty, table not found.")
            
    else:
        print(f"Folder not found: {folder}")
