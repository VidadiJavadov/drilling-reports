import pdfplumber
import re
import os

# Yoxlamaq istədiyin faylın yolu
pdf_yolu = "D:/drilling-report-nlp/data/raw/pdf_reports/PDF_version_1000/15_9_19_A_1997_07_28.pdf"

# --- YENİ REGEX ŞABLONLARI ---
# P+e+r+i+o+d+ həm "Period", həm də "PPeerriioodd" variantlarını tutur
regex_patterns = {
    "Period": r"(?i)P+e+r+i+o+d+\s*:\s*([0-9\-: ]+\-\s*[0-9\-: ]+)",
    "Status": r"(?i)Status\s*:\s*([A-Za-z]+)"
}

if not os.path.exists(pdf_yolu):
    print(f"❌ Fayl tapılmadı: {pdf_yolu}")
else:
    with pdfplumber.open(pdf_yolu) as pdf:
        text = pdf.pages[0].extract_text()
        
        print("--- [TEST BAŞLADI] ---")
        print(f"Fayl: {os.path.basename(pdf_yolu)}\n")

        for field, pattern in regex_patterns.items():
            match = re.search(pattern, text)
            if match:
                print(f"✅ {field} TAPILDI: '{match.group(1).strip()}'")
            else:
                print(f"❌ {field} tapılmadı!")
                # Əgər tapılmasa, mətndə həmin sözün necə göründüyünü göstərək
                # (Məsələn, "Status" sözü "SSttaattuuss" kimi də ola bilər)
                print(f"   Məsləhət: Mətndə '{field[0]*2}' hərfi ilə başlayan hissəni yoxla.")

        print("\n--- [MƏTNİN İLK HİSSƏSİ (REPR)] ---")
        print(repr(text[:500]))