import pdfplumber
import os

# Yolu faylına uyğun tam olaraq veririk
pdf_path = "../../data/raw/pdf_reports/PDF_version_1000/15_9_19_A_1980_01_01.pdf"

if not os.path.exists(pdf_path):
    print("Fayl tapılmadı, yolu yoxla.")
else:
    with pdfplumber.open(pdf_path) as pdf:
        # 1. İlk səhifənin mətnini çapa veririk ki, Regex üçün formatı görək
        first_page_text = pdf.pages[0].extract_text()
        print("--- İLK SƏHİFƏNİN MƏTNİ (Metadata üçün) ---")
        print(first_page_text[:1000])  # İlk 1000 simvol bəs edər
        
        print("\n" + "="*50 + "\n")
        
        # 2. Bütün cədvəllərin başlıqlarını çapa veririk ki, Operations Table-ı tapaq
        print("--- CƏDVƏLLƏRİN STRUKTURU ---")
        table_counter = 1
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                cleaned_table = [row for row in table if any(cell for cell in row)]
                if cleaned_table:
                    print(f"Cədvəl {table_counter} (Səhifə {i+1}):")
                    print(f"Başlıq (Row 1): {cleaned_table[0]}")
                    if len(cleaned_table) > 1:
                        print(f"Məlumat (Row 2): {cleaned_table[1]}")
                    print("-" * 40)
                    table_counter += 1