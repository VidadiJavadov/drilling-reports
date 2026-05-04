import pandas as pd
import os

# Excel faylının yolu (Sənin strukturuna uyğunlaşdırılıb)
excel_path = "../../data/raw/nds_events.xlsx"

if os.path.exists(excel_path):
    # Excel faylını oxuyuruq
    df_nds = pd.read_excel(excel_path)
    
    print("--- NDS EVENTS EXCEL FAYLI ---")
    print(f"Ümumi sətir (qəza) sayı: {len(df_nds)}")
    print("\nSütun adları:")
    print(df_nds.columns.tolist())
    
    print("\nİlk 2 sətir:")
    # Bütün sütunları tam görmək üçün bəzi pandas tənzimləmələri
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_nds.head(2))
else:
    print("Excel faylı tapılmadı! Yolu yoxlayın.")