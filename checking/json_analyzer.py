import json
import os

def check_all_requirements(json_path):
    print(f"🔍 Task 1 Tələblərinin Analizi Başladı: {json_path}\n")
    print("=" * 60)
    
    if not os.path.exists(json_path):
        print(f"❌ Xəta: {json_path} faylı tapılmadı!")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_docs = len(data)
    if total_docs == 0:
        print("Baza boşdur.")
        return

    # Task 1-də tələb olunan bütün sahələr
    requirements = {
        "Metadata": [
            "Wellbore ID", "Report number", "Period", "Status", 
            "Operator", "Rig Name", "Drilling Contractor", "Spud Date", 
            "Water depth MSL (m)", "Elevation RKB-MSL (m)", "Tight well", 
            "HPHT", "Current Depth mMD", "Current Depth mTVD", 
            "Kick Off Depth mMD", "Kick Off Depth mTVD", "Last Casing Depth mMD", 
            "Plug Back Depth mMD", "Formation Strength (g/cm3)", 
            "Hole Diameter (in)", "Pressure Test Type"
        ],
        "Summaries": [
            "Summary of activities", 
            "Summary of planned activities"
        ],
        "Tables": [
            "Operations Table", 
            "Equipment Failure Table", 
            "Drilling Fluid Table"
        ]
    }

    # Statistikaları toplamaq üçün lüğət
    stats = {
        "Metadata": {field: 0 for field in requirements["Metadata"]},
        "Summaries": {field: 0 for field in requirements["Summaries"]},
        "Tables": {field: 0 for field in requirements["Tables"]}
    }

    # JSON-u oxuyub sayırıq
    for doc in data:
        # Metadata
        meta = doc.get("Metadata", {})
        for field in requirements["Metadata"]:
            val = meta.get(field)
            if val is not None and str(val).strip() not in ["", "null", "None"]:
                stats["Metadata"][field] += 1
                
        # Summaries
        sums = doc.get("Summaries", {})
        for field in requirements["Summaries"]:
            val = sums.get(field)
            if val is not None and str(val).strip() not in ["", "null", "None"]:
                stats["Summaries"][field] += 1
                
        # Tables (Cədvəllərin içində ən az 1 sətir varsa, uğurlu sayırıq)
        tabs = doc.get("Tables", {})
        for field in requirements["Tables"]:
            val = tabs.get(field)
            if isinstance(val, list) and len(val) > 0:
                stats["Tables"][field] += 1

    # Nəticələrin Çap Edilməsi
    def print_section(title, section_key):
        print(f"\n📌 {title.upper()}")
        print("-" * 60)
        for field, count in stats[section_key].items():
            rate = (count / total_docs) * 100
            
            # Status rəngləndiricisi (Terminalda vizual üçün)
            if rate >= 90:
                icon = "🟢"
            elif rate >= 50:
                icon = "🟡"
            elif rate > 10:
                icon = "🟠"
            else:
                icon = "🔴"
                
            print(f"{icon} {field:<30}: {count}/{total_docs} doludur ({rate:.1f}%)")

    print(f"Ümumi analiz edilən sənəd sayı: {total_docs}")
    print_section("1. Metadata (Sənəd Başlıqları)", "Metadata")
    print_section("2. Summary (Xülasə Mətnləri)", "Summaries")
    print_section("3. Tables (Əsas Cədvəllər)", "Tables")
    
    print("\n" + "=" * 60)
    print("💡 QEYD: Bəzi cədvəllərin (məs. Equipment Failure) və sahələrin az olması")
    print("normaldır, çünki bu hadisələr hər hesabatda baş vermir.")

if __name__ == "__main__":
    # JSON faylının yolunu bura qeyd et:
    json_file_path = "D:\drilling-report-nlp\data\processed\document_database.json" 
    check_all_requirements(json_file_path)