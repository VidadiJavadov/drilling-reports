import json
import os

def check_all_requirements(json_path):
    print(f"🔍 Analysis of Task 1 Requirements Started: {json_path}\n")
    print("=" * 60)
    
    if not os.path.exists(json_path):
        print(f" Error: {json_path} file not found!")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_docs = len(data)
    if total_docs == 0:
        print("Database is empty.")
        return

    # All required fields in Task 1
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

    # Dictionary to store statistics
    stats = {
        "Metadata": {field: 0 for field in requirements["Metadata"]},
        "Summaries": {field: 0 for field in requirements["Summaries"]},
        "Tables": {field: 0 for field in requirements["Tables"]}
    }

    # Read JSON and calculate statistics
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
                
        # Tables (if at least one row exists, count as valid)
        tabs = doc.get("Tables", {})
        for field in requirements["Tables"]:
            val = tabs.get(field)
            if isinstance(val, list) and len(val) > 0:
                stats["Tables"][field] += 1

    # Print results
    def print_section(title, section_key):
        print(f"\n{title.upper()}")
        print("-" * 60)
        for field, count in stats[section_key].items():
            rate = (count / total_docs) * 100
            
            # Visual status indicator (for terminal output)
            if rate >= 90:
                icon = "🟢"
            elif rate >= 50:
                icon = "🟡"
            elif rate > 10:
                icon = "🟠"
            else:
                icon = "🔴"
                
            print(f"{icon} {field:<30}: {count}/{total_docs} filled ({rate:.1f}%)")

    print(f"Total number of documents analyzed: {total_docs}")
    print_section("1. Metadata (Document Fields)", "Metadata")
    print_section("2. Summary Texts", "Summaries")
    print_section("3. Tables", "Tables")
    
    print("\n" + "=" * 60)
    print("💡 NOTE: Some tables (e.g., Equipment Failure) and fields may have low coverage")
    print("which is normal because these events do not occur in every report.")

if __name__ == "__main__":
    # Specify JSON file path here:
    json_file_path = r"D:\drilling-report-nlp\data\processed\document_database.json"
    check_all_requirements(json_file_path)