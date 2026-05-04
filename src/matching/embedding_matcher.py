import pandas as pd
import os
from sentence_transformers import SentenceTransformer, util
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def match_nds_events():
    print("Loading data...")
    ops_path = "../../data/processed/operations_table.csv"
    nds_path = "../../data/raw/nds_events.xlsx"
    results_dir = "../../data/results"
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    df_ops = pd.read_csv(ops_path)
    df_nds = pd.read_excel(nds_path)

    # Drop rows where 'Remark' is NaN and ensure they are strings
    df_ops = df_ops.dropna(subset=['Remark'])
    df_ops['Remark'] = df_ops['Remark'].astype(str)

    # Load a small and fast BERT model from Hugging Face (downloads ~80MB on first run)
    print("Loading AI Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    results = []

    for index, row in df_nds.iterrows():
        well_raw = row['Well']
        event_text = row['Event']
        
        # Convert well name to match PDF filename format: "15/9-F-10" -> "15_9_F_10"
        well_prefix = str(well_raw).replace('/', '_').replace('-', '_')
        
        print(f"\nSearching for: Well '{well_raw}' | Event: '{event_text[:50]}...'")
        
        # Filter operations to include only PDFs belonging to this specific well
        well_ops = df_ops[df_ops['file_name'].str.startswith(well_prefix, na=False)].copy()
        
        if well_ops.empty:
            print(f"  WARNING: No PDFs found for well {well_raw}. Searching across all files...")
            well_ops = df_ops  # Fallback: search across all files if specific well is not found

        # Encode the NDS event text into a vector embedding
        event_embedding = model.encode(event_text, convert_to_tensor=True)
        
        # Encode all remarks for the corresponding well into vector embeddings
        remarks = well_ops['Remark'].tolist()
        remark_embeddings = model.encode(remarks, convert_to_tensor=True)
        
        # Calculate cosine similarity between the event and all remarks
        cosine_scores = util.cos_sim(event_embedding, remark_embeddings)[0]
        
        # Find the highest similarity score (Top 1 Match)
        best_idx = int(cosine_scores.argmax())
        best_score = float(cosine_scores[best_idx])
        best_remark = remarks[best_idx]
        best_file = well_ops.iloc[best_idx]['file_name']
        
        print(f"  ✅ Match found: {best_file} (Similarity Score: {best_score:.4f})")
        
        # Append the best match to our results list
        results.append({
            'Well': well_raw,
            'NDS_Event': event_text,
            'Matched_File': best_file,
            'Similarity_Score': round(best_score, 4),
            'Matched_Remark': best_remark
        })

    # Convert results into a DataFrame and save as CSV
    df_results = pd.DataFrame(results)
    output_file = os.path.join(results_dir, "event_matching_results.csv")
    df_results.to_csv(output_file, index=False, encoding='utf-8')
    
    print(f"\n✅ All events matched successfully! Results saved to: {output_file}")
    
    print("\n--- FINAL RESULTS (TOP 2) ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df_results[['Well', 'Matched_File', 'Similarity_Score', 'Matched_Remark']].head(2))

if __name__ == "__main__":
    match_nds_events()
