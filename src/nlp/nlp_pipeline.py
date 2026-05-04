import pandas as pd
import re
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class AdvancedNLPPipeline:
    def __init__(self, data_path, output_dir):
        print("Initializing Advanced NLP Pipeline...")
        self.data_path = data_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load Data
        print("1. Loading Extracted Remarks...")
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.dropna(subset=['Remark']).copy()
        self.df['Remark'] = self.df['Remark'].astype(str)
        
        # Load Sentence-BERT Model
        print("   Loading Sentence-BERT Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define Activity Classes and their descriptions for AI Classification
        self.activity_definitions = {
            'TRIPPING': "pulling out of hole or running in hole with drill pipe or casing",
            'DRILLING': "drilling ahead, making hole, rotary drilling, orienting",
            'CEMENTING': "pumping cement, waiting on cement, curing",
            'PRESSURE_TEST': "pressure testing, bop test, leak off test",
            'FISHING': "fishing operations, catching or retrieving lost tools in hole",
            'MAINTENANCE_REPAIR': "repairing equipment, fixing failure, maintenance downtime",
            'CIRCULATION': "circulating mud, conditioning drilling fluid"
        }
        
        # Pre-compute embeddings for activity definitions
        self.activity_embeddings = {
            act: self.model.encode(desc, convert_to_tensor=True) 
            for act, desc in self.activity_definitions.items()
        }

    def apply_ner(self):
        print("2. Running Advanced NER (Depths, Equipment, Measurements)...")
        
        def extract_entities(text):
            text = text.lower()
            entities = {'Depths': [], 'Measurements': [], 'Time': [], 'Equipment': []}
            
            # Regex for domain-specific metrics
            entities['Depths'] = list(set(re.findall(r'\b\d+(?:\.\d+)?\s*(?:mmd|mtvd|m\b)', text)))
            entities['Measurements'] = list(set(re.findall(r'\b\d+(?:\.\d+)?\s*(?:bar|psi|rpm|lpm|gpm|mt|ton|kg|ppg)\b', text)))
            entities['Time'] = list(set(re.findall(r'\b\d{1,2}:\d{2}\b|\b\d+(?:\.\d+)?\s*(?:hrs|hours|mins)\b', text)))
            
            # Domain Equipment Lexicon
            equipment_list = ['bop', 'tds', 'flx packer', 'xo', 'spear bha', 'whipstock', 'drill pipe', 'casing', 'milling assembly', 'bit', 'mud motor']
            entities['Equipment'] = [eq for eq in equipment_list if eq in text]
            
            return json.dumps(entities) # Store as JSON string for neat CSV saving

        self.df['NER_Entities'] = self.df['Remark'].apply(extract_entities)

    def classify_activity_hybrid(self):
        print("3. Running Hybrid Activity Classification (Rule-based + Sentence-BERT)...")
        
        def hybrid_classifier(text):
            text_lower = text.lower()
            
            # STEP 1: Rule-Based Fast Matching (High Confidence)
            if any(w in text_lower for w in ['rih', 'tih', 'pooh', 'trip', 'slug']): return 'TRIPPING' # 'slug' əlavə edildi
            elif any(w in text_lower for w in ['circulat', 'circ', 'pill', 'hi-vis']): return 'CIRCULATION' # Yeni kateqoriya əlavə edildi
            elif 'cement' in text_lower or 'cmt' in text_lower: return 'CEMENTING'
            elif 'fish' in text_lower: return 'FISHING'
            elif 'bop' in text_lower and 'test' in text_lower: return 'PRESSURE_TEST'
            elif any(w in text_lower for w in ['repair', 'broke', 'failure']): return 'MAINTENANCE_REPAIR'
            
            # STEP 2: Sentence-BERT Semantic Matching (For ambiguous texts)
            text_embedding = self.model.encode(text, convert_to_tensor=True)
            best_activity = "OTHER"
            best_score = 0.0
            
            for activity, act_emb in self.activity_embeddings.items():
                score = float(util.cos_sim(text_embedding, act_emb)[0][0])
                if score > best_score and score > 0.35: # 0.35 is our semantic threshold
                    best_score = score
                    best_activity = activity
                    
            return best_activity

        self.df['Hybrid_Activity'] = self.df['Remark'].apply(hybrid_classifier)

    def extract_tfidf_keywords(self):
        print("4. Extracting TF-IDF Keywords per Report...")
        grouped = self.df.groupby('file_name')['Remark'].apply(lambda x: ' '.join(x)).reset_index()
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(grouped['Remark'])
        feature_names = vectorizer.get_feature_names_out()
        
        keywords_dict = {}
        for idx, row in grouped.iterrows():
            doc_vector = tfidf_matrix[idx].toarray()[0]
            top_indices = doc_vector.argsort()[-5:][::-1]
            keywords_dict[row['file_name']] = ", ".join([feature_names[i] for i in top_indices])
            
        self.df['Top_Keywords_TFIDF'] = self.df['file_name'].map(keywords_dict)

    def generate_report_analysis(self):
        print("5. Generating Final Report Analysis...")
        # Summarize the entire dataset findings
        total_reports = self.df['file_name'].nunique()
        total_operations = len(self.df)
        activity_counts = self.df['Hybrid_Activity'].value_counts().to_dict()
        
        report_text = (
            "=================================================\n"
            "      O&G DAILY REPORTS NLP ANALYSIS SUMMARY     \n"
            "=================================================\n\n"
            f"Total PDF Reports Analyzed : {total_reports}\n"
            f"Total Operations Processed : {total_operations}\n\n"
            "--- ACTIVITY DISTRIBUTION ---\n"
        )
        
        for act, count in activity_counts.items():
            report_text += f"- {act}: {count} operations ({round((count/total_operations)*100, 1)}%)\n"
            
        report_text += "\n--- PIPELINE ARCHITECTURE ---\n"
        report_text += "1. Data Extraction -> Operations Tables\n"
        report_text += "2. NER -> Depths, Measurements, Equipment\n"
        report_text += "3. Hybrid Classification -> Rule-based + Sentence-BERT zero-shot\n"
        report_text += "4. Semantic Matching -> Matching NDS events to reports (Done in Task 2)\n"
        report_text += "5. TF-IDF -> Keywords generation\n"
        
        report_path = os.path.join(self.output_dir, "pipeline_summary_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
            
        print(f"   Analysis Report saved to: {report_path}")

    def run_pipeline(self):
        self.apply_ner()
        self.classify_activity_hybrid()
        self.extract_tfidf_keywords()
        self.generate_report_analysis()
        
        # Save enriched dataset
        output_csv = os.path.join(self.output_dir, "advanced_nlp_enriched_data.csv")
        self.df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"\n✅ PIPELINE COMPLETED! Enriched data saved to: {output_csv}")
        
        print("\n--- PIPELINE PREVIEW (First Row) ---")
        preview = self.df[['file_name', 'Remark', 'NER_Entities', 'Hybrid_Activity']].head(1).to_dict(orient='records')[0]
        for k, v in preview.items():
            print(f"{k}: {v}")

if __name__ == "__main__":
    DATA_PATH = "../../data/processed/operations_table.csv"
    RESULTS_DIR = "../../data/results"
    
    pipeline = AdvancedNLPPipeline(DATA_PATH, RESULTS_DIR)
    pipeline.run_pipeline()
