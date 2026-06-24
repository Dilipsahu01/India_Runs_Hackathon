import pandas as pd
import json
import sys
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_experience_score(yoe):
    # JD strictly requests 5-9 years.
    # We assign 1.0 if within range, and strictly penalize outside of it.
    if 5 <= yoe <= 9:
        return 1.0
    diff = min(abs(yoe - 5), abs(yoe - 9))
    return max(0.0, 1.0 - (diff * 0.15))

def calculate_signal_score(signals):
    # Extracts the behavioral metrics requested by the hackathon rules
    rr = signals.get('recruiter_response_rate', 0.5)
    
    # Calculate recency of login (Current date for hackathon context is mid-2026)
    last_active = signals.get('last_active_date', '2025-01-01')
    try:
        days_since = (datetime(2026, 6, 24) - datetime.strptime(last_active, "%Y-%m-%d")).days
    except:
        days_since = 180
        
    active_score = max(0.0, 1.0 - (days_since / 180.0)) # Drops to 0 if inactive for 6 months
    
    return (rr * 0.6) + (active_score * 0.4)

def generate_reasoning(row):
    # Deterministic 1-2 sentence generation to satisfy CSV requirements
    # Modeled exactly after sample_submission.csv format
    title = row['profile'].get('current_title', 'Engineer')
    yoe = row['profile'].get('years_of_experience', 0)
    skills_count = len(row.get('skills', []))
    rr = row.get('redrob_signals', {}).get('recruiter_response_rate', 0.0)
    
    return f"{title} with {yoe:.1f} yrs; {skills_count} skills listed; response rate {rr:.2f}."

def run_semantic_ranking(input_path, output_csv):
    print(f"Loading top 500 candidates from {input_path}...")
    df = pd.read_json(input_path, lines=True)
    
    # Extract structural data
    df['yoe'] = [p.get('years_of_experience', 0) for p in df['profile']]
    df['exp_score'] = df['yoe'].apply(calculate_experience_score)
    df['signal_score'] = [calculate_signal_score(s) for s in df['redrob_signals']]
    
    print("Loading SentenceTransformer model (all-MiniLM-L6-v2) on CPU...")
    # This model is ~80MB and runs blazing fast on standard CPUs
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # The idealized core JD representation to measure against
    jd_text = ("Senior AI Engineer with strong Python. Production experience deploying "
               "embedding-based retrieval systems, SentenceTransformers, OpenAI embeddings, "
               "and vector databases like Pinecone and Qdrant.")
    
    print("Encoding JD and generating candidate embeddings...")
    jd_embedding = model.encode([jd_text])
    
    # Build text representation for the model
    cand_texts = []
    for _, row in df.iterrows():
        title = row['profile'].get('current_title', '')
        summary = row['profile'].get('summary', '')
        skills = ' '.join([s.get('name', '') for s in row.get('skills', [])])
        cand_texts.append(f"{title}. {skills}. {summary}")
        
    cand_embeddings = model.encode(cand_texts)
    
    print("Calculating cosine similarity matrix...")
    sim_scores = cosine_similarity(cand_embeddings, jd_embedding).flatten()
    
    # Normalize semantic scores between 0 and 1
    df['semantic_score'] = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min() + 1e-9)
    
    # Compute Final Composite Score
    # 40% Semantic Vector Math, 30% Hard Experience Range, 30% Behavioral Reality
    print("Calculating final composite scores...")
    df['score'] = (0.4 * df['semantic_score']) + (0.3 * df['exp_score']) + (0.3 * df['signal_score'])
    
    # Ensure score is rounded safely so floating point errors don't trigger validator sorting bugs
    # We round BEFORE sorting, so that tie-breaks are calculated on the visible scores.
    df['score'] = df['score'].round(4)
    
    # Sort by score (descending) and then candidate_id (ascending) to resolve tie-breaks
    top_100 = df.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).head(100).copy()
    top_100['rank'] = range(1, len(top_100) + 1)
    
    # Generate the reasoning column
    top_100['reasoning'] = top_100.apply(generate_reasoning, axis=1)
    
    # Format to strictly match sample_submission.csv
    submission_df = top_100[['candidate_id', 'rank', 'score', 'reasoning']]
    
    print("\n--- STAGE 3 EXECUTION STATS ---")
    print(f"Total Ranked:    100")
    print(f"Top Score:       {submission_df['score'].max():.4f}")
    print(f"Rank 100 Score:  {submission_df['score'].min():.4f}")
    print("-------------------------------")
    
    submission_df.to_csv(output_csv, index=False)
    print(f"Top 100 finalized and saved to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stage3_semantic_ranker.py <input.jsonl> <submission.csv>")
        sys.exit(1)
    run_semantic_ranking(sys.argv[1], sys.argv[2])
