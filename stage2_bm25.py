import pandas as pd
import json
import sys
import re
from rank_bm25 import BM25Okapi

# The core keywords extracted directly from the "Must Haves" in the JD
JD_QUERY = "python embedding embeddings retrieval sentencetransformers openai pinecone qdrant vector database rag ml ai ranking search"

def tokenize(text):
    # Simple, fast alphanumeric tokenization
    return re.findall(r'\b\w+\b', str(text).lower())

def build_candidate_corpus(df):
    print("Extracting and tokenizing candidate documents...")
    corpus_tokens = []
    
    # We use a fast list comprehension to build the text blobs
    for _, row in df.iterrows():
        # Safely extract fields
        profile = row.get('profile', {})
        title = profile.get('current_title', '')
        headline = profile.get('headline', '')
        summary = profile.get('summary', '')
        
        # Extract skill names
        skills_list = row.get('skills', [])
        skills_text = ' '.join([s.get('name', '') for s in skills_list])
        
        # Give extra weight to the current title by duplicating it
        combined_text = f"{title} {title} {headline} {skills_text} {summary}"
        
        corpus_tokens.append(tokenize(combined_text))
        
    return corpus_tokens

def run_bm25_filter(input_path, output_path, top_k=500):
    print(f"Loading {input_path}...")
    df = pd.read_json(input_path, lines=True)
    initial_count = len(df)
    
    # 1. Build the tokenized corpus
    corpus_tokens = build_candidate_corpus(df)
    
    # 2. Initialize the BM25 Index
    print("Building BM25 Index (this takes a few seconds)...")
    bm25 = BM25Okapi(corpus_tokens)
    
    # 3. Tokenize the Target Query
    query_tokens = tokenize(JD_QUERY)
    
    # 4. Score all candidates
    print(f"Scoring {initial_count} candidates against JD query...")
    doc_scores = bm25.get_scores(query_tokens)
    
    # Add scores to the dataframe
    df['bm25_score'] = doc_scores
    
    # 5. Sort and extract the Top K
    df_sorted = df.sort_values(by='bm25_score', ascending=False)
    top_candidates = df_sorted.head(top_k).copy()
    
    print("\n--- STAGE 2 EXECUTION STATS ---")
    print(f"Candidates Evaluated: {initial_count}")
    print(f"Top Score:            {top_candidates['bm25_score'].max():.2f}")
    print(f"Cutoff Score:         {top_candidates['bm25_score'].min():.2f}")
    print(f"Advancing to Stage 3: {len(top_candidates)}")
    print("-------------------------------")
    
    # Drop the temporary score column and export
    top_candidates = top_candidates.drop(columns=['bm25_score'])
    top_candidates.to_json(output_path, orient='records', lines=True)
    print(f"Top {top_k} candidates saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stage2_bm25.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    run_bm25_filter(sys.argv[1], sys.argv[2])
