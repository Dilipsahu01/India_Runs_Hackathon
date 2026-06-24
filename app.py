import streamlit as st
import json
import os
import pandas as pd
from stage1_filter import filter_candidates
from stage2_bm25 import run_bm25_filter
from stage3_semantic_ranker import run_semantic_ranking

st.title("Redrob Hackathon: Intelligent Candidate Ranker")
st.markdown("### 3-Stage Pipeline: Pandas Filter → BM25 Lexical → Semantic Vector Math")

if st.button("Run Ranker on Sample Data", type="primary"):
    with st.spinner("Processing..."):
        input_file = "sample_candidates.json"
        jsonl_file = "sample_candidates.jsonl"
        
        # Convert JSON array to JSON Lines for our pipeline
        if os.path.exists(input_file):
            with open(input_file, "r") as f:
                data = json.load(f)
            with open(jsonl_file, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
        else:
            st.error(f"{input_file} not found! Please upload it.")
            st.stop()
            
        stage1_out = "stage1_out.jsonl"
        stage2_out = "stage2_out.jsonl"
        final_csv = "final_output.csv"
        
        try:
            st.info("Stage 1: Filtering timeline anomalies and service firm traps...")
            filter_candidates(jsonl_file, stage1_out)
            
            st.info("Stage 2: BM25 Lexical Scoring to extract top subsets...")
            # We use top_k=20 for the sandbox since the sample is only 50 candidates
            run_bm25_filter(stage1_out, stage2_out, top_k=20) 
            
            st.info("Stage 3: MiniLM Semantic + Behavioral final scoring...")
            run_semantic_ranking(stage2_out, final_csv)
            
            st.success("Pipeline Execution Complete!")
            
            df = pd.read_csv(final_csv)
            st.dataframe(df)
            
            with open(final_csv, "rb") as f:
                st.download_button(
                    label="Download Final CSV",
                    data=f,
                    file_name="team_dilip.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Pipeline failed: {e}")
