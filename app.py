import gradio as gr
import json
import os
import pandas as pd
from stage1_filter import filter_candidates
from stage2_bm25 import run_bm25_filter
from stage3_semantic_ranker import run_semantic_ranking

def run_pipeline():
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
        return f"Error: {input_file} not found! Please ensure it is uploaded.", None
        
    stage1_out = "stage1_out.jsonl"
    stage2_out = "stage2_out.jsonl"
    final_csv = "final_output.csv"
    
    try:
        filter_candidates(jsonl_file, stage1_out)
        # We use top_k=20 for the sandbox since the sample is only 50 candidates
        run_bm25_filter(stage1_out, stage2_out, top_k=20) 
        run_semantic_ranking(stage2_out, final_csv)
        
        return "Pipeline Execution Complete! Download your final CSV below.", final_csv
            
    except Exception as e:
        return f"Pipeline failed: {str(e)}", None

with gr.Blocks(title="Redrob Hackathon: Intelligent Candidate Ranker") as demo:
    gr.Markdown("# 🚀 Intelligent Candidate Ranker Sandbox")
    gr.Markdown("### 3-Stage Pipeline: Pandas Filter → BM25 Lexical → Semantic Vector Math")
    gr.Markdown("Click the button below to process the 50-candidate sample dataset through the end-to-end pipeline.")
    
    with gr.Row():
        run_btn = gr.Button("Run Ranker on Sample Data", variant="primary")
    
    with gr.Row():
        status_text = gr.Textbox(label="Execution Status", lines=2)
        download_btn = gr.File(label="Download Final CSV (Top Candidates)")
        
    run_btn.click(fn=run_pipeline, inputs=[], outputs=[status_text, download_btn])

if __name__ == "__main__":
    demo.launch()
