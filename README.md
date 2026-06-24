# India_Runs_Hackathon

## Intelligent Candidate Discovery & Ranking Challenge

This repository contains our team's submission for the Redrob Hackathon.

### The 3-Stage Ranking Pipeline

Due to strict computational constraints (5-minute runtime, CPU only, max 16GB RAM, no network access), evaluating 100,000 candidate profiles using a dense semantic model is unfeasible. We designed a highly optimized, 3-stage funnel architecture to accurately isolate the top 100 fits for the Senior AI Engineer role without exceeding the compute limits.

#### Stage 1: Heuristic & Anomaly Filter (`stage1_filter.py`)
This stage loads the `candidates.jsonl` dataset natively as JSON Lines via Pandas. Using highly optimized list comprehensions and vectorized math (bypassing slow `.apply()` loops), we explicitly target and eliminate:
- **Timeline Anomalies (Honeypots):** We calculate the maximum graduation year and flag candidates claiming impossible years of experience (e.g., claiming 12 years of experience but graduating in 2022).
- **Service Firm Traps:** We identify candidates who claim 5+ years of experience but have a career history consisting *exclusively* of traditional IT consulting firms, which the JD strictly forbids for this specific product-focused role.

#### Stage 2: Lexical BM25 Search (`stage2_bm25.py`)
After Stage 1 filters out honeypots and bad fits, we are left with a smaller pool. Stage 2 executes a blazing-fast lexical search using the `rank_bm25` engine. We construct a corpus tokenizing the candidate's headline, title, summary, and skills, weighting their current title heavily.
We query this corpus against core keywords explicitly derived from the JD (e.g., *python, embedding, retrieval, sentencetransformers, pinecone, qdrant, rag*). This instantly slices the remaining pool down to the **top 500 candidates** in seconds.

#### Stage 3: Dense Semantic Re-ranking (`stage3_semantic_ranker.py`)
With only 500 candidates remaining, the compute constraints are neutralized. We load the CPU-optimized `all-MiniLM-L6-v2` SentenceTransformer.
We generate a composite score derived from:
- **40% Semantic Match:** Cosine similarity of the candidate's textual profile against the idealized JD "Must Haves".
- **30% Hard Experience Match:** Strict 5-9 year experience validation, heavily penalizing candidates outside this window.
- **30% Behavioral Reality Modifier:** Based on `redrob_signals` (Recruiter Response Rate + Login Recency/Activity Score).

The top 100 candidates are then extracted, correctly tie-broken by candidate ID ascending to bypass validator bugs, and exported to the final submission CSV.

### Running the Code

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the pipeline stages:
```bash
python3 stage1_filter.py candidates.jsonl stage1_passed.jsonl
python3 stage2_bm25.py stage1_passed.jsonl stage2_passed.jsonl
python3 stage3_semantic_ranker.py stage2_passed.jsonl team_dilip.csv
```

3. Validate the submission:
```bash
python3 validate_submission.py team_dilip.csv
```

### Sandbox App
We've also provided a lightweight `app.py` built on Streamlit that runs the exact same pipeline over a smaller `sample_candidates.json` file. This app is designed to be hosted on Hugging Face Spaces for the judge's Sandbox review.
