import pandas as pd
import sys
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

PROFICIENCY_WEIGHTS = {
    "expert": 1.0,
    "advanced": 0.75,
    "intermediate": 0.5,
    "beginner": 0.25
}

JD_CORE_SKILLS_EXACT = {
    "python", "pytorch", "tensorflow", "keras", "embeddings",
    "sentencetransformers", "sentence-transformers", "sentence transformers",
    "openai", "pinecone", "qdrant", "chromadb", "faiss", "weaviate",
    "langchain", "llamaindex", "llama index",
    "kubernetes", "docker", "mlops",
    "nlp", "rag", "llm", "bert", "gpt",
}

JD_CORE_SKILLS_SUBSTRING = {
    "vector database", "machine learning", "deep learning",
    "neural network", "natural language processing",
    "retrieval augmented", "retrieval-augmented",
    "large language model", "transformer", "embedding",
    "chatgpt", "gpt-", "distilbert", "roberta",
}

def is_jd_relevant_skill(skill_name_lower):
    if skill_name_lower in JD_CORE_SKILLS_EXACT:
        return True
    for phrase in JD_CORE_SKILLS_SUBSTRING:
        if phrase in skill_name_lower:
            return True
    return False


def calculate_skill_quality_score(skills_list):
    if not skills_list:
        return 0.0

    total_weighted = 0.0
    jd_match_count = 0

    for skill in skills_list:
        name = skill.get('name', '').lower().strip()
        prof = skill.get('proficiency', 'beginner')
        endorsements = skill.get('endorsements', 0)

        weight = PROFICIENCY_WEIGHTS.get(prof, 0.25)

        if is_jd_relevant_skill(name):
            jd_match_count += 1
            weight *= 1.5

        endorsement_bonus = min(endorsements / 50.0, 0.3)
        total_weighted += weight + endorsement_bonus

    max_expected = 20 * 1.5 * 1.0 + 20 * 0.3
    normalized = min(total_weighted / max_expected, 1.0)
    coverage = min(jd_match_count / 10.0, 1.0)

    return (normalized * 0.6) + (coverage * 0.4)


def calculate_experience_score(yoe):
    if 5 <= yoe <= 9:
        return 1.0
    elif 4 <= yoe < 5:
        return 0.7 + (yoe - 4) * 0.3
    elif 9 < yoe <= 10:
        return 0.7 + (10 - yoe) * 0.3
    else:
        diff = min(abs(yoe - 5), abs(yoe - 9))
        return max(0.0, 0.5 - (diff * 0.2))


def calculate_github_score(signals):
    github = signals.get('github_activity_score', -1)
    if github is None or github < 0:
        return 0.3
    return min(github / 100.0, 1.0)


def calculate_assessment_score(signals):
    assessments = signals.get('skill_assessment_scores', {})
    if not assessments or not isinstance(assessments, dict):
        return 0.5

    scores = [v for v in assessments.values() if isinstance(v, (int, float))]
    if not scores:
        return 0.5

    avg_score = sum(scores) / len(scores)
    breadth_bonus = min(len(scores) / 5.0, 1.0) * 0.2

    return min((avg_score / 100.0) * 0.8 + breadth_bonus, 1.0)


def calculate_signal_score(signals):
    rr = signals.get('recruiter_response_rate', 0.5)

    last_active = signals.get('last_active_date', '2025-01-01')
    try:
        days_since = (datetime(2026, 6, 24) - datetime.strptime(last_active, "%Y-%m-%d")).days
    except:
        days_since = 180

    active_score = max(0.0, 1.0 - (days_since / 180.0))
    github_score = calculate_github_score(signals)
    assessment_score = calculate_assessment_score(signals)

    return (rr * 0.35) + (active_score * 0.25) + (github_score * 0.20) + (assessment_score * 0.20)


def generate_reasoning(row):
    title = row['profile'].get('current_title', 'Engineer')
    yoe = row['profile'].get('years_of_experience', 0)
    skills_list = row.get('skills', [])
    skills_count = len(skills_list)
    rr = row.get('redrob_signals', {}).get('recruiter_response_rate', 0.0)
    github = row.get('redrob_signals', {}).get('github_activity_score', -1)

    top_skills = []
    for s in skills_list:
        name = s.get('name', '')
        if is_jd_relevant_skill(name.lower().strip()):
            top_skills.append(name)

    top_skills_str = '; '.join(top_skills[:3]) if top_skills else 'general ML stack'
    github_str = f"GitHub {github}/100" if github is not None and github >= 0 else "no GitHub linked"

    return (f"{title} with {yoe:.1f} yrs; "
            f"{skills_count} skills ({len(top_skills)} JD-relevant incl. {top_skills_str}); "
            f"response rate {rr:.2f}; {github_str}.")


def build_candidate_text(row):
    title = row['profile'].get('current_title', '')
    summary = row['profile'].get('summary', '')
    headline = row['profile'].get('headline', '')

    skills = ' '.join([s.get('name', '') for s in row.get('skills', [])])

    career_snippet = ''
    career_history = row.get('career_history', [])
    if isinstance(career_history, list):
        for job in career_history:
            if job.get('is_current', False):
                desc = job.get('description', '')
                career_snippet = desc[:200]
                break
        if not career_snippet and career_history:
            desc = career_history[0].get('description', '')
            career_snippet = desc[:200]

    summary_short = summary[:300]

    return f"{title}. {skills}. {headline}. {summary_short}. {career_snippet}"


def run_semantic_ranking(input_path, output_csv):
    print(f"Loading top 500 candidates from {input_path}...")
    df = pd.read_json(input_path, lines=True)

    df['yoe'] = [p.get('years_of_experience', 0) for p in df['profile']]
    df['exp_score'] = df['yoe'].apply(calculate_experience_score)
    df['signal_score'] = [calculate_signal_score(s) for s in df['redrob_signals']]
    df['skill_quality'] = [calculate_skill_quality_score(s) for s in df['skills']]

    print("Loading SentenceTransformer model (all-MiniLM-L6-v2) on CPU...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    jd_text = ("Senior AI Engineer with strong Python. Production experience deploying "
               "embedding-based retrieval systems, SentenceTransformers, OpenAI embeddings, "
               "and vector databases like Pinecone and Qdrant.")

    print("Encoding JD and generating candidate embeddings...")
    jd_embedding = model.encode([jd_text])

    cand_texts = [build_candidate_text(row) for _, row in df.iterrows()]
    cand_embeddings = model.encode(cand_texts)

    print("Calculating cosine similarity matrix...")
    sim_scores = cosine_similarity(cand_embeddings, jd_embedding).flatten()

    df['semantic_score'] = (sim_scores - sim_scores.min()) / (sim_scores.max() - sim_scores.min() + 1e-9)

    print("Calculating final composite scores...")
    df['score'] = (
        0.30 * df['semantic_score'] +
        0.25 * df['exp_score'] +
        0.25 * df['signal_score'] +
        0.20 * df['skill_quality']
    )

    df['score'] = df['score'].round(4)

    top_100 = df.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).head(100).copy()
    top_100['rank'] = range(1, len(top_100) + 1)

    top_100['reasoning'] = top_100.apply(generate_reasoning, axis=1)

    submission_df = top_100[['candidate_id', 'rank', 'score', 'reasoning']]

    print("\n--- STAGE 3 EXECUTION STATS ---")
    print(f"Total Ranked:    {len(submission_df)}")
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
