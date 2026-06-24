import pandas as pd
import sys
import os

def filter_candidates(input_path, output_path, chunk_size=100000):
    print(f"Loading data from {input_path} in chunks of {chunk_size}...")

    if os.path.exists(output_path):
        os.remove(output_path)

    total_loaded = 0
    total_honeypots = 0
    total_service_traps = 0
    total_title_traps = 0
    total_dropped = 0
    total_passed = 0

    for chunk_idx, df in enumerate(pd.read_json(input_path, lines=True, chunksize=chunk_size)):
        chunk_count = len(df)
        total_loaded += chunk_count
        print(f"Processing chunk {chunk_idx + 1} ({total_loaded} candidates so far)...")

        df['yoe'] = [p.get('years_of_experience', 0) for p in df['profile']]

        def get_max_grad(edu_list):
            if isinstance(edu_list, list) and edu_list:
                years = [e.get('end_year') for e in edu_list if e.get('end_year') is not None]
                if years:
                    return max(years)
            return 2026

        df['grad_year'] = [get_max_grad(edu) for edu in df['education']]

        company_lists = [
            [c.get('company', '').lower() for c in (hist if isinstance(hist, list) else [])]
            for hist in df['career_history']
        ]

        timeline_anomaly_mask = df['yoe'] > (2026 - df['grad_year'] + 1)

        service_firms = {"tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini", "deloitte", "hcl"}

        def is_service_only(companies):
            if not companies:
                return False
            return all(any(sf in c for sf in service_firms) for c in companies)

        service_only_mask = [is_service_only(comps) for comps in company_lists]
        service_filter_mask = (df['yoe'] >= 5) & pd.Series(service_only_mask, index=df.index)

        forbidden_titles = {
            "marketing", "sales", "hr", "recruiter", "content writer",
            "operations manager", "accountant", "customer support",
            "business analyst", "mechanical engineer", "civil engineer",
            "graphic designer", "project manager"
        }

        def is_forbidden_role(title):
            title_lower = str(title).lower()
            return any(bad_word in title_lower for bad_word in forbidden_titles)

        title_trap_mask = df['profile'].apply(lambda p: is_forbidden_role(p.get('current_title', '')))

        bad_rows = timeline_anomaly_mask | service_filter_mask | title_trap_mask
        clean_df = df[~bad_rows].copy()

        total_honeypots += timeline_anomaly_mask.sum()
        total_service_traps += service_filter_mask.sum()
        total_title_traps += title_trap_mask.sum()
        total_dropped += bad_rows.sum()
        total_passed += len(clean_df)

        clean_df = clean_df.drop(columns=['yoe', 'grad_year'])
        
        clean_df.to_json(output_path, orient='records', lines=True, mode='a')

    print("\n--- STAGE 1 EXECUTION STATS ---")
    print(f"Total Candidates Loaded:  {total_loaded}")
    print(f"Honeypots Caught:         {total_honeypots}")
    print(f"Service Traps Caught:     {total_service_traps}")
    print(f"Title Traps Caught:       {total_title_traps}")
    print(f"Total Dropped:            {total_dropped}")
    print(f"Remaining for Stage 2:    {total_passed}")
    print("-------------------------------")
    print(f"Filtered dataset saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stage1_filter.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    filter_candidates(sys.argv[1], sys.argv[2])
