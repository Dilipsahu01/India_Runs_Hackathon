import pandas as pd
import sys

def filter_candidates(input_path, output_path):
    print(f"Loading data from {input_path}...")

    df = pd.read_json(input_path, lines=True)
    initial_count = len(df)

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
    service_filter_mask = (df['yoe'] >= 5) & pd.Series(service_only_mask)

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

    print("\n--- STAGE 1 EXECUTION STATS ---")
    print(f"Total Candidates Loaded:  {initial_count}")
    print(f"Honeypots Caught:         {timeline_anomaly_mask.sum()}")
    print(f"Service Traps Caught:     {service_filter_mask.sum()}")
    print(f"Title Traps Caught:       {title_trap_mask.sum()}")
    print(f"Total Dropped:            {bad_rows.sum()}")
    print(f"Remaining for Stage 2:    {len(clean_df)}")
    print("-------------------------------")

    clean_df = clean_df.drop(columns=['yoe', 'grad_year'])
    clean_df.to_json(output_path, orient='records', lines=True)
    print(f"Filtered dataset saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stage1_filter.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    filter_candidates(sys.argv[1], sys.argv[2])
