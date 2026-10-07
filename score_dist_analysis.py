import pandas as pd
import numpy as np
import ast

city = 'london'

# file map: (variant, eps_label) -> csv path
files = {
    # KSP
    ('ksp', '0.01'):  'data/london/graph/paths_scored_step2_london_ksp_eps0.01.csv',
    ('ksp', '0.025'): 'data/london/graph/paths_scored_step2_london_ksp_eps0.03.csv',  # naming quirk
    ('ksp', '0.05'):  'data/london/graph/paths_scored_step2_london_ksp_eps0.05.csv',
    ('ksp', '0.10'):  'data/london/graph/paths_scored_step2_london_ksp.csv',
    ('ksp', '0.15'):  'data/london/graph/paths_scored_step2_london_ksp_eps0.15.csv',
    ('ksp', '0.20'):  'data/london/graph/paths_scored_step2_london_ksp_eps0.20.csv',
    # Randomwalk
    ('rw', '0.01'):   'data/london/graph/paths_scored_step2_london_randomwalk_eps0.01.csv',
    ('rw', '0.025'):  'data/london/graph/paths_scored_step2_london_randomwalk_eps0.03.csv',
    ('rw', '0.05'):   'data/london/graph/paths_scored_step2_london_randomwalk_eps0.05.csv',
    ('rw', '0.10'):   'data/london/graph/paths_scored_step2_london_randomwalk.csv',
    ('rw', '0.15'):   'data/london/graph/paths_scored_step2_london_randomwalk_eps0.15.csv',
    ('rw', '0.20'):   'data/london/graph/paths_scored_step2_london_randomwalk_eps0.20.csv',
}

def parse_scores(s):
    return [float(x) for x in str(s).strip().rstrip(';').split(';') if x.strip()]

def score_stats(csv_path):
    df = pd.read_csv(csv_path)
    all_scores = []
    pct_at_100 = []   # fraction of paths per OD with score >= 99 (i.e., admissible)
    score_ranges = []  # max - min per OD group (spread among non-admissible)

    for _, row in df.iterrows():
        scores = parse_scores(row['final_scores'])
        if not scores:
            continue
        all_scores.extend(scores)
        admissible = [s for s in scores if s >= 99.0]
        non_admissible = [s for s in scores if s < 99.0]
        pct_at_100.append(len(admissible) / len(scores))
        if len(non_admissible) >= 2:
            score_ranges.append(max(non_admissible) - min(non_admissible))

    arr = np.array(all_scores)
    return {
        'mean':       np.mean(arr),
        'std':        np.std(arr),
        'pct_ge99':   np.mean(pct_at_100) * 100,   # avg % of paths scoring >=99 per OD
        'non_adm_spread_mean': np.mean(score_ranges) if score_ranges else 0,
        'non_adm_spread_med':  np.median(score_ranges) if score_ranges else 0,
        'pct_below50': np.mean(arr < 50) * 100,
    }

for variant in ['ksp', 'rw']:
    vname = 'KSP' if variant == 'ksp' else 'Randomwalk'
    print(f'\n=== {vname} ===')
    print(f'  eps    mean   std    %>=99  non-adm-spread(mean) non-adm-spread(med)  %<50')
    for eps in ['0.01', '0.025', '0.05', '0.10', '0.15', '0.20']:
        key = (variant, eps)
        if key not in files:
            continue
        s = score_stats(files[key])
        marker = ' *' if eps == '0.10' else ''
        print(f'  {eps:<6} {s["mean"]:5.1f}  {s["std"]:5.1f}  {s["pct_ge99"]:5.1f}  {s["non_adm_spread_mean"]:8.2f}            {s["non_adm_spread_med"]:8.2f}     {s["pct_below50"]:5.1f}{marker}')
