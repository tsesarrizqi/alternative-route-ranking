import pandas as pd, numpy as np

files = {
    'ksp': {
        '0.01':  'data/london/graph/paths_scored_step2_london_ksp_eps0.01.csv',
        '0.05':  'data/london/graph/paths_scored_step2_london_ksp_eps0.05.csv',
        '0.10':  'data/london/graph/paths_scored_step2_london_ksp.csv',
        '0.20':  'data/london/graph/paths_scored_step2_london_ksp_eps0.20.csv',
    },
    'plateau': {
        '0.01':  'data/london/graph/paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.01.csv',
        '0.05':  'data/london/graph/paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.05.csv',
        '0.10':  'data/london/graph/paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07.csv',
        '0.20':  'data/london/graph/paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.20.csv',
    },
    'rw': {
        '0.01':  'data/london/graph/paths_scored_step2_london_randomwalk_eps0.01.csv',
        '0.05':  'data/london/graph/paths_scored_step2_london_randomwalk_eps0.05.csv',
        '0.10':  'data/london/graph/paths_scored_step2_london_randomwalk.csv',
        '0.20':  'data/london/graph/paths_scored_step2_london_randomwalk_eps0.20.csv',
    },
    'stochastic': {
        '0.01':  'data/london/graph/paths_scored_step2_london_stochastic_mu15_eps0.01.csv',
        '0.05':  'data/london/graph/paths_scored_step2_london_stochastic_mu15_eps0.05.csv',
        '0.10':  'data/london/graph/paths_scored_step2_london_stochastic_mu15.csv',
        '0.20':  'data/london/graph/paths_scored_step2_london_stochastic_mu15_eps0.20.csv',
    },
}

def parse(s):
    return [float(x) for x in str(s).strip().rstrip(';').split(';') if x.strip()]

def near_boundary_analysis(csv_path, eps_val):
    """
    Key question: at each OD, how many paths are near the admissibility boundary
    (i.e., within a narrow band just below 99), vs clearly non-admissible?
    These near-boundary paths are not ties but have very similar scores -> hard to rank.
    """
    df = pd.read_csv(csv_path)
    
    n_paths_total = 0
    n_adm = 0           # score >= 99
    n_near_boundary = 0 # score in [85, 99) -- close enough that ranking is ambiguous
    n_clearly_nonadm = 0 # score < 85 -- clearly non-admissible, easier to rank last

    # Per-OD: how many ODs have mixed (some adm, some near-boundary) -- hardest case
    hard_ods = 0
    total_ods = 0
    
    # Score gap between best non-adm and worst adm (or between top two non-adm)
    intra_nonadm_gaps = []  # gap between rank-1 and rank-2 non-admissible per OD
    
    for _, row in df.iterrows():
        sc = sorted(parse(row['stretch_scores']), reverse=True)
        if not sc:
            continue
        total_ods += 1
        n_paths_total += len(sc)
        
        adm = [s for s in sc if s >= 99.0]
        near = [s for s in sc if 85.0 <= s < 99.0]
        clearly = [s for s in sc if s < 85.0]
        
        n_adm += len(adm)
        n_near_boundary += len(near)
        n_clearly_nonadm += len(clearly)
        
        # Hard OD: has admissible paths AND near-boundary paths that are NOT ties
        if len(adm) > 0 and len(near) > 0:
            hard_ods += 1
        
        # Gap between top-2 non-admissible paths
        nonadm = sorted([s for s in sc if s < 99.0], reverse=True)
        if len(nonadm) >= 2:
            intra_nonadm_gaps.append(nonadm[0] - nonadm[1])
    
    return {
        'pct_adm':          n_adm / n_paths_total * 100,
        'pct_near_boundary': n_near_boundary / n_paths_total * 100,
        'pct_clearly_nonadm': n_clearly_nonadm / n_paths_total * 100,
        'pct_hard_ods':     hard_ods / total_ods * 100,
        'mean_top2_gap':    np.mean(intra_nonadm_gaps) if intra_nonadm_gaps else 0,
        'median_top2_gap':  np.median(intra_nonadm_gaps) if intra_nonadm_gaps else 0,
        'pct_gap_lt1':      np.mean(np.array(intra_nonadm_gaps) < 1.0) * 100 if intra_nonadm_gaps else 0,
        'pct_gap_lt5':      np.mean(np.array(intra_nonadm_gaps) < 5.0) * 100 if intra_nonadm_gaps else 0,
    }

print("Near-boundary analysis: score >= 99 = admissible, [85,99) = near-boundary, <85 = clearly non-admissible")
print("'hard ODs' = ODs with both admissible and near-boundary paths (model must discriminate fine differences)")
print("'top-2 gap' = score gap between rank-1 and rank-2 non-admissible paths (small = hard to rank)")
print()

for vname, vfiles in [('KSP', files['ksp']), ('Plateau', files['plateau']), ('Randomwalk', files['rw']), ('Stochastic', files['stochastic'])]:
    print(f'\n=== {vname} ===')
    print(f'  eps    %adm  %near-boundary  %clearly-nonadm  %hard-ODs  top2-gap(mean)  top2-gap(med)  %gap<1  %gap<5')
    for eps, path in sorted(vfiles.items()):
        s = near_boundary_analysis(path, float(eps))
        m = ' <--' if eps == '0.10' else ''
        print(f'  {eps:<6} {s["pct_adm"]:5.1f}   {s["pct_near_boundary"]:8.1f}        {s["pct_clearly_nonadm"]:8.1f}        {s["pct_hard_ods"]:6.1f}      {s["mean_top2_gap"]:8.3f}         {s["median_top2_gap"]:7.3f}    {s["pct_gap_lt1"]:5.1f}  {s["pct_gap_lt5"]:5.1f}{m}')
