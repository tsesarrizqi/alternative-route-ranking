import pandas as pd, numpy as np

files = {
    'ksp': {
        '0.01':  'data/london/graph/paths_scored_step2_london_ksp_eps0.01.csv',
        '0.025': 'data/london/graph/paths_scored_step2_london_ksp_eps0.03.csv',
        '0.05':  'data/london/graph/paths_scored_step2_london_ksp_eps0.05.csv',
        '0.10':  'data/london/graph/paths_scored_step2_london_ksp.csv',
        '0.15':  'data/london/graph/paths_scored_step2_london_ksp_eps0.15.csv',
        '0.20':  'data/london/graph/paths_scored_step2_london_ksp_eps0.20.csv',
    },
    'rw': {
        '0.01':  'data/london/graph/paths_scored_step2_london_randomwalk_eps0.01.csv',
        '0.025': 'data/london/graph/paths_scored_step2_london_randomwalk_eps0.03.csv',
        '0.05':  'data/london/graph/paths_scored_step2_london_randomwalk_eps0.05.csv',
        '0.10':  'data/london/graph/paths_scored_step2_london_randomwalk.csv',
        '0.15':  'data/london/graph/paths_scored_step2_london_randomwalk_eps0.15.csv',
        '0.20':  'data/london/graph/paths_scored_step2_london_randomwalk_eps0.20.csv',
    },
}

def parse(s):
    return [float(x) for x in str(s).strip().rstrip(';').split(';') if x.strip()]

def analyze(csv_path):
    df = pd.read_csv(csv_path)
    all_s, pct_adm_per_od, spread_nonadm = [], [], []
    for _, row in df.iterrows():
        sc = parse(row['stretch_scores'])
        if not sc:
            continue
        all_s.extend(sc)
        adm    = [s for s in sc if s >= 99.0]
        nonadm = [s for s in sc if s < 99.0]
        pct_adm_per_od.append(len(adm) / len(sc))
        if len(nonadm) >= 2:
            spread_nonadm.append(max(nonadm) - min(nonadm))
    arr = np.array(all_s)
    nonadm_arr = arr[arr < 99.0]
    return {
        'pct_adm_paths':   np.mean(pct_adm_per_od) * 100,   # avg % admissible paths per OD
        'pct_adm_total':   np.mean(arr >= 99.0) * 100,       # total % of all paths admissible
        'mean_nonadm':     float(np.mean(nonadm_arr)) if len(nonadm_arr) else 0,
        'std_nonadm':      float(np.std(nonadm_arr))  if len(nonadm_arr) else 0,
        'spread_nonadm':   float(np.mean(spread_nonadm)) if spread_nonadm else 0,
        'pct_60_99':       float(np.mean((arr >= 60) & (arr < 99.0)) * 100),  # compressed near-admissible zone
        'pct_below60':     float(np.mean(arr < 60) * 100),
    }

for vname, vfiles in [('KSP', files['ksp']), ('Randomwalk', files['rw'])]:
    print(f'\n=== {vname}: stretch_scores distribution ===')
    print(f'  eps    %OD-adm  %all-adm  mean(non-adm)  std(non-adm)  spread(non-adm)  %in[60-99]  %<60')
    for eps in ['0.01', '0.025', '0.05', '0.10', '0.15', '0.20']:
        s = analyze(vfiles[eps])
        m = ' <-- default' if eps == '0.10' else ''
        print(f'  {eps:<6}  {s["pct_adm_paths"]:6.1f}   {s["pct_adm_total"]:6.1f}     {s["mean_nonadm"]:8.2f}       {s["std_nonadm"]:8.2f}          {s["spread_nonadm"]:8.2f}      {s["pct_60_99"]:6.1f}  {s["pct_below60"]:5.1f}{m}')
