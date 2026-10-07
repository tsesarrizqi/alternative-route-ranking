import json, glob, os

def show_condition(top_k, cond_key, cond_label):
    files = sorted(glob.glob(f'data/*/results/endtoend_*_top{top_k}.json'))
    print(f'\n=== {cond_label} (top_k={top_k}) ===')
    print(f'{"City/Generator":<38} {"AvgStr":>7} {"Jaccard":>7} {"AdmRate":>7} {"DivComp":>7}')
    print('-' * 70)
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
        name = os.path.basename(f).replace('endtoend_','').replace(f'_top{top_k}.json','')
        s = d['summary'].get(cond_key)
        if s:
            ar = s.get('admissibility_rate_mean', float('nan'))
            dc = s.get('diversity_compliance_mean', float('nan'))
            print(f'{name:<38} {s["avg_stretch_mean"]:>7.2f} {s["pairwise_jaccard_mean"]:>7.4f} {ar:>7.4f} {dc:>7.4f}')
        else:
            print(f'{name:<38} [key "{cond_key}" not found; available: {list(d["summary"].keys())}]')

conds = [
    ('model_filter', 'Model + Filter (Ours)'),
    ('gen_filter', 'Generator Order + Filter'),
    ('model_nofilter', 'Model, No Filter'),
    ('gen_nofilter', 'Generator Order, No Filter'),
    ('oracle_filter', 'Oracle + Filter'),
]

for top_k in [3, 5]:
    for cond_key, cond_label in conds:
        show_condition(top_k, cond_key, cond_label)
