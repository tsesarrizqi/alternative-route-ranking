import json, os, numpy as np

cities = ['london','manhattan','paris','singapore','tokyo',
          'amsterdam','bangalore','manchester','rome','seattle','sydney']

variants = [
    ('ksp',        '_ksp'),
    ('plateau',    '_plateau_ub_2_dm_4_sim_07'),
    ('randomwalk', '_randomwalk'),
    ('stochastic', '_stochastic_mu15'),
]
conds = [
    ('oracle_filter',  'Oracle + Filter'),
    ('model_filter',   'Model + Filter'),
    ('gen_filter',     'Generator Order + Filter'),
    ('model_nofilter', 'Model, No Filter'),
    ('gen_nofilter',   'Generator Order, No Filter'),
]
metrics = ['avg_stretch_mean', 'pairwise_jaccard_mean', 'admissibility_rate_mean', 'diversity_compliance_mean']
mlabels = ['str', 'jac', 'adm', 'dc']

for top_k in [3]:
    print(f'top_k={top_k}')
    for vname, vsuffix in variants:
        print(f'\n=== {vname} ===')
        header = f"  {'Condition':<38}" + ''.join(f'  {ml}_mean  {ml}_std' for ml in mlabels)
        print(header)
        for ckey, clabel in conds:
            vals = {m: [] for m in metrics}
            for city in cities:
                path = f'data/{city}/results/endtoend_{city}{vsuffix}_top{top_k}.json'
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    d = json.load(f)
                s = d['summary'].get(ckey, {})
                if s:
                    for m in metrics:
                        v = s.get(m)
                        if v is not None:
                            vals[m].append(v)
            row = f'  {clabel:<38}'
            for m in metrics:
                if vals[m]:
                    row += f'  {np.mean(vals[m]):8.4f}  {np.std(vals[m]):7.4f}'
                else:
                    row += f'  {"N/A":>8}  {"N/A":>7}'
            print(row)
