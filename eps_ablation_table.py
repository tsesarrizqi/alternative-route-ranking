import json, os

city = 'london'
variants = [
    ('KSP',        '_london_ksp'),
    ('Plateau',    '_london_plateau_ub_2_dm_4_sim_07'),
    ('Randomwalk', '_london_randomwalk'),
    ('Stochastic', '_london_stochastic_mu15'),
]
eps_list = [
    ('0.01',  '_eps0.01'),
    ('0.025', '_eps0.025'),
    ('0.05',  '_eps0.05'),
    ('0.10',  ''),
    ('0.15',  '_eps0.15'),
    ('0.20',  '_eps0.20'),
]

for vname, vsuffix in variants:
    print(f'\n=== {vname} ===')
    print('  eps     Top3%   Top5%   RDA@3%  Ties%  NDCG@3')
    for elabel, esuffix in eps_list:
        fname = f'data/{city}/results/stretch_ml_best{vsuffix}{esuffix}_result.json'
        if not os.path.exists(fname):
            print(f'  {elabel}  MISSING: {fname}')
            continue
        with open(fname) as f:
            d = json.load(f)
        top3  = d['item_top3_accuracy_with_ties'] * 100
        top5  = d['item_top5_accuracy_with_ties'] * 100
        rda3  = d['paths_within_3_ranks_ties'] * 100
        ties  = d['tie_statistics']['pct_od_with_ties']
        ndcg3 = d['ndcg@3'] * 100
        marker = ' *' if elabel == '0.10' else ''
        print(f'  {elabel:<6}  {top3:6.2f}  {top5:6.2f}  {rda3:6.2f}  {ties:5.1f}  {ndcg3:7.4f}{marker}')
