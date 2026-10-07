import json, glob, os
from collections import defaultdict

cities = ['london', 'manhattan', 'paris', 'singapore', 'tokyo']
generators = ['ksp', 'plateau_ub_2_dm_4_sim_07', 'randomwalk', 'stochastic_mu15']
gen_labels = ['KSP', 'Plateau', 'Randomwalk', 'Stochastic']

conds = ['oracle_filter', 'model_filter', 'gen_filter', 'model_nofilter', 'gen_nofilter']
cond_labels = {
    'oracle_filter':   'Oracle + Filter (upper bound)',
    'model_filter':    'Model + Filter (Ours)',
    'gen_filter':      'Generator Order + Filter',
    'model_nofilter':  'Model, No Filter',
    'gen_nofilter':    'Generator Order, No Filter',
}

for top_k in [3, 5]:
    print(f'\n====== top_k = {top_k} (5-city averages: London, Manhattan, Paris, Singapore, Tokyo) ======')
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for city in cities:
        for gen in generators:
            fname = f'data/{city}/results/endtoend_{city}_{gen}_top{top_k}.json'
            if not os.path.exists(fname):
                print(f'  MISSING: {fname}')
                continue
            with open(fname) as fp:
                d = json.load(fp)
            for cond in conds:
                s = d['summary'].get(cond)
                if s is None:
                    print(f'  MISSING cond "{cond}" in {fname}')
                    continue
                results[cond][gen]['str'].append(s['avg_stretch_mean'])
                results[cond][gen]['jac'].append(s['pairwise_jaccard_mean'])
                results[cond][gen]['adm'].append(s.get('admissibility_rate_mean', float('nan')))
                results[cond][gen]['dc'].append(s.get('diversity_compliance_mean', float('nan')))

    # Print table
    for metric in ['str', 'jac', 'adm', 'dc']:
        metric_label = {'str': 'Avg Stretch', 'jac': 'Pairwise Jaccard', 'adm': 'Adm Rate', 'dc': 'Div Compl'}[metric]
        print(f'\n  [{metric_label}]')
        print(f'  {"Condition":<40}' + ''.join(f' {g:>12}' for g in gen_labels))
        print('  ' + '-' * (40 + 12 * len(gen_labels)))
        for cond in conds:
            row = f'  {cond_labels[cond]:<40}'
            for gen in generators:
                vals = results[cond][gen][metric]
                if vals:
                    avg = sum(vals) / len(vals)
                    row += f' {avg:>12.4f}'
                else:
                    row += f' {"N/A":>12}'
            print(row)
