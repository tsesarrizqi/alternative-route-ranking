import json, os, numpy as np

cities = ['london','manhattan','paris','singapore','tokyo',
          'amsterdam','bangalore','manchester','rome','seattle','sydney']
variants = [
    ('ksp',        ['_london_ksp','_manhattan_ksp','_paris_ksp','_singapore_ksp','_tokyo_ksp',
                    '_amsterdam_ksp','_bangalore_ksp','_manchester_ksp','_rome_ksp','_seattle_ksp','_sydney_ksp']),
    ('plateau',    ['_london_plateau_ub_2_dm_4_sim_07','_manhattan_plateau_ub_2_dm_4_sim_07','_paris_plateau_ub_2_dm_4_sim_07','_singapore_plateau_ub_2_dm_4_sim_07','_tokyo_plateau_ub_2_dm_4_sim_07',
                    '_amsterdam_plateau_ub_2_dm_4_sim_07','_bangalore_plateau_ub_2_dm_4_sim_07','_manchester_plateau_ub_2_dm_4_sim_07','_rome_plateau_ub_2_dm_4_sim_07','_seattle_plateau_ub_2_dm_4_sim_07','_sydney_plateau_ub_2_dm_4_sim_07']),
    ('randomwalk', ['_london_randomwalk','_manhattan_randomwalk','_paris_randomwalk','_singapore_randomwalk','_tokyo_randomwalk',
                    '_amsterdam_randomwalk','_bangalore_randomwalk','_manchester_randomwalk','_rome_randomwalk','_seattle_randomwalk','_sydney_randomwalk']),
    ('stochastic', ['_london_stochastic_mu15','_manhattan_stochastic_mu15','_paris_stochastic_mu15','_singapore_stochastic_mu15','_tokyo_stochastic_mu15',
                    '_amsterdam_stochastic_mu15','_bangalore_stochastic_mu15','_manchester_stochastic_mu15','_rome_stochastic_mu15','_seattle_stochastic_mu15','_sydney_stochastic_mu15']),
]
cname_map = [
    ('Oracle + Filter',             'oracle_filter'),
    ('Model + Filter',              'model_filter'),
    ('Generator Order + Filter',    'gen_filter'),
    ('Model, No Filter',            'model_nofilter'),
    ('Generator Order, No Filter',  'gen_nofilter'),
]

for top_k in [3, 5]:
    print(f'\n{"="*80}')
    print(f'top_k = {top_k}')
    print(f'{"="*80}')
    for vname, suffixes in variants:
        city_data = {}
        for city, suffix in zip(cities, suffixes):
            path = f'data/{city}/results/endtoend{suffix}_top{top_k}.json'
            if not os.path.exists(path):
                print(f'MISSING: {path}')
                continue
            with open(path) as f:
                d = json.load(f)
            city_data[city] = d

        print(f'\nVariant: {vname}')
        hdr = f'  {"Condition":<38} {"Div":>7} {"Stretch":>9} {"PW-Jacc":>9}'
        print(hdr)
        print('  ' + '-'*(len(hdr)-2))
        # For randomwalk variant, append a Length-based ranking condition.
        local_cname_map = list(cname_map)
        if vname == 'randomwalk':
            local_cname_map.append(('Length + Filter', 'length_filter'))

        for cname, ckey in local_cname_map:
            divs, stretches, jaccards = [], [], []
            for city in cities:
                if city not in city_data:
                    continue
                s = city_data[city]['summary'].get(ckey, {})
                if s:
                    divs.append(s['diversity_compliance_mean'])
                    stretches.append(s['avg_stretch_mean'])
                    jaccards.append(s['pairwise_jaccard_mean'])
            if divs:
                print(f'  {cname:<38} {np.mean(divs):7.4f} {np.mean(stretches):9.4f} {np.mean(jaccards):9.4f}')
            else:
                print(f'  {cname:<38} N/A')
