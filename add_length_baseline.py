import argparse
import json, os
from itertools import product
import numpy as np
from evaluate_endtoend import compute_overlap, greedy_diversity_filter
from evaluate_endtoend import compute_diversity, compute_avg_stretch, compute_pairwise_jaccard
from evaluate_endtoend import compute_admissibility_rate, compute_diversity_compliance, _edge_length_sum
from dataset import RoadNetworkDataset

cities = ['london','manhattan','paris','singapore','tokyo',
          'amsterdam','bangalore','manchester','rome','seattle','sydney']

parser = argparse.ArgumentParser(description="Add length-ranking controls to existing epsilon evaluations.")
parser.add_argument('--cities', nargs='+', default=cities)
parser.add_argument('--variants', nargs='+', default=['randomwalk'],
                    choices=['randomwalk', 'plateau_ub_2_dm_4_sim_07', 'stochastic_mu15'])
parser.add_argument('--top_k', nargs='+', type=int, default=[3, 5])
args = parser.parse_args()

for top_k in args.top_k:
    for city, variant in product(args.cities, args.variants):
        results_path = f"data/{city}/results/endtoend_{city}_{variant}_top{top_k}.json"
        if not os.path.exists(results_path):
            raise FileNotFoundError(results_path)
        print(f"Processing: {results_path}")
        with open(results_path) as f:
            out = json.load(f)

        data_dir = out['config']['data_dir']
        paths_path = out['config']['paths_path']
        theta = out['config'].get('theta', 0.5)
        delta = out['config'].get('delta', 0.1)
        adm_threshold = out['config'].get('adm_threshold', 100.0)

        # load dataset
        ds = RoadNetworkDataset(data_dir=f"{data_dir}/graph", paths_path=paths_path,
                    precompute_subpaths=False)
        edge_lengths_np = ds.edge_lengths.numpy()
        # reproduce split
        _, _, _, _, _, test_indices = ds.split_dataset_three_ways(val_size=0.1, test_size=0.2, random_state=42)

        per_od_len_filter = []
        per_od_len_nofilter = []
        processed = 0

        for od_idx in test_indices:
            optimal_distance = float(ds.paths_df.iloc[od_idx]['optimal_distance'])
            if optimal_distance <= 1e-9:
                continue
            edge_idxs = ds.candidate_edge_indices[od_idx]
            true_scores = ds.candidate_stretch_scores[od_idx]
            n_paths = len(edge_idxs)
            if n_paths < 1:
                continue
            dataset_paths = []
            for i in range(n_paths):
                ts = float(true_scores[i]) if i < len(true_scores) else 0.0
                dataset_paths.append((0.0, edge_idxs[i], ts))

            # length scoring
            length_scored = []
            for _, edges, ts in dataset_paths:
                length = _edge_length_sum(edges, edge_lengths_np)
                length_scored.append((-float(length), edges, ts))
            length_sorted = sorted(length_scored, key=lambda x: x[0], reverse=True)
            c_filter = greedy_diversity_filter(length_sorted, top_k, theta, delta, edge_lengths_np)
            c_nofilter = length_sorted[:top_k]

            od_src = int(ds.source_nodes[od_idx])
            od_dst = int(ds.dest_nodes[od_idx])
            per_od_len_filter.append({
                'od_pair': [od_src, od_dst],
                'diversity': float(compute_diversity(c_filter, optimal_distance, edge_lengths_np)),
                'avg_stretch': float(compute_avg_stretch(c_filter)),
                'pairwise_jaccard': float(compute_pairwise_jaccard(c_filter, edge_lengths_np)),
                'admissibility_rate': float(compute_admissibility_rate(c_filter, adm_threshold)),
                'diversity_compliance': float(compute_diversity_compliance(c_filter, edge_lengths_np, theta)),
                'n_selected': int(len(c_filter)),
            })

            per_od_len_nofilter.append({
                'od_pair': [od_src, od_dst],
                'diversity': float(compute_diversity(c_nofilter, optimal_distance, edge_lengths_np)),
                'avg_stretch': float(compute_avg_stretch(c_nofilter)),
                'pairwise_jaccard': float(compute_pairwise_jaccard(c_nofilter, edge_lengths_np)),
                'admissibility_rate': float(compute_admissibility_rate(c_nofilter, adm_threshold)),
                'diversity_compliance': float(compute_diversity_compliance(c_nofilter, edge_lengths_np, theta)),
                'n_selected': int(len(c_nofilter)),
            })

            processed += 1
            if processed % 200 == 0:
                print(f"  processed {processed} ...")

        expected_ods = {tuple(row['od_pair']) for row in out['per_od']['model_filter']}
        actual_ods = {tuple(row['od_pair']) for row in per_od_len_filter}
        if actual_ods != expected_ods or len(per_od_len_filter) != len(out['per_od']['model_filter']):
            raise ValueError(f"Length baseline OD split differs from original evaluation: {results_path}")

        # compute summary stats
        def summarize(lst):
            if not lst:
                return None
            stretches = [r['avg_stretch'] for r in lst]
            jaccards = [r.get('pairwise_jaccard',0.0) for r in lst]
            adm = [r.get('admissibility_rate',0.0) for r in lst]
            divc = [r.get('diversity_compliance',0.0) for r in lst]
            return {
                'n_od_pairs': len(lst),
                'avg_stretch_mean': float(np.mean(stretches)),
                'avg_stretch_std': float(np.std(stretches)),
                'pairwise_jaccard_mean': float(np.mean(jaccards)),
                'pairwise_jaccard_std': float(np.std(jaccards)),
                'admissibility_rate_mean': float(np.mean(adm)),
                'admissibility_rate_std': float(np.std(adm)),
                'diversity_compliance_mean': float(np.mean(divc)),
                'diversity_compliance_std': float(np.std(divc)),
            }

        out['per_od']['length_filter'] = per_od_len_filter
        out['per_od']['length_nofilter'] = per_od_len_nofilter
        out['summary']['length_filter'] = summarize(per_od_len_filter)
        out['summary']['length_nofilter'] = summarize(per_od_len_nofilter)

        # write back
        with open(results_path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"  Wrote updated results to {results_path}")
