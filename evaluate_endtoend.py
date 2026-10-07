"""
evaluate_endtoend.py

End-to-end output quality evaluation as described in the paper (Section 4.2).
Evaluates all 5 conditions over the test split:

  1. Model + Filter       — predicted scores → greedy diversity filter → top-k
  2. Generator Order + Filter — native generator order → greedy diversity filter → top-k
  3. Model, No Filter     — predicted scores → top-k directly
  4. Generator Order, No Filter — first-k in native generator order
  5. Oracle + Filter      — true stretch scores → greedy diversity filter → top-k

Metrics (per condition, averaged over test OD pairs):
  - Set diversity   : sum of unique edge lengths in top-k set / optimal_distance
  - Avg stretch     : mean ground-truth stretch score of the top-k paths

Usage:
  python evaluate_endtoend.py \\
      --data_dir data/london \\
      --model_suffix _london_ksp \\
      --paths_path paths_scored_step2_london_ksp.csv \\
      --genorder_path paths_genorder_london_ksp.csv \\
      --top_k 3 --theta 0.5 --delta 0.1
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import RoadNetworkDataset, RoadNetworkDatasetSubset
from model_stretch import PathStretchScoringModel
from train_sharing import collate_fn


# ---------------------------------------------------------------------------
# Diversity filter  (paper Algorithm 2)
# ---------------------------------------------------------------------------

def _edge_length_sum(edge_indices, edge_lengths_np):
    return sum(edge_lengths_np[e] for e in edge_indices if e >= 0)


def compute_overlap(edges_a, edges_b, edge_lengths_np):
    """overlap(p1,p2) = |E(p1) ∩ E(p2)| / min(|p1|, |p2|)  (length-weighted)"""
    set_a = set(e for e in edges_a if e >= 0)
    set_b = set(e for e in edges_b if e >= 0)
    shared = sum(edge_lengths_np[e] for e in (set_a & set_b))
    denom = min(
        sum(edge_lengths_np[e] for e in set_a),
        sum(edge_lengths_np[e] for e in set_b),
    )
    return shared / denom if denom > 1e-9 else 1.0


def greedy_diversity_filter(scored_paths, k, theta, delta, edge_lengths_np):
    """
    scored_paths : list of (score, edge_indices, stretch_score)  — sorted score desc
    Returns list of up to k selected items.
    """
    result = []
    theta_cur = theta
    while len(result) < k and theta_cur <= 1.0 + 1e-9:
        result = []
        for item in scored_paths:
            if len(result) >= k:
                break
            edges = item[1]
            if all(compute_overlap(edges, sel[1], edge_lengths_np) <= theta_cur
                   for sel in result):
                result.append(item)
        if len(result) >= k:
            break
        theta_cur += delta
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_diversity(selected, optimal_distance, edge_lengths_np):
    """Set diversity = unique edge length sum / optimal_distance."""
    if not selected or optimal_distance <= 1e-9:
        return 0.0
    union_edges = set()
    for _, edges, _ in selected:
        union_edges.update(e for e in edges if e >= 0)
    return float(sum(edge_lengths_np[e] for e in union_edges) / optimal_distance)


def compute_avg_stretch(selected):
    """Mean ground-truth stretch score of selected paths."""
    if not selected:
        return 0.0
    return float(np.mean([s for _, _, s in selected]))


def compute_pairwise_jaccard(selected, edge_lengths_np):
    """Mean pairwise Jaccard distance (length-weighted) within the selected set.
    Jaccard distance = 1 - |E(p) ∩ E(q)| / |E(p) ∪ E(q)|  (edge-length weighted).
    Returns 0.0 when fewer than 2 paths are selected.
    """
    if len(selected) < 2:
        return 0.0
    edge_sets = [set(e for e in item[1] if e >= 0) for item in selected]
    total, count = 0.0, 0
    for i in range(len(edge_sets)):
        for j in range(i + 1, len(edge_sets)):
            inter = sum(edge_lengths_np[e] for e in (edge_sets[i] & edge_sets[j]))
            union = sum(edge_lengths_np[e] for e in (edge_sets[i] | edge_sets[j]))
            jaccard_sim = inter / union if union > 1e-9 else 1.0
            total += 1.0 - jaccard_sim
            count += 1
    return float(total / count) if count > 0 else 0.0


def compute_admissibility_rate(selected, threshold):
    """Fraction of k selected paths whose ground-truth stretch score >= threshold.
    A path is considered admissible if at least `threshold`% of its subpaths
    satisfy the UBS condition (i.e., true_score >= threshold).
    selected : list of (pred_score, edge_indices, true_stretch_score)
    """
    if not selected:
        return 0.0
    return float(sum(1 for _, _, ss in selected if ss >= threshold) / len(selected))


def compute_diversity_compliance(selected, edge_lengths_np, theta):
    """Returns 1.0 if every pair of selected paths satisfies overlap(p,q) <= theta,
    0.0 otherwise.  This directly checks whether the returned set meets the
    diversity filter's target threshold without relaxation.
    """
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            if compute_overlap(selected[i][1], selected[j][1], edge_lengths_np) > theta:
                return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Load generator-order file
# ---------------------------------------------------------------------------

def load_genorder_file(genorder_path, data_dir, edge_lookup):
    """
    Parse a paths_genorder_*.csv file.
    Returns dict: (source, dest) -> list of (edge_indices, stretch_score)
    in native generator order.
    """
    full_path = os.path.join(data_dir, "graph", genorder_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Generator-order file not found: {full_path}")

    od_data = {}
    with open(full_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = int(row["source"])
            dst = int(row["destination"])
            node_paths = [
                list(map(int, p.split(",")))
                for p in row["paths"].split(";")
            ]
            stretch_scores = [float(x) for x in row["stretch_scores"].split(";")]

            path_list = []
            for nseq, ss in zip(node_paths, stretch_scores):
                edge_indices = []
                valid = True
                for i in range(len(nseq) - 1):
                    key = (nseq[i], nseq[i + 1])
                    if key in edge_lookup:
                        edge_indices.append(edge_lookup[key])
                    else:
                        valid = False
                        break
                if valid and edge_indices:
                    path_list.append((edge_indices, ss))

            if path_list:
                od_data[(src, dst)] = path_list

    return od_data


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def collect_predictions(model, data_loader, graph_data, device):
    """Returns dict: od_pair -> {pred_scores, true_scores, optimal_distance}"""
    model.eval()
    results = {}
    with torch.no_grad():
        for batch in data_loader:
            for k2, v in batch.items():
                if torch.is_tensor(v):
                    batch[k2] = v.to(device)
            pred_stretch = model(graph_data, batch, task="stretch")
            for i, od_pair in enumerate(batch["od_pairs"]):
                od_key = tuple(int(x) for x in od_pair)
                results[od_key] = {
                    "pred_scores":     pred_stretch[i].cpu().numpy().tolist(),
                    "true_scores":     [float(x) for x in batch["stretch_scores"][i]],
                    "optimal_distance": float(batch["optimal_distances"][i]),
                }
    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset (always required for edge info, test split, path sequences)
    # ------------------------------------------------------------------
    print("\nLoading dataset (scored file)...")
    dataset = RoadNetworkDataset(
        data_dir=f"{args.data_dir}/graph",
        exclude_main_path=args.exclude_main,
        precompute_subpaths=True,
        paths_path=args.paths_path,
    )
    edge_lengths_np = dataset.edge_lengths.numpy()

    print("Loading generator-order file...")
    od_genorder = load_genorder_file(args.genorder_path, args.data_dir, dataset.edge_lookup)
    print(f"  Loaded {len(od_genorder)} OD pairs from genorder file")

    print("Reproducing training split (random_state=42)...")
    _, _, _, _, _, test_indices = dataset.split_dataset_three_ways(
        val_size=0.1, test_size=0.2, random_state=42
    )
    test_dataset = RoadNetworkDatasetSubset(dataset, test_indices)
    print(f"Test samples: {len(test_dataset)}")

    # ------------------------------------------------------------------
    # Predictions: load from cache or run model inference
    # ------------------------------------------------------------------
    preds_cache = args.preds_cache
    if preds_cache and os.path.exists(preds_cache):
        print(f"\nLoading cached predictions from: {preds_cache}")
        with open(preds_cache) as f:
            raw = json.load(f)
        # JSON keys are strings "src,dst"; restore to int tuples
        od_preds = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in raw.items()
        }
        print(f"  Loaded predictions for {len(od_preds)} OD pairs")
    else:
        model_path = f"{args.data_dir}/models/stretch_ml_best{args.model_suffix}.pth"
        print(f"\nLoading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        ckpt_args = checkpoint["args"]

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        model = PathStretchScoringModel(
            hidden_dim=ckpt_args.hidden_dim,
            dropout=0.0,
            enable_gnn_od=ckpt_args.enable_gnn_od,
            epsilon=ckpt_args.epsilon,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        graph_data = dataset.graph_data.to(device)

        print("\nRunning model inference on test set...")
        od_preds = collect_predictions(model, test_loader, graph_data, device)
        print(f"Got predictions for {len(od_preds)} OD pairs")

        if preds_cache:
            cache_dir = os.path.dirname(preds_cache)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            serializable = {f"{k[0]},{k[1]}": v for k, v in od_preds.items()}
            with open(preds_cache, "w") as f:
                json.dump(serializable, f)
            print(f"Predictions cached to: {preds_cache}")

    # ------------------------------------------------------------------
    # Per-condition accumulators
    # ------------------------------------------------------------------
    compute_length = any(
        variant in identifier
        for variant in ("randomwalk", "plateau", "ub_2_dm_4_sim_07", "stochastic")
        for identifier in (args.model_suffix, args.paths_path, args.genorder_path)
    )

    cond_names = [
        "model_filter",
        "gen_filter",
        "model_nofilter",
        "gen_nofilter",
        "oracle_filter",
    ]
    if compute_length:
        cond_names.extend(["length_filter", "length_nofilter"])

    stats = {c: [] for c in cond_names}
    skipped_no_preds = 0
    skipped_no_genorder = 0
    processed = 0

    print(f"\nEvaluating (top_k={args.top_k}, theta={args.theta}, delta={args.delta}) ...")

    for od_idx in test_indices:
        source = int(dataset.source_nodes[od_idx])
        dest   = int(dataset.dest_nodes[od_idx])
        od_key = (source, dest)

        if od_key not in od_preds:
            skipped_no_preds += 1
            continue

        info = od_preds[od_key]
        pred_scores      = np.array(info["pred_scores"])
        true_scores      = np.array(info["true_scores"])
        optimal_distance = info["optimal_distance"]

        if optimal_distance <= 1e-9:
            continue

        node_seqs  = dataset.candidate_node_sequences[od_idx]
        edge_idxs  = dataset.candidate_edge_indices[od_idx]
        n_paths    = len(node_seqs)
        if n_paths < 1:
            continue

        # Build path tuples from scored dataset: (score, edge_indices, stretch_score)
        # for model and oracle conditions
        dataset_paths = []
        for i in range(n_paths):
            ps = float(pred_scores[i]) if i < len(pred_scores) else 0.0
            ts = float(true_scores[i]) if i < len(true_scores) else 0.0
            dataset_paths.append((ps, edge_idxs[i], ts))

        # ---- Condition 1: Model + Filter ----
        model_sorted = sorted(dataset_paths, key=lambda x: x[0], reverse=True)
        c1 = greedy_diversity_filter(model_sorted, args.top_k, args.theta, args.delta, edge_lengths_np)

        # ---- Condition 3: Model, No Filter ----
        c3 = model_sorted[:args.top_k]

        # ---- Condition 5: Oracle + Filter ----
        oracle_sorted = sorted(dataset_paths, key=lambda x: x[2], reverse=True)
        c5 = greedy_diversity_filter(oracle_sorted, args.top_k, args.theta, args.delta, edge_lengths_np)

        # ---- Length-based ranking: shortest path length (randomwalk only) ----
        c6 = c7 = None
        if compute_length:
            # Build scored list where higher score implies shorter total length
            length_scored = []
            for _, edges, ts in dataset_paths:
                length = _edge_length_sum(edges, edge_lengths_np)
                # use negative length so that sorting descending puts shortest first
                length_scored.append((-float(length), edges, ts))
            length_sorted = sorted(length_scored, key=lambda x: x[0], reverse=True)
            c6 = greedy_diversity_filter(length_sorted, args.top_k, args.theta, args.delta, edge_lengths_np)
            c7 = length_sorted[:args.top_k]

        # ---- Conditions 2 & 4: Generator Order ----
        if od_key not in od_genorder:
            skipped_no_genorder += 1
            # Still record model/oracle conditions
            record_items = [("model_filter", c1), ("model_nofilter", c3), ("oracle_filter", c5)]
            if compute_length:
                record_items.extend([("length_filter", c6), ("length_nofilter", c7)])
            for cname, sel in record_items:
                stats[cname].append({
                    "od_pair":              list(od_key),
                    "diversity":            compute_diversity(sel, optimal_distance, edge_lengths_np),
                    "avg_stretch":          compute_avg_stretch(sel),
                    "pairwise_jaccard":     compute_pairwise_jaccard(sel, edge_lengths_np),
                    "admissibility_rate":   compute_admissibility_rate(sel, args.adm_threshold),
                    "diversity_compliance": compute_diversity_compliance(sel, edge_lengths_np, args.theta),
                    "n_selected":           len(sel),
                })
            processed += 1
            continue

        # Generator-order paths: list of (edge_indices, stretch_score) in native order
        gen_paths_raw = od_genorder[od_key]
        # Convert to same 3-tuple format; score field unused for ranking (already in order)
        gen_paths = [(0.0, ep, ss) for ep, ss in gen_paths_raw]

        # ---- Condition 2: Generator Order + Filter ----
        c2 = greedy_diversity_filter(gen_paths, args.top_k, args.theta, args.delta, edge_lengths_np)

        # ---- Condition 4: Generator Order, No Filter ----
        c4 = gen_paths[:args.top_k]

        # Record all conditions (length-based ones only included when
        # `compute_length` is True). Build the list in the same order as
        # `cond_names` so the printed table stays consistent.
        record_list = [c1, c2, c3, c4, c5]
        if compute_length:
            record_list.extend([c6, c7])
        for cname, sel in zip(cond_names, record_list):
            stats[cname].append({
                "od_pair":              list(od_key),
                "diversity":            compute_diversity(sel, optimal_distance, edge_lengths_np),
                "avg_stretch":          compute_avg_stretch(sel),
                "pairwise_jaccard":     compute_pairwise_jaccard(sel, edge_lengths_np),
                "admissibility_rate":   compute_admissibility_rate(sel, args.adm_threshold),
                "diversity_compliance": compute_diversity_compliance(sel, edge_lengths_np, args.theta),
                "n_selected":           len(sel),
            })

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{len(test_indices)} ...")

    print(f"\nDone. Processed {processed} OD pairs "
          f"(skipped: {skipped_no_preds} no-pred, {skipped_no_genorder} no-genorder)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    labels = {
        "model_filter":    "Model + Filter        (Ours)",
        "gen_filter":      "Generator Order + Filter",
        "model_nofilter":  "Model, No Filter",
        "gen_nofilter":    "Generator Order, No Filter",
        "oracle_filter":   "Oracle + Filter       (Upper bound)",
        "length_filter":   "Length + Filter       (Shortest length)",
        "length_nofilter": "Length, No Filter",
    }

    summary = {}
    print(f"\n{'='*95}")
    print(f"END-TO-END EVALUATION  |  top_k={args.top_k}  theta={args.theta}  delta={args.delta}")
    print(f"{'='*95}")
    print(f"{'Condition':<42}  {'Avg Stretch':>11}  {'Pairwise Jaccard':>16}  {'Adm Rate':>8}  {'Div Compl':>9}")
    print(f"{'-'*95}")

    for cname in cond_names:
        s = stats[cname]
        if not s:
            continue
        stretches = [r["avg_stretch"]                    for r in s]
        jaccards  = [r.get("pairwise_jaccard", 0.0)      for r in s]
        adm_rates = [r.get("admissibility_rate", 0.0)    for r in s]
        div_compl = [r.get("diversity_compliance", 0.0)  for r in s]
        s_mean, s_std = float(np.mean(stretches)), float(np.std(stretches))
        j_mean, j_std = float(np.mean(jaccards)),  float(np.std(jaccards))
        a_mean, a_std = float(np.mean(adm_rates)), float(np.std(adm_rates))
        c_mean, c_std = float(np.mean(div_compl)), float(np.std(div_compl))
        summary[cname] = {
            "n_od_pairs":                len(s),
            "avg_stretch_mean":          s_mean,
            "avg_stretch_std":           s_std,
            "pairwise_jaccard_mean":     j_mean,
            "pairwise_jaccard_std":      j_std,
            "admissibility_rate_mean":   a_mean,
            "admissibility_rate_std":    a_std,
            "diversity_compliance_mean": c_mean,
            "diversity_compliance_std":  c_std,
        }
        print(f"  {labels[cname]:<40}  {s_mean:>11.4f}  {j_mean:>16.4f}  {a_mean:>8.4f}  {c_mean:>9.4f}")

    print(f"{'='*95}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    results_dir = f"{args.data_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    out_file = f"{results_dir}/endtoend{args.model_suffix}_top{args.top_k}.json"

    output = {
        "config": {
            "data_dir":       args.data_dir,
            "model_suffix":   args.model_suffix,
            "paths_path":     args.paths_path,
            "genorder_path":  args.genorder_path,
            "top_k":          args.top_k,
            "theta":          args.theta,
            "delta":          args.delta,
            "adm_threshold":  args.adm_threshold,
            "n_test_od":      processed,
        },
        "summary": summary,
        "per_od":  stats,
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_file}")
    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation (5 conditions)")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--data_dir",      type=str,   default="data/london")
    parser.add_argument("--model_suffix",  type=str,   default="_london_ksp")
    parser.add_argument("--paths_path",    type=str,   default="paths_scored_step2_london_ksp.csv")
    parser.add_argument("--genorder_path", type=str,   default="paths_genorder_london_ksp.csv")
    parser.add_argument("--exclude_main",  action="store_true")
    parser.add_argument("--top_k",         type=int,   default=3)
    parser.add_argument("--theta",         type=float, default=0.5)
    parser.add_argument("--delta",         type=float, default=0.1)
    parser.add_argument("--adm_threshold", type=float, default=100.0,
                        help="Stretch score threshold for admissibility rate "
                             "(a path is admissible if true_score >= this value). "
                             "Default: 100.0 — fully UBS-admissible at epsilon=0.1.")
    parser.add_argument("--preds_cache",   type=str,   default=None,
                        help="Path to JSON file for caching model predictions. "
                             "If the file exists, inference is skipped and predictions "
                             "are loaded from it. Otherwise inference runs and results "
                             "are saved to this path.")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
