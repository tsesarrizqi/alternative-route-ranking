"""
Pool accuracy analysis across all cities and candidate generator variants.

Pool accuracy: for each OD pair, the model uses exactly two types of pool
lookups per path node (mirroring collate_fn in train_sharing.py):
  - pool_start: (origin -> node[i])  -- prefix shortest distance
  - pool_end:   (node[i] -> dest)    -- suffix shortest distance
We compare the pool's min distance (across all candidate paths for that OD)
against the true Dijkstra distance. High accuracy means the pool is a faithful
proxy for shortest-path distances, which is important for correct stretch scoring.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


def build_csr(edges_df: pd.DataFrame):
    node_ids = sorted(set(edges_df["start_id"].tolist() + edges_df["end_id"].tolist()))
    node_to_idx = {n: i for i, n in enumerate(node_ids)}
    n = len(node_ids)

    rows, cols, data = [], [], []
    for r in edges_df.itertuples(index=False):
        u = node_to_idx[int(r.start_id)]
        v = node_to_idx[int(r.end_id)]
        rows.append(u)
        cols.append(v)
        data.append(float(r.length))

    mat = csr_matrix((data, (rows, cols)), shape=(n, n))
    return mat, node_ids, node_to_idx


def build_edge_lookup(edges_df: pd.DataFrame) -> dict:
    return {
        (int(r.start_id), int(r.end_id)): float(r.length)
        for r in edges_df.itertuples(index=False)
    }


def parse_paths(paths_str: str) -> list:
    return [[int(n) for n in p.split(",")] for p in paths_str.split(";")]


def check_pool_accuracy(
    city: str,
    variant: str,
    graph_dir: str,
    paths_file: str,
    verbose: bool = False,
) -> dict | None:
    if not os.path.isfile(paths_file):
        print(f"  [SKIP] paths file not found: {paths_file}")
        return None

    edges_file = os.path.join(graph_dir, "edges.csv")
    if not os.path.isfile(edges_file):
        print(f"  [SKIP] edges.csv not found in {graph_dir}")
        return None

    t0 = time.time()
    print(f"  Loading data ...")
    paths_df = pd.read_csv(paths_file)
    edges_df = pd.read_csv(edges_file)

    mat, node_list, node_to_idx = build_csr(edges_df)
    edge_lookup = build_edge_lookup(edges_df)

    print(f"  Graph: {len(node_list):,} nodes, {len(edges_df):,} edges")
    print(f"  OD pairs: {len(paths_df):,}")

    print(f"  Parsing paths and computing pool distances ...")

    od_pool_start = {}
    od_pool_end   = {}
    unique_origins = set()
    unique_dests   = set()

    for row in paths_df.itertuples(index=False):
        origin = int(row.source)
        dest   = int(row.destination)
        od_key = (origin, dest)

        pool_start = {}  # node -> min dist from origin
        pool_end   = {}  # node -> min dist to dest

        for path in parse_paths(row.paths):
            # forward pass: pool_start
            cum = 0.0
            for i, node in enumerate(path):
                if node not in pool_start or cum < pool_start[node]:
                    pool_start[node] = cum
                if i < len(path) - 1:
                    d = edge_lookup.get((path[i], path[i + 1]))
                    cum = (cum + d) if d is not None else float("inf")

            # backward pass: pool_end
            sfx = 0.0
            for i in range(len(path) - 1, -1, -1):
                node = path[i]
                if node not in pool_end or sfx < pool_end[node]:
                    pool_end[node] = sfx
                if i > 0:
                    d = edge_lookup.get((path[i - 1], path[i]))
                    sfx = (sfx + d) if d is not None else float("inf")

        od_pool_start[od_key] = pool_start
        od_pool_end[od_key]   = pool_end
        unique_origins.add(origin)
        unique_dests.add(dest)

    print(f"  Unique origins: {len(unique_origins):,}  Unique dests: {len(unique_dests):,}")

    valid_origins = [o for o in unique_origins if o in node_to_idx]
    valid_dests   = [d for d in unique_dests   if d in node_to_idx]
    origin_indices = [node_to_idx[o] for o in valid_origins]
    dest_indices   = [node_to_idx[d] for d in valid_dests]

    mat_T = mat.T.tocsr()

    t_dijk = time.time()
    print(f"  Running {len(origin_indices):,} forward Dijkstra (batched) ...")
    fwd_matrix = dijkstra(mat,   directed=True, indices=origin_indices, return_predecessors=False)
    # fwd_matrix[i, j] = dist from valid_origins[i] to node_list[j]
    fwd_dist = {orig: fwd_matrix[i] for i, orig in enumerate(valid_origins)}

    print(f"  Running {len(dest_indices):,} backward Dijkstra (batched) ...")
    bwd_matrix = dijkstra(mat_T, directed=True, indices=dest_indices,   return_predecessors=False)
    # bwd_matrix[i, j] = dist from node_list[j] to valid_dests[i]
    bwd_dist = {dest: bwd_matrix[i] for i, dest in enumerate(valid_dests)}

    print(f"  Dijkstra done in {time.time()-t_dijk:.1f}s")

    print(f"  Comparing pool vs true distances ...")

    total = 0
    exact = near = small = medium = large = 0
    all_errors = []

    for od_key, pool_start in od_pool_start.items():
        origin, dest = od_key
        if origin not in fwd_dist or dest not in bwd_dist:
            continue
        fwd = fwd_dist[origin]
        bwd = bwd_dist[dest]
        pool_end = od_pool_end[od_key]

        for node, pd_dist in pool_start.items():
            if pd_dist == float("inf") or node not in node_to_idx:
                continue
            true_d = fwd[node_to_idx[node]]
            if true_d == np.inf or true_d == 0:
                continue
            rel_err = (pd_dist - true_d) / true_d
            all_errors.append(rel_err)
            total += 1
            if   rel_err < 0.001: exact  += 1
            elif rel_err < 0.01:  near   += 1
            elif rel_err < 0.05:  small  += 1
            elif rel_err < 0.10:  medium += 1
            else:                 large  += 1

        for node, pd_dist in pool_end.items():
            if pd_dist == float("inf") or node not in node_to_idx:
                continue
            true_d = bwd[node_to_idx[node]]
            if true_d == np.inf or true_d == 0:
                continue
            rel_err = (pd_dist - true_d) / true_d
            all_errors.append(rel_err)
            total += 1
            if   rel_err < 0.001: exact  += 1
            elif rel_err < 0.01:  near   += 1
            elif rel_err < 0.05:  small  += 1
            elif rel_err < 0.10:  medium += 1
            else:                 large  += 1

    elapsed = time.time() - t0
    arr = np.array(all_errors) if all_errors else np.array([0.0])

    return {
        "city": city,
        "variant": variant,
        "num_od_total": int(len(paths_df)),
        "num_subpaths_checked": int(total),
        "exact_rate":    round(exact  / total, 6) if total else None,
        "near_rate":     round(near   / total, 6) if total else None,
        "small_rate":    round(small  / total, 6) if total else None,
        "medium_rate":   round(medium / total, 6) if total else None,
        "large_rate":    round(large  / total, 6) if total else None,
        "pool_accuracy": round((exact + near) / total, 6) if total else None,
        "mean_rel_error":   round(float(np.mean(arr)), 6),
        "median_rel_error": round(float(np.median(arr)), 6),
        "p90_rel_error":    round(float(np.percentile(arr, 90)), 6),
        "p99_rel_error":    round(float(np.percentile(arr, 99)), 6),
        "max_rel_error":    round(float(np.max(arr)), 6),
        "elapsed_s": round(elapsed, 1),
    }


def print_summary_table(results: list) -> None:
    cities   = sorted({r["city"]    for r in results})
    variants = sorted({r["variant"] for r in results})
    lookup   = {(r["city"], r["variant"]): r for r in results}

    col_w = 28
    header = f"{'City':<15}" + "".join(f"{v[:col_w]:<{col_w}}" for v in variants)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("POOL ACCURACY  (% of (origin->node)+(node->dest) pairs with <1% error vs Dijkstra)")
    print(sep)
    print(header)
    print("-" * len(header))
    for city in cities:
        row = f"{city:<15}"
        for v in variants:
            r = lookup.get((city, v))
            if r is None:
                row += f"{'---':<{col_w}}"
            elif r["pool_accuracy"] is None:
                row += f"{'N/A':<{col_w}}"
            else:
                acc  = r["pool_accuracy"] * 100
                n    = r["num_subpaths_checked"]
                row += f"{acc:>6.2f}%  ({n:>8,}){'':<5}"
        print(row)
    print(sep)

    print(f"\n{sep}")
    print("MEAN RELATIVE ERROR  (pool dist vs true shortest)")
    print(sep)
    print(header)
    print("-" * len(header))
    for city in cities:
        row = f"{city:<15}"
        for v in variants:
            r = lookup.get((city, v))
            if r is None:
                row += f"{'---':<{col_w}}"
            else:
                err = r["mean_rel_error"] * 100
                row += f"{err:>7.3f}%{'':<{col_w - 8}}"
        print(row)
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Compute pool accuracy for all city/variant combinations"
    )
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--cities", nargs="+",
        default=["london", "manhattan", "paris", "singapore", "tokyo"])
    parser.add_argument("--variants", nargs="+",
        default=["ksp", "plateau_ub_2_dm_4_sim_07", "randomwalk", "stochastic_mu15"])
    parser.add_argument("--output", default="analysis/pool_accuracy_all_cities.json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    def paths_filename(city, variant):
        if variant == "ksp":
            return f"paths_scored_step2_{city}_ksp.csv"
        elif variant == "plateau_ub_2_dm_4_sim_07":
            return f"paths_scored_step2_paths_raw_step1_{city}_ub_2_dm_4_sim_07.csv"
        elif variant == "randomwalk":
            return f"paths_scored_step2_{city}_randomwalk.csv"
        elif variant == "stochastic_mu15":
            return f"paths_scored_step2_{city}_stochastic_mu15.csv"
        else:
            return f"paths_scored_step2_{city}_{variant}.csv"

    grand_t0 = time.time()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Auto-resume: load already-completed (city, variant) pairs from existing output
    all_results = []
    completed = set()
    if os.path.isfile(args.output):
        with open(args.output) as f:
            all_results = json.load(f)
        completed = {(r["city"], r["variant"]) for r in all_results}
        print(f"[RESUME] Found {len(completed)} completed result(s) in {args.output}")

    for city in args.cities:
        graph_dir = os.path.join(args.data_root, city, "graph")
        for variant in args.variants:
            if (city, variant) in completed:
                print(f"\n{'='*60}")
                print(f"[{city} / {variant}] Already done; skipping.")
                continue
            paths_file = os.path.join(graph_dir, paths_filename(city, variant))
            print(f"\n{'='*60}")
            print(f"[{city} / {variant}]")

            result = check_pool_accuracy(
                city=city, variant=variant,
                graph_dir=graph_dir, paths_file=paths_file,
                verbose=args.verbose,
            )
            if result is None:
                continue

            all_results.append(result)
            print(f"  pool_accuracy : {result['pool_accuracy']*100:.2f}%  "
                  f"({result['num_subpaths_checked']:,} pairs, {result['elapsed_s']:.1f}s)")
            print(f"  mean_rel_err  : {result['mean_rel_error']*100:.3f}%  "
                  f"p99: {result['p99_rel_error']*100:.3f}%")

            with open(args.output, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nResults written to {args.output}")
    if all_results:
        print_summary_table(all_results)
    print(f"\nTotal elapsed: {time.time()-grand_t0:.1f}s")


if __name__ == "__main__":
    main()
