"""
rescore_stretch.py

Re-compute the stretch_scores column in a paths CSV using a given epsilon.
Uses the real directed road network from nodes.csv + edges.csv.

Design: streaming + threading.
  - Pass 1: build source_usages index (no giant pair cache).
  - Pass 2: ThreadPoolExecutor runs Dijkstra batches in parallel.
            scipy's Dijkstra releases the GIL -> real parallelism, no pickling.
  - Pass 3: write output CSVs.

Usage:
    python rescore_stretch.py \\
        --data_dir data/london \\
        --paths_path graph/paths_scored_step2_london_ksp.csv \\
        --epsilon 0.2 \\
        --output_path graph/paths_scored_step2_london_ksp_eps0.20.csv

    # Batch mode: multiple epsilons, single Dijkstra pass
    python rescore_stretch.py \\
        --data_dir data/london \\
        --paths_path graph/paths_scored_step2_london_ksp.csv \\
        --epsilons 0.05 0.2 \\
        --output_dir graph/
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as scipy_dijkstra


# ---------------------------------------------------------------------------
# Module-level state shared across threads (read-only during Pass 2)
# ---------------------------------------------------------------------------

_g_all_paths     = None   # list[list[list[int]]]
_g_source_usages = None   # dict u_idx -> [(row_idx, path_idx, i), ...]
_g_edge_len      = None   # dict (start_id, end_id) -> float
_g_id_to_idx     = None   # dict node_id -> int
_g_thresholds    = None   # list of float  (1+eps per epsilon)
_g_csr           = None   # scipy csr_matrix (read-only)


def _process_batch(batch_sources):
    """
    Thread worker: run Dijkstra for batch_sources, stream results into
    local score accumulators, return small dicts (no large arrays).
    """
    dist_mat = scipy_dijkstra(_g_csr, directed=True, indices=batch_sources,
                               return_predecessors=False)

    total_delta = defaultdict(float)
    sat_delta   = [defaultdict(float) for _ in _g_thresholds]

    for li, u_idx in enumerate(batch_sources):
        usages = _g_source_usages.get(u_idx)
        if not usages:
            continue
        dist_row = dist_mat[li]

        for (row_idx, path_idx, i) in usages:
            path = _g_all_paths[row_idx][path_idx]
            L = len(path)
            subpath_len = 0.0

            for j in range(i + 1, L):
                seg = _g_edge_len.get((path[j - 1], path[j]))
                if seg is None:
                    break
                subpath_len += seg

                v_idx = _g_id_to_idx.get(path[j])
                if v_idx is None:
                    continue
                d = dist_row[v_idx]
                if not np.isfinite(d) or d <= 0.0:
                    continue

                key = (row_idx, path_idx)
                total_delta[key] += subpath_len
                for ei, thresh in enumerate(_g_thresholds):
                    if subpath_len <= thresh * d:
                        sat_delta[ei][key] += subpath_len

    return dict(total_delta), [dict(s) for s in sat_delta]


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def build_graph(data_dir):
    nodes_df = pd.read_csv(os.path.join(data_dir, "graph", "nodes.csv"))
    edges_df = pd.read_csv(os.path.join(data_dir, "graph", "edges.csv"))

    node_ids = nodes_df["id"].values
    id_to_idx = {int(nid): i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    starts  = edges_df["start_id"].values.astype(np.int64)
    ends    = edges_df["end_id"].values.astype(np.int64)
    lengths = edges_df["length"].values.astype(np.float64)

    src_idx = np.array([id_to_idx[int(s)] for s in starts], dtype=np.int32)
    dst_idx = np.array([id_to_idx[int(d)] for d in ends],   dtype=np.int32)

    csr = csr_matrix((lengths, (src_idx, dst_idx)), shape=(n, n))
    edge_len = {(int(s), int(e)): float(l) for s, e, l in zip(starts, ends, lengths)}
    return id_to_idx, csr, edge_len


# ---------------------------------------------------------------------------
# Pass 1: parse paths + build source-usage index
# ---------------------------------------------------------------------------

def parse_paths(paths_str):
    return [list(map(int, p.split(","))) for p in paths_str.split(";")]


def collect_source_usages(df, id_to_idx):
    all_paths     = []
    source_usages = defaultdict(list)

    for row_idx, (_, row) in enumerate(df.iterrows()):
        paths = parse_paths(row["paths"])
        all_paths.append(paths)
        for path_idx, path in enumerate(paths):
            for i in range(len(path) - 1):
                u = id_to_idx.get(path[i])
                if u is not None:
                    source_usages[u].append((row_idx, path_idx, i))

    return all_paths, dict(source_usages)


# ---------------------------------------------------------------------------
# Pass 2: threaded streaming Dijkstra
# ---------------------------------------------------------------------------

def compute_scores_streaming(all_paths, source_usages, edge_len, id_to_idx,
                              csr, epsilons, n_workers=8, batch_size=200):
    global _g_all_paths, _g_source_usages, _g_edge_len
    global _g_id_to_idx, _g_thresholds, _g_csr

    # Set module globals (threads read these without pickling)
    _g_all_paths     = all_paths
    _g_source_usages = source_usages
    _g_edge_len      = edge_len
    _g_id_to_idx     = id_to_idx
    _g_thresholds    = [1.0 + e for e in epsilons]
    _g_csr           = csr

    source_list = list(source_usages.keys())
    n_sources   = len(source_list)
    batches     = [source_list[i:i + batch_size]
                   for i in range(0, n_sources, batch_size)]
    n_batches   = len(batches)
    print(f"  {n_sources:,} unique source nodes, {n_batches} batches of {batch_size}")

    total_len = defaultdict(float)
    sat_len   = [defaultdict(float) for _ in epsilons]

    def _merge(td, sdl):
        for k, v in td.items():
            total_len[k] += v
        for ei, sd in enumerate(sdl):
            for k, v in sd.items():
                sat_len[ei][k] += v

    t0   = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_batch, b): b for b in batches}
        for fut in as_completed(futures):
            td, sdl = fut.result()
            _merge(td, sdl)
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (n_batches - done)
            print(f"  Batch {done}/{n_batches}  ({done/n_batches*100:.0f}%)  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s    ", end="\r")

    print()
    print(f"  Done in {time.time()-t0:.1f}s")
    return total_len, sat_len


# ---------------------------------------------------------------------------
# Pass 3: write output CSVs
# ---------------------------------------------------------------------------

def write_output_csvs(df, all_paths, total_len, sat_len, epsilons, output_paths):
    for ei, eps in enumerate(epsilons):
        print(f"\n  Writing epsilon={eps:.4f} ...")
        t0  = time.time()
        col = []
        for row_idx in range(len(df)):
            paths  = all_paths[row_idx]
            scores = []
            for path_idx in range(len(paths)):
                key = (row_idx, path_idx)
                tot = total_len.get(key, 0.0)
                sat = sat_len[ei].get(key, 0.0)
                scores.append(f"{sat/tot*100:.6f}" if tot > 0 else "100.000000")
            col.append(";".join(scores))

        out_df = df.copy()
        out_df["stretch_scores"] = col
        op = output_paths[eps]
        os.makedirs(os.path.dirname(os.path.abspath(op)), exist_ok=True)
        out_df.to_csv(op, index=False)
        print(f"  Saved: {op}  ({time.time()-t0:.1f}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Re-score stretch_scores (streaming + threading, memory-safe).")
    parser.add_argument("--data_dir",    required=True)
    parser.add_argument("--paths_path",  required=True)
    parser.add_argument("--epsilon",     type=float, default=None)
    parser.add_argument("--output_path", type=str,   default=None)
    parser.add_argument("--epsilons",    type=float, nargs="+", default=None)
    parser.add_argument("--output_dir",  type=str,   default=None)
    parser.add_argument("--batch_size",  type=int,   default=200)
    parser.add_argument("--n_workers",   type=int,   default=8,
                        help="Thread workers for parallel Dijkstra. Default 8.")
    args = parser.parse_args()

    if args.epsilons is not None:
        epsilon_list = args.epsilons
    elif args.epsilon is not None:
        epsilon_list = [args.epsilon]
    else:
        parser.error("Provide --epsilon or --epsilons.")

    input_csv = os.path.join(args.data_dir, args.paths_path)
    if not os.path.isfile(input_csv):
        input_csv = args.paths_path
    if not os.path.isfile(input_csv):
        print(f"ERROR: input CSV not found: {input_csv}"); sys.exit(1)

    if len(epsilon_list) > 1 or args.epsilons is not None:
        out_dir = args.output_dir or os.path.join(args.data_dir, "graph")
        base = os.path.splitext(os.path.basename(input_csv))[0]
        output_paths = {eps: os.path.join(out_dir, f"{base}_eps{eps:.2f}.csv")
                        for eps in epsilon_list}
    else:
        if args.output_path:
            output_paths = {epsilon_list[0]: args.output_path}
        else:
            base = os.path.splitext(input_csv)[0]
            output_paths = {epsilon_list[0]: f"{base}_eps{epsilon_list[0]:.2f}.csv"}

    print(f"Input CSV : {input_csv}")
    print(f"Epsilon(s): {epsilon_list}")
    for eps, op in output_paths.items():
        print(f"  eps={eps:.4f} -> {op}")

    print("\nLoading graph ...")
    t0 = time.time()
    id_to_idx, csr, edge_len = build_graph(args.data_dir)
    print(f"  {len(id_to_idx):,} nodes, {len(edge_len):,} edges  ({time.time()-t0:.1f}s)")

    print("\nLoading paths CSV ...")
    df = pd.read_csv(input_csv)
    print(f"  {len(df):,} OD pairs")

    print("\nPass 1: parsing paths and indexing source usages ...")
    t0 = time.time()
    all_paths, source_usages = collect_source_usages(df, id_to_idx)
    n_usages = sum(len(v) for v in source_usages.values())
    print(f"  {len(source_usages):,} unique source nodes, "
          f"{n_usages:,} usage entries  ({time.time()-t0:.1f}s)")

    print(f"\nPass 2: streaming Dijkstra -> accumulators "
          f"({args.n_workers} thread(s), batch={args.batch_size}) ...")
    total_len, sat_len = compute_scores_streaming(
        all_paths, source_usages, edge_len, id_to_idx, csr,
        epsilon_list, n_workers=args.n_workers, batch_size=args.batch_size)

    print("\nPass 3: writing output CSVs ...")
    write_output_csvs(df, all_paths, total_len, sat_len, epsilon_list, output_paths)

    print("\nAll done.")


if __name__ == "__main__":
    main()
