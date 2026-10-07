"""
Produce two alternative path orderings per city/variant for baseline comparison:

1. oracle_order   — paths sorted by stretch_score descending (upper bound)
2. generator_order — paths in original generator output order, with stretch scores joined

Usage:
    python prepare_baseline_orderings.py
"""

import csv
import io
import os

CITIES = ["london", "manhattan", "paris", "singapore", "tokyo"]

# (variant_key, raw_file_template, scored_file_template, out_suffix)
VARIANTS = [
    (
        "ksp",
        "original_candidate_paths/ksp/paths_raw_step1_{city}_ksp.csv",
        "data/{city}/graph/paths_scored_step2_{city}_ksp.csv",
        "ksp",
    ),
    (
        "plateau",
        "original_candidate_paths/plateau/paths_raw_step1_{city}_ub_2_dm_4_sim_07.csv",
        "data/{city}/graph/paths_scored_step2_paths_raw_step1_{city}_ub_2_dm_4_sim_07.csv",
        "plateau_ub_2_dm_4_sim_07",
    ),
    (
        "randomwalk",
        "original_candidate_paths/random/paths_raw_step1_{city}_randomwalk.csv",
        "data/{city}/graph/paths_scored_step2_{city}_randomwalk.csv",
        "randomwalk",
    ),
    (
        "stochastic_mu15",
        "original_candidate_paths/stochastic/paths_raw_step1_{city}_stochastic_mu15.csv",
        "data/{city}/graph/paths_scored_step2_{city}_stochastic_mu15.csv",
        "stochastic_mu15",
    ),
]

SCORE_COLS = ["final_scores", "sharing_scores", "stretch_scores", "locality_scores"]
ALL_OUT_COLS = [
    "source", "destination", "paths",
    "final_scores", "sharing_scores", "stretch_scores", "locality_scores",
    "bucket_id", "optimal_edges", "optimal_distance",
]


def parse_paths(paths_str):
    return paths_str.split(";")


def parse_scores(scores_str):
    return [float(x) for x in scores_str.split(";")]


def join_scores_to_raw_order(raw_paths, scored_row):
    """
    Given a list of path strings in raw (generator) order, and the scored row
    (which has the same paths in a different order with scores attached),
    return per-path score dicts in raw order.
    """
    scored_paths = parse_paths(scored_row["paths"])
    scored_scores = {col: parse_scores(scored_row[col]) for col in SCORE_COLS}

    # Build lookup: path_string -> index in scored list
    path_to_idx = {p: i for i, p in enumerate(scored_paths)}

    reordered = {col: [] for col in SCORE_COLS}
    missing = []
    for p in raw_paths:
        if p in path_to_idx:
            idx = path_to_idx[p]
            for col in SCORE_COLS:
                reordered[col].append(scored_scores[col][idx])
        else:
            missing.append(p)

    if missing:
        print(f"    WARNING: {len(missing)} paths not found in scored file")

    return reordered, missing


def write_csv(out_path, rows):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ALL_OUT_COLS)
        w.writeheader()
        w.writerows(rows)


def process(city, raw_tpl, scored_tpl, out_suffix):
    raw_path = raw_tpl.format(city=city)
    scored_path = scored_tpl.format(city=city)

    if not os.path.exists(raw_path):
        print(f"  [SKIP] raw file not found: {raw_path}")
        return
    if not os.path.exists(scored_path):
        print(f"  [SKIP] scored file not found: {scored_path}")
        return

    # Load scored file indexed by (source, destination)
    scored_index = {}
    with open(scored_path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["source"], row["destination"])
            scored_index[key] = row

    oracle_rows = []
    genorder_rows = []

    with open(raw_path, newline="") as f:
        for raw_row in csv.DictReader(f):
            key = (raw_row["source"], raw_row["destination"])
            if key not in scored_index:
                print(f"  WARNING: OD pair {key} missing from scored file, skipping")
                continue

            scored_row = scored_index[key]
            raw_paths = parse_paths(raw_row["paths"])

            # --- Generator Order: raw order + joined scores ---
            reordered_scores, missing = join_scores_to_raw_order(raw_paths, scored_row)
            if missing:
                # Drop unmatched paths
                matched_paths = [p for p in raw_paths if p in {pp: None for pp in parse_paths(scored_row["paths"])}]
            else:
                matched_paths = raw_paths

            genorder_rows.append({
                "source": raw_row["source"],
                "destination": raw_row["destination"],
                "paths": ";".join(matched_paths),
                **{col: ";".join(f"{v:.6f}" for v in reordered_scores[col]) for col in SCORE_COLS},
                "bucket_id": scored_row["bucket_id"],
                "optimal_edges": scored_row["optimal_edges"],
                "optimal_distance": scored_row["optimal_distance"],
            })

            # --- Oracle Order: sort by stretch_score descending ---
            scored_paths = parse_paths(scored_row["paths"])
            all_scores = {col: parse_scores(scored_row[col]) for col in SCORE_COLS}
            n = len(scored_paths)
            order = sorted(range(n), key=lambda i: all_scores["stretch_scores"][i], reverse=True)

            oracle_rows.append({
                "source": scored_row["source"],
                "destination": scored_row["destination"],
                "paths": ";".join(scored_paths[i] for i in order),
                **{col: ";".join(f"{all_scores[col][i]:.6f}" for i in order) for col in SCORE_COLS},
                "bucket_id": scored_row["bucket_id"],
                "optimal_edges": scored_row["optimal_edges"],
                "optimal_distance": scored_row["optimal_distance"],
            })

    out_dir = f"data/{city}/graph"
    genorder_out = f"{out_dir}/paths_genorder_{city}_{out_suffix}.csv"
    oracle_out = f"{out_dir}/paths_oracle_{city}_{out_suffix}.csv"

    write_csv(genorder_out, genorder_rows)
    write_csv(oracle_out, oracle_rows)

    print(f"  -> {genorder_out}  ({len(genorder_rows)} OD pairs)")
    print(f"  -> {oracle_out}  ({len(oracle_rows)} OD pairs)")


if __name__ == "__main__":
    for city in CITIES:
        print(f"\n=== {city} ===")
        for variant_key, raw_tpl, scored_tpl, out_suffix in VARIANTS:
            print(f"  [{variant_key}]")
            process(city, raw_tpl, scored_tpl, out_suffix)

    print("\nDone.")
