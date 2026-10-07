"""
Generate LaTeX table from test_stretch result JSONs.

Metrics reported (all tie-friendly):
  - Spearman ρ
  - Top-3 item accuracy  (tie-friendly)
  - Top-5 item accuracy  (tie-friendly)
  - Item accuracy        (tie-friendly)
    - RDA@1                (tie-friendly)
"""

import json
import os

CITIES = [
    "amsterdam", "bangalore", "london", "manchester", "manhattan",
    "paris", "rome", "seattle", "singapore", "sydney", "tokyo",
]
VARIANTS = ["ksp", "plateau_ub_2_dm_4_sim_07", "randomwalk", "stochastic_mu15"]

# Display labels for the table
CITY_LABELS = {
    "amsterdam": "Amsterdam",
    "bangalore": "Bangalore",
    "london":    "London",
    "manchester":"Manchester",
    "manhattan": "Manhattan",
    "paris":     "Paris",
    "rome":      "Rome",
    "seattle":   "Seattle",
    "singapore": "Singapore",
    "sydney":    "Sydney",
    "tokyo":     "Tokyo",
}

VARIANT_LABELS = {
    "ksp":                      r"\ksp{}",
    "plateau_ub_2_dm_4_sim_07": r"\plateau{}",
    "randomwalk":               r"\rw{}",
    "stochastic_mu15":          r"\stoch{}",
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_result(city, variant):
    path = os.path.join(
        DATA_DIR, city, "results",
        f"stretch_ml_best_{city}_{variant}_result.json"
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(val, pct=False, decimals=3):
    """Format a float. pct=True multiplies by 100 and appends %."""
    if val is None:
        return "--"
    if pct:
        return f"{val * 100:.1f}"
    return f"{val:.{decimals}f}"


# ── Table 1: grouped by city, one row per variant ─────────────────────────────
def make_table_by_city():
    col_spec = "ll" + "r" * 5
    header = (
        r"\begin{tabular}{" + col_spec + r"}" + "\n"
        r"\toprule" + "\n"
        r"City & Variant "
        r"& $\rho_S$ "
        r"& Top-3 Acc$^\dag$ "
        r"& Top-5 Acc$^\dag$ "
        r"& Item Acc$^\dag$ "
        r"& RDA@1$^\dag$ \\" + "\n"
        r"& & & (\%) & (\%) & (\%) & (\%) \\" + "\n"
        r"\midrule"
    )
    rows = []
    for city in CITIES:
        first = True
        for variant in VARIANTS:
            r = load_result(city, variant)
            varlabel = VARIANT_LABELS[variant]
            if r is None:
                row = f"  & {varlabel} & -- & -- & -- & -- & -- \\\\"
            else:
                first_col = r"\multirow{4}{*}{" + CITY_LABELS[city] + "}" if first else ""
                row = (
                    f"  {first_col}"
                    f" & {varlabel}"
                    f" & {fmt(r['spearman_corr'])}"
                    f" & {fmt(r['item_top3_accuracy_with_ties'], pct=True)}"
                    f" & {fmt(r['item_top5_accuracy_with_ties'], pct=True)}"
                    f" & {fmt(r['item_accuracy_with_ties'], pct=True)}"
                    f" & {fmt(r['paths_within_1_rank_ties'], pct=True)}"
                    r" \\"
                )
            rows.append(row)
            first = False
        rows.append(r"  \midrule")

    # remove trailing \midrule
    if rows and rows[-1].strip() == r"\midrule":
        rows[-1] = r"  \bottomrule"

    footer = (
        r"\end{tabular}"
    )

    note = (
        r"\smallskip"
        "\n"
        r"{\small $\dag$~Tie-friendly: paths with equal ground-truth scores are "
        r"all considered correctly ranked.}"
    )

    return "\n".join([header] + rows + [footer, note])


# ── Table 2: one table per metric, cities as rows, variants as columns ─────────
def make_table_by_metric(metric_key, metric_label, pct=False):
    col_spec = "l" + "r" * len(VARIANTS)
    variant_header = " & ".join(VARIANT_LABELS[v] for v in VARIANTS)
    header = (
        r"\begin{tabular}{" + col_spec + r"}" + "\n"
        r"\toprule" + "\n"
        f"City & {variant_header} \\\\\n"
        r"\midrule"
    )
    rows = []
    for city in CITIES:
        vals = []
        raw_vals = []
        for variant in VARIANTS:
            r = load_result(city, variant)
            raw_vals.append(r[metric_key] if r else None)

        # bold the best value per row
        valid = [v for v in raw_vals if v is not None]
        best = max(valid) if valid else None

        for v in raw_vals:
            cell = fmt(v, pct=pct)
            if v is not None and best is not None and abs(v - best) < 1e-9:
                cell = r"\textbf{" + cell + "}"
            vals.append(cell)

        row = f"  {CITY_LABELS[city]} & " + " & ".join(vals) + r" \\"
        rows.append(row)

    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    return "\n".join([header] + rows + [footer])


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "stretch_results_tables.tex")

    metrics = [
        ("spearman_corr",          r"Spearman $\rho$",         False),
        ("item_top3_accuracy_with_ties", r"Top-3 Item Accuracy$^\dag$ (\%)", True),
        ("item_top5_accuracy_with_ties", r"Top-5 Item Accuracy$^\dag$ (\%)", True),
        ("item_accuracy_with_ties",r"Item Accuracy$^\dag$ (\%)",True),
        ("paths_within_1_rank_ties",r"RDA@1$^\dag$ (\%)",True),
    ]

    lines = []
    lines.append("% ================================================================")
    lines.append("% Auto-generated by analysis/generate_latex_table.py")
    lines.append("% Requires: \\usepackage{booktabs,multirow}")
    lines.append("% ================================================================")
    lines.append("")

    # Main combined table
    lines.append("% --- Combined table (city x variant, all metrics) ---")
    lines.append(make_table_by_city())
    lines.append("")

    # Per-metric tables (city rows, variant columns)
    for key, label, pct in metrics:
        lines.append(f"% --- Per-metric table: {label} ---")
        lines.append(f"% {label}")
        lines.append(make_table_by_metric(key, label, pct))
        lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Tables written to: {out_path}")

    # Also print a quick summary to stdout
    print("\n=== Quick summary ===")
    print(f"{'City':<12} {'Variant':<34} {'Spearman':>9} {'Top3%':>7} {'Top5%':>7} {'ItemAcc%':>9} {'RDA@1%':>8}")
    print("-" * 90)
    for city in CITIES:
        for variant in VARIANTS:
            r = load_result(city, variant)
            if r is None:
                print(f"{CITY_LABELS[city]:<12} {variant:<34}  {'--':>9}")
                continue
            print(
                f"{CITY_LABELS[city]:<12} {variant:<34}"
                f"  {r['spearman_corr']:>9.4f}"
                f"  {r['item_top3_accuracy_with_ties']*100:>6.1f}"
                f"  {r['item_top5_accuracy_with_ties']*100:>6.1f}"
                f"  {r['item_accuracy_with_ties']*100:>8.1f}"
                f"  {r['paths_within_1_rank_ties']*100:>7.1f}"
            )
