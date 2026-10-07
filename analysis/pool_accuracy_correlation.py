"""
Analyse correlation between pool_accuracy and model ranking accuracy
across 11 cities x 4 generators.

Pool accuracy data:
    - pool_accuracy_all_cities.json
    - pool_accuracy_additional_cities.json

Model accuracy data: hardcoded from paper's city-level Top-3 and tie-aware RDA@1 results.
"""

import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import pearsonr, spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--single-column", action="store_true",
                    help="Generate a portrait figure for a one-column paper layout.")
parser.add_argument("--output", default=None,
                    help="Output image path; defaults to the existing wide figure or a portrait variant.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Model accuracy from the paper's city-level ranking results.
# Keys: (city, variant)  Values: (top3_pct, rda1_pct)
# ---------------------------------------------------------------------------
MODEL_ACC = {
    ("amsterdam", "ksp"):                   (95.5, 95.9),
    ("amsterdam", "plateau_ub_2_dm_4_sim_07"): (85.8, 89.4),
    ("amsterdam", "randomwalk"):            (80.8, 89.7),
    ("amsterdam", "stochastic_mu15"):       (81.0, 88.7),

    ("bangalore", "ksp"):                   (96.4, 96.8),
    ("bangalore", "plateau_ub_2_dm_4_sim_07"): (85.4, 87.0),
    ("bangalore", "randomwalk"):            (70.5, 80.8),
    ("bangalore", "stochastic_mu15"):       (68.6, 75.5),

    ("london",    "ksp"):                   (96.4, 97.0),
    ("london",    "plateau_ub_2_dm_4_sim_07"): (88.5, 88.1),
    ("london",    "randomwalk"):            (81.3, 89.1),
    ("london",    "stochastic_mu15"):       (79.1, 87.8),

    ("manchester", "ksp"):                   (95.7, 96.3),
    ("manchester", "plateau_ub_2_dm_4_sim_07"): (80.5, 82.1),
    ("manchester", "randomwalk"):            (82.2, 90.6),
    ("manchester", "stochastic_mu15"):       (76.8, 86.5),

    ("manhattan", "ksp"):                   (98.0, 98.2),
    ("manhattan", "plateau_ub_2_dm_4_sim_07"): (94.3, 94.5),
    ("manhattan", "randomwalk"):            (76.5, 84.4),
    ("manhattan", "stochastic_mu15"):       (75.2, 79.6),

    ("paris",     "ksp"):                   (93.6, 94.4),
    ("paris",     "plateau_ub_2_dm_4_sim_07"): (88.5, 88.5),
    ("paris",     "randomwalk"):            (76.6, 86.1),
    ("paris",     "stochastic_mu15"):       (79.2, 88.7),

    ("rome",      "ksp"):                   (94.3, 95.2),
    ("rome",      "plateau_ub_2_dm_4_sim_07"): (83.6, 85.1),
    ("rome",      "randomwalk"):            (82.1, 89.1),
    ("rome",      "stochastic_mu15"):       (82.2, 90.2),

    ("seattle",   "ksp"):                   (96.6, 96.9),
    ("seattle",   "plateau_ub_2_dm_4_sim_07"): (90.4, 90.6),
    ("seattle",   "randomwalk"):            (70.7, 78.4),
    ("seattle",   "stochastic_mu15"):       (67.3, 69.3),

    ("singapore", "ksp"):                   (93.6, 94.5),
    ("singapore", "plateau_ub_2_dm_4_sim_07"): (83.0, 85.1),
    ("singapore", "randomwalk"):            (78.5, 87.2),
    ("singapore", "stochastic_mu15"):       (83.5, 91.0),

    ("sydney",    "ksp"):                   (93.3, 94.0),
    ("sydney",    "plateau_ub_2_dm_4_sim_07"): (84.5, 85.7),
    ("sydney",    "randomwalk"):            (79.4, 86.8),
    ("sydney",    "stochastic_mu15"):       (76.7, 84.0),

    ("tokyo",     "ksp"):                   (92.7, 93.9),
    ("tokyo",     "plateau_ub_2_dm_4_sim_07"): (82.3, 87.3),
    ("tokyo",     "randomwalk"):            (67.3, 77.8),
    ("tokyo",     "stochastic_mu15"):       (65.7, 71.5),
}

VARIANT_LABELS = {
    "ksp":                      "KSP",
    "plateau_ub_2_dm_4_sim_07": "Plateau",
    "randomwalk":               "Randomwalk",
    "stochastic_mu15":          "Stochastic",
}

VARIANT_COLORS = {
    "ksp":                      "#1f77b4",   # blue
    "plateau_ub_2_dm_4_sim_07": "#ff7f0e",   # orange
    "randomwalk":               "#2ca02c",   # green
    "stochastic_mu15":          "#d62728",   # red
}

CITY_MARKERS = {
    "amsterdam": "o",
    "bangalore": "s",
    "london":    "^",
    "manchester": "D",
    "manhattan": "P",
    "paris":     "X",
    "rome":      "v",
    "seattle":   "<",
    "singapore": ">",
    "sydney":    "*",
    "tokyo":     "h",
}

CITY_ORDER = [
    "amsterdam", "bangalore", "london", "manchester", "manhattan",
    "paris", "rome", "seattle", "singapore", "sydney", "tokyo",
]

# ---------------------------------------------------------------------------
# Load pool accuracy
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def load_pool_data() -> list:
    data = []
    for filename in ["pool_accuracy_all_cities.json", "pool_accuracy_additional_cities.json"]:
        path = os.path.join(REPO_ROOT, filename)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                data.extend(json.load(f))
            print(f"Loaded {filename}")
        else:
            print(f"[WARN] Missing file: {path}")
    return data


pool_data = load_pool_data()

records = []
for entry in pool_data:
    city    = entry["city"]
    variant = entry["variant"]
    pool_acc = entry["pool_accuracy"] * 100  # convert to %
    key = (city, variant)
    if key in MODEL_ACC:
        top3, rda1 = MODEL_ACC[key]
        records.append({
            "city":     city,
            "variant":  variant,
            "pool_acc": pool_acc,
            "top3":     top3,
            "rda1":     rda1,
        })

# ---------------------------------------------------------------------------
# Correlation statistics
# ---------------------------------------------------------------------------
pool_vals = np.array([r["pool_acc"] for r in records])
top3_vals = np.array([r["top3"]     for r in records])
rda1_vals = np.array([r["rda1"] for r in records])

pr_top3,  pv_top3  = pearsonr(pool_vals, top3_vals)
sr_top3,  sv_top3  = spearmanr(pool_vals, top3_vals)
pr_rda1,  pv_rda1  = pearsonr(pool_vals, rda1_vals)
sr_rda1,  sv_rda1  = spearmanr(pool_vals, rda1_vals)

print("=" * 60)
print("Pool Accuracy vs Model Accuracy - Correlation Analysis")
print("=" * 60)
print(f"\nN = {len(records)} data points (11 cities x 4 generators)\n")

print("--- Top-3 Accuracy ---")
print(f"  Pearson  r = {pr_top3:.4f}  (p = {pv_top3:.4e})")
print(f"  Spearman rho = {sr_top3:.4f}  (p = {sv_top3:.4e})")

print("\n--- RDA@1 ---")
print(f"  Pearson  r = {pr_rda1:.4f}  (p = {pv_rda1:.4e})")
print(f"  Spearman rho = {sr_rda1:.4f}  (p = {sv_rda1:.4e})")

print("\n--- Per-Variant Breakdown ---")
for v in ["ksp", "plateau_ub_2_dm_4_sim_07", "randomwalk", "stochastic_mu15"]:
    sub = [r for r in records if r["variant"] == v]
    mean_pool = np.mean([r["pool_acc"] for r in sub])
    mean_top3 = np.mean([r["top3"]     for r in sub])
    mean_rda1 = np.mean([r["rda1"] for r in sub])
    print(f"  {VARIANT_LABELS[v]:12s}  pool={mean_pool:.1f}%  top3={mean_top3:.1f}%  RDA@1={mean_rda1:.1f}%")

print("\n--- Per-City Breakdown ---")
for city in CITY_ORDER:
    sub = [r for r in records if r["city"] == city]
    if not sub:
        continue
    mean_pool = np.mean([r["pool_acc"] for r in sub])
    mean_top3 = np.mean([r["top3"]     for r in sub])
    print(f"  {city:12s}  avg_pool={mean_pool:.1f}%  avg_top3={mean_top3:.1f}%")

metrics = [
    ("Top-3 Accuracy (%)", top3_vals, pr_top3, sr_top3),
    ("RDA@1 (%)", rda1_vals, pr_rda1, sr_rda1),
]

generator_keys = ["ksp", "plateau_ub_2_dm_4_sim_07", "randomwalk", "stochastic_mu15"]
gen_patches = [
    mpatches.Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
    for v in generator_keys
]
city_handles = [
    plt.scatter([], [], marker=CITY_MARKERS[c], color="gray",
                label=c.capitalize(), s=24)
    for c in CITY_ORDER
]

if args.single_column:
    fig, axes = plt.subplots(2, 1, figsize=(3.0, 3.25), sharex=True)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.88, bottom=0.14, hspace=0.15)

    for ax, (ylabel, y_vals, pr, sr) in zip(axes, metrics):
        for record, y in zip(records, y_vals):
            ax.scatter(
                record["pool_acc"], y,
                color=VARIANT_COLORS[record["variant"]],
                marker="o", s=17, zorder=3, edgecolors="white", linewidths=0.3,
            )

        slope, intercept = np.polyfit(pool_vals, y_vals, 1)
        xs = np.linspace(pool_vals.min() - 2, pool_vals.max() + 2, 100)
        ax.plot(xs, slope * xs + intercept, "k--", linewidth=0.9, alpha=0.7)
        ax.set_ylabel(ylabel, fontsize=7.2)
        ax.tick_params(axis="both", labelsize=6.2, pad=2)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.text(
            0.04, 0.95,
            f"Pearson $r$={pr:.3f}\nSpearman $\\rho$={sr:.3f}",
            transform=ax.transAxes, fontsize=6.2,
            verticalalignment="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )

    axes[-1].set_xlabel("Distance-pool accuracy (%)", fontsize=7.2, labelpad=2)
    axes[-1].set_xlim(pool_vals.min() - 3, pool_vals.max() + 3)
    fig.legend(handles=gen_patches, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=4, fontsize=6.2, frameon=False, columnspacing=0.65,
               handlelength=0.9, handletextpad=0.3)
    out_path = args.output or os.path.join(
        REPO_ROOT, "figures", "pool_accuracy_correlation_single_column.png"
    )
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Pool Accuracy vs Model Ranking Accuracy\n(11 cities × 4 generators, N=44)",
                 fontsize=13, fontweight="bold")

    for ax, (ylabel, y_vals, pr, sr) in zip(axes, metrics):
        for record, y in zip(records, y_vals):
            ax.scatter(
                record["pool_acc"], y,
                color=VARIANT_COLORS[record["variant"]],
                marker=CITY_MARKERS[record["city"]],
                s=90, zorder=3, edgecolors="white", linewidths=0.5,
            )

        slope, intercept = np.polyfit(pool_vals, y_vals, 1)
        xs = np.linspace(pool_vals.min() - 2, pool_vals.max() + 2, 200)
        ax.plot(xs, slope * xs + intercept, "k--", linewidth=1.2, alpha=0.6,
                label="_nolegend_")
        ax.set_xlabel("Pool Accuracy (%)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.text(
            0.05, 0.95,
            f"Pearson r = {pr:.3f}\nSpearman ρ = {sr:.3f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].legend(handles=gen_patches, title="Generator", fontsize=8,
                   title_fontsize=8, loc="lower right")
    axes[1].legend(handles=city_handles, title="City", fontsize=8,
                   title_fontsize=8, loc="lower right")
    plt.tight_layout()
    out_path = args.output or os.path.join(REPO_ROOT, "figures", "pool_accuracy_correlation.png")

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
fig.savefig(out_path, dpi=300 if args.single_column else 150, bbox_inches="tight")
print(f"\nPlot saved to {out_path}")
if not args.single_column:
    plt.show()
