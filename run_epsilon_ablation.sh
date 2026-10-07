#!/bin/bash
# Epsilon sensitivity ablation: end-to-end pipeline for London
#
# Phase 1 (priority): KSP + plateau   eps={0.05, 0.20}  -> rescore -> train
# Phase 2:            randomwalk + stochastic_mu15  eps={0.05, 0.20}  -> rescore -> train
#
# Usage (background):
#   nohup bash run_epsilon_ablation.sh > logs/epsilon_ablation.log 2>&1 &
#
# All models saved to data/london/models/
# Rescored CSVs saved to data/london/graph/

set -euo pipefail

CITY="london"
DATA_DIR="data/${CITY}"
GRAPH_DIR="${DATA_DIR}/graph"
LOG_DIR="logs/epsilon_ablation"
EPSILONS="0.05 0.20"

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# Helper: rescore a CSV for both epsilons in one Dijkstra pass
# Input basename (no dir, no .csv) is derived from paths_path arg.
# ---------------------------------------------------------------------------
rescore() {
    local paths_path="$1"   # relative path inside graph dir, e.g. paths_scored_step2_london_ksp.csv
    local label="$2"
    echo ""
    echo "=========================================="
    echo "RESCORE: ${label}  eps={${EPSILONS}}"
    echo "=========================================="
    python -u rescore_stretch.py \
        --data_dir "${DATA_DIR}" \
        --paths_path "graph/${paths_path}" \
        --epsilons ${EPSILONS} \
        --output_dir "${GRAPH_DIR}" \
        --n_workers 8
}

# ---------------------------------------------------------------------------
# Helper: train one model
# ---------------------------------------------------------------------------
train() {
    local csv="$1"          # filename only (no dir), must exist in graph/
    local suffix="$2"
    local eps="$3"
    local batch="$4"
    local log="${LOG_DIR}/train${suffix}.log"
    echo ""
    echo "=========================================="
    echo "TRAIN: ${suffix}  eps=${eps}  batch=${batch}"
    echo "=========================================="
    python -u train_stretch.py \
        --data_dir "${DATA_DIR}" \
        --paths_path "${csv}" \
        --epochs 50 \
        --lr 0.001 \
        --batch_size "${batch}" \
        --loss listmle \
        --hidden_dim 128 \
        --use_weighted \
        --n_bins 15 \
        --weight_power 0.6 \
        --train_alpha 1 \
        --val_alpha 0.05 \
        --epsilon "${eps}" \
        --model_suffix "${suffix}" \
        2>&1 | tee "${log}"
}

# ===========================================================================
# PHASE 1: KSP
# ===========================================================================
echo ""
echo "######################################################################"
echo "# PHASE 1a: Rescore London KSP"
echo "######################################################################"
rescore "paths_scored_step2_london_ksp.csv" "london_ksp"

echo ""
echo "######################################################################"
echo "# PHASE 1b: Train London KSP  eps=0.05 and eps=0.20"
echo "######################################################################"
train "paths_scored_step2_london_ksp_eps0.05.csv" "_london_ksp_eps0.05" 0.05 16
train "paths_scored_step2_london_ksp_eps0.20.csv" "_london_ksp_eps0.20" 0.20 16

# ===========================================================================
# PHASE 1: Plateau
# ===========================================================================
echo ""
echo "######################################################################"
echo "# PHASE 1c: Rescore London plateau"
echo "######################################################################"
rescore "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07.csv" "london_plateau"

echo ""
echo "######################################################################"
echo "# PHASE 1d: Train London plateau  eps=0.05 and eps=0.20"
echo "######################################################################"
train "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.05.csv" "_london_plateau_eps0.05" 0.05 32
train "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.20.csv" "_london_plateau_eps0.20" 0.20 32

echo ""
echo "######################################################################"
echo "# PHASE 1 COMPLETE: KSP + plateau epsilon sensitivity done"
echo "######################################################################"

# ===========================================================================
# PHASE 2: Randomwalk
# ===========================================================================
echo ""
echo "######################################################################"
echo "# PHASE 2a: Rescore London randomwalk"
echo "######################################################################"
rescore "paths_scored_step2_london_randomwalk.csv" "london_randomwalk"

echo ""
echo "######################################################################"
echo "# PHASE 2b: Train London randomwalk  eps=0.05 and eps=0.20"
echo "######################################################################"
train "paths_scored_step2_london_randomwalk_eps0.05.csv" "_london_randomwalk_eps0.05" 0.05 32
train "paths_scored_step2_london_randomwalk_eps0.20.csv" "_london_randomwalk_eps0.20" 0.20 32

# ===========================================================================
# PHASE 2: Stochastic
# ===========================================================================
echo ""
echo "######################################################################"
echo "# PHASE 2c: Rescore London stochastic_mu15"
echo "######################################################################"
rescore "paths_scored_step2_london_stochastic_mu15.csv" "london_stochastic_mu15"

echo ""
echo "######################################################################"
echo "# PHASE 2d: Train London stochastic_mu15  eps=0.05 and eps=0.20"
echo "######################################################################"
train "paths_scored_step2_london_stochastic_mu15_eps0.05.csv" "_london_stochastic_mu15_eps0.05" 0.05 32
train "paths_scored_step2_london_stochastic_mu15_eps0.20.csv" "_london_stochastic_mu15_eps0.20" 0.20 32

echo ""
echo "######################################################################"
echo "# ALL PHASES COMPLETE"
echo "######################################################################"
