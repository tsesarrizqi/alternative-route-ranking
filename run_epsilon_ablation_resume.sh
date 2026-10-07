#!/bin/bash
# Resume script for epsilon ablation — picks up from where run_epsilon_ablation_updated.sh stopped.
#
# Assumes already completed:
#   - Phase 1a: KSP rescore
#   - Phase 1b: KSP train + test (all 5 epsilons)
#   - Phase 1c: Plateau rescore (all CSVs exist)
#   - Phase 1d: Plateau train eps0.01 (model saved, not yet tested)
#
# This script completes:
#   - Phase 1d: Plateau train eps0.025 and eps0.15
#   - Phase 1d (test): Plateau test all 5 epsilons (including eps0.01)
#   - Phase 2: Randomwalk rescore + train + test
#   - Phase 2: Stochastic rescore + train + test
#
# Usage (background):
#   nohup bash run_epsilon_ablation_resume.sh > logs/epsilon_ablation_resume.log 2>&1 &

set -euo pipefail

CITY="london"
DATA_DIR="data/${CITY}"
GRAPH_DIR="${DATA_DIR}/graph"
LOG_DIR="logs/epsilon_ablation"
EPSILONS="0.01 0.025 0.15"

mkdir -p "$LOG_DIR"
mkdir -p "${DATA_DIR}/models"
mkdir -p "${GRAPH_DIR}"

# ---------------------------------------------------------------------------
# Helper: rescore a CSV for new epsilons in one Dijkstra pass
# ---------------------------------------------------------------------------
rescore() {
    local paths_path="$1"
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
# Helper: test one model
# ---------------------------------------------------------------------------
test_model() {
    local csv="$1"
    local suffix="$2"
    local log="${LOG_DIR}/test${suffix}.log"
    echo ""
    echo "=========================================="
    echo "TEST: ${suffix}"
    echo "=========================================="
    python -u test_stretch.py \
        --data_dir "${DATA_DIR}" \
        --paths_path "${csv}" \
        --model_suffix "${suffix}" \
        --batch_size 32 \
        2>&1 | tee "${log}"
}

# ---------------------------------------------------------------------------
# Helper: train one model
# ---------------------------------------------------------------------------
train() {
    local csv="$1"
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
# PHASE 1d (resume): Train remaining Plateau epsilons
# ===========================================================================
echo ""
echo "######################################################################"
echo "# PHASE 1d (resume): Train London plateau eps=0.025 and eps=0.15"
echo "######################################################################"
train "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.03.csv" "_london_plateau_eps0.025" 0.025 32
train "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.15.csv" "_london_plateau_eps0.15"  0.15  32

echo ""
echo "######################################################################"
echo "# PHASE 1d (test): Test London plateau all epsilons (including eps0.01)"
echo "######################################################################"
test_model "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.01.csv" "_london_plateau_eps0.01"
test_model "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.03.csv" "_london_plateau_eps0.025"
test_model "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.05.csv" "_london_plateau_eps0.05"
test_model "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.15.csv" "_london_plateau_eps0.15"
test_model "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.20.csv" "_london_plateau_eps0.20"

echo ""
echo "######################################################################"
echo "# PHASE 1 COMPLETE"
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
echo "# PHASE 2b: Train London randomwalk  eps=0.01 0.025 0.15"
echo "######################################################################"
train "paths_scored_step2_london_randomwalk_eps0.01.csv" "_london_randomwalk_eps0.01"  0.01  32
train "paths_scored_step2_london_randomwalk_eps0.03.csv" "_london_randomwalk_eps0.025" 0.025 32
train "paths_scored_step2_london_randomwalk_eps0.15.csv" "_london_randomwalk_eps0.15"  0.15  32

echo ""
echo "######################################################################"
echo "# PHASE 2b (test): Test London randomwalk all epsilons"
echo "######################################################################"
test_model "paths_scored_step2_london_randomwalk_eps0.01.csv" "_london_randomwalk_eps0.01"
test_model "paths_scored_step2_london_randomwalk_eps0.03.csv" "_london_randomwalk_eps0.025"
test_model "paths_scored_step2_london_randomwalk_eps0.05.csv" "_london_randomwalk_eps0.05"
test_model "paths_scored_step2_london_randomwalk_eps0.15.csv" "_london_randomwalk_eps0.15"
test_model "paths_scored_step2_london_randomwalk_eps0.20.csv" "_london_randomwalk_eps0.20"

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
echo "# PHASE 2d: Train London stochastic_mu15  eps=0.01 0.025 0.15"
echo "######################################################################"
train "paths_scored_step2_london_stochastic_mu15_eps0.01.csv" "_london_stochastic_mu15_eps0.01"  0.01  32
train "paths_scored_step2_london_stochastic_mu15_eps0.03.csv" "_london_stochastic_mu15_eps0.025" 0.025 32
train "paths_scored_step2_london_stochastic_mu15_eps0.15.csv" "_london_stochastic_mu15_eps0.15"  0.15  32

echo ""
echo "######################################################################"
echo "# PHASE 2d (test): Test London stochastic_mu15 all epsilons"
echo "######################################################################"
test_model "paths_scored_step2_london_stochastic_mu15_eps0.01.csv" "_london_stochastic_mu15_eps0.01"
test_model "paths_scored_step2_london_stochastic_mu15_eps0.03.csv" "_london_stochastic_mu15_eps0.025"
test_model "paths_scored_step2_london_stochastic_mu15_eps0.05.csv" "_london_stochastic_mu15_eps0.05"
test_model "paths_scored_step2_london_stochastic_mu15_eps0.15.csv" "_london_stochastic_mu15_eps0.15"
test_model "paths_scored_step2_london_stochastic_mu15_eps0.20.csv" "_london_stochastic_mu15_eps0.20"

echo ""
echo "######################################################################"
echo "# ALL PHASES COMPLETE"
echo "######################################################################"
