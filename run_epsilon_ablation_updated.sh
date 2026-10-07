#!/bin/bash
# Epsilon sensitivity ablation: end-to-end pipeline for London
#
# Phase 1 (priority): KSP + plateau            eps={0.01, 0.025, 0.15} new -> rescore -> train -> test
#                                               eps={0.05, 0.20} existing -> test only
# Phase 2:            randomwalk + stochastic_mu15  eps={0.01, 0.025, 0.15} new -> rescore -> train -> test
#                                                    eps={0.05, 0.20} existing -> test only
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
EPSILONS="0.01 0.025 0.15"        # new only; 0.05 and 0.20 CSVs already exist on server

mkdir -p "$LOG_DIR"
mkdir -p "${DATA_DIR}/models"
mkdir -p "${GRAPH_DIR}"

# ---------------------------------------------------------------------------
# Helper: rescore a CSV for both epsilons in one Dijkstra pass
# Skips if all output CSVs already exist.
# ---------------------------------------------------------------------------
rescore() {
    local paths_path="$1"   # relative path inside graph dir, e.g. paths_scored_step2_london_ksp.csv
    local label="$2"
    local base="${paths_path%.csv}"
    # Check if all rescored CSVs already exist
    local all_exist=true
    for eps in ${EPSILONS}; do
        local out_csv="${GRAPH_DIR}/${base}_eps${eps}.csv"
        if [ ! -f "${out_csv}" ]; then
            all_exist=false
            break
        fi
    done
    if [ "${all_exist}" = true ]; then
        echo ""
        echo "SKIP RESCORE: ${label} (all output CSVs already exist)"
        return 0
    fi
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
# Helper: test one model (no skip — called only when needed via run_eps)
# ---------------------------------------------------------------------------
test_model() {
    local csv="$1"          # filename only (no dir), must exist in graph/
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
# Helper: train one model (no skip — called only when needed via run_eps)
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

# ---------------------------------------------------------------------------
# Helper: run train+test as a unit.
# Skips BOTH only if model .pth AND result .json already exist.
# If either is missing the whole unit (train + test) is re-run.
# ---------------------------------------------------------------------------
run_eps() {
    local csv="$1"          # filename only (no dir), must exist in graph/
    local suffix="$2"
    local eps="$3"
    local batch="$4"
    local result_json="${DATA_DIR}/results/stretch_ml_best${suffix}_result.json"
    if [ -f "${result_json}" ]; then
        echo ""
        echo "SKIP [${suffix}]: result already exists"
        return 0
    fi
    train "${csv}" "${suffix}" "${eps}" "${batch}"
    test_model "${csv}" "${suffix}"
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
echo "# PHASE 1b: Train+Test London KSP all epsilons"
echo "######################################################################"
run_eps "paths_scored_step2_london_ksp_eps0.01.csv" "_london_ksp_eps0.01"  0.01  16
run_eps "paths_scored_step2_london_ksp_eps0.03.csv" "_london_ksp_eps0.025" 0.025 16
run_eps "paths_scored_step2_london_ksp_eps0.15.csv" "_london_ksp_eps0.15"  0.15  16

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
echo "# PHASE 1d: Train+Test London plateau all epsilons"
echo "######################################################################"
run_eps "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.01.csv" "_london_plateau_eps0.01"  0.01  32
run_eps "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.03.csv" "_london_plateau_eps0.025" 0.025 32
run_eps "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.15.csv" "_london_plateau_eps0.15"  0.15  32

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
echo "# PHASE 2b: Train+Test London randomwalk all epsilons"
echo "######################################################################"
run_eps "paths_scored_step2_london_randomwalk_eps0.01.csv" "_london_randomwalk_eps0.01"  0.01  32
run_eps "paths_scored_step2_london_randomwalk_eps0.03.csv" "_london_randomwalk_eps0.025" 0.025 32
run_eps "paths_scored_step2_london_randomwalk_eps0.15.csv" "_london_randomwalk_eps0.15"  0.15  32

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
echo "# PHASE 2d: Train+Test London stochastic_mu15 all epsilons"
echo "######################################################################"
run_eps "paths_scored_step2_london_stochastic_mu15_eps0.01.csv" "_london_stochastic_mu15_eps0.01"  0.01  32
run_eps "paths_scored_step2_london_stochastic_mu15_eps0.03.csv" "_london_stochastic_mu15_eps0.025" 0.025 32
run_eps "paths_scored_step2_london_stochastic_mu15_eps0.15.csv" "_london_stochastic_mu15_eps0.15"  0.15  32

echo ""
echo "######################################################################"
echo "# VERIFICATION: Check all expected outputs"
echo "######################################################################"

GENERATORS=("ksp" "plateau" "randomwalk" "stochastic_mu15")
GEN_LABELS=("ksp" "plateau" "randomwalk" "stochastic_mu15")
EPSILONS_LIST=("0.01" "0.025" "0.15")

MISSING=0

for i in "${!GENERATORS[@]}"; do
    gen="${GENERATORS[$i]}"
    label="${GEN_LABELS[$i]}"
    for eps in "${EPSILONS_LIST[@]}"; do
        suffix="_london_${gen}_eps${eps}"
        pth="${DATA_DIR}/models/stretch_ml_best${suffix}.pth"
        result="${DATA_DIR}/results/stretch_ml_best${suffix}_result.json"
        ok=1
        if [ ! -f "${result}" ]; then
            echo "MISSING RESULT: ${result}"
            ok=0
            MISSING=$((MISSING + 1))
        fi
        if [ "${ok}" -eq 1 ]; then
            echo "OK  [${label} eps=${eps}]"
        fi
    done
done

echo ""
if [ "${MISSING}" -eq 0 ]; then
    echo "All outputs are present; run complete."
else
    echo "INCOMPLETE: ${MISSING} file(s) missing. Re-run this script to resume."
fi

echo ""
echo "######################################################################"
echo "# ALL PHASES COMPLETE"
echo "######################################################################"
