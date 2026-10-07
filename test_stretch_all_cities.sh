#!/bin/bash
# Run test_stretch.py for 5 cities x 4 path variants

CITIES=(
    "london"
    "manhattan"
    "paris"
    "singapore"
    "tokyo"
)

VARIANTS=(
    "ksp"
    "plateau_ub_2_dm_4_sim_07"
    "randomwalk"
    "stochastic_mu15"
)

mkdir -p test_logs

for CITY in "${CITIES[@]}"; do
    for VARIANT in "${VARIANTS[@]}"; do
        SUFFIX="_${CITY}_${VARIANT}"
        MODEL="data/${CITY}/models/stretch_ml_best${SUFFIX}.pth"
        LOG="test_logs/test_stretch${SUFFIX}.log"

        # Map variant to paths filename (mirrors the train scripts)
        case "$VARIANT" in
            "ksp")
                PATHS="paths_scored_step2_${CITY}_ksp.csv" ;;
            "plateau_ub_2_dm_4_sim_07")
                PATHS="paths_scored_step2_paths_raw_step1_${CITY}_ub_2_dm_4_sim_07.csv" ;;
            "randomwalk"|"stochastic_mu15")
                PATHS="paths_scored_step2_${CITY}_${VARIANT}.csv" ;;
        esac

        if [ ! -f "$MODEL" ]; then
            echo "[SKIP] Model not found: $MODEL"
            continue
        fi

        echo "=========================================="
        echo "Testing: $CITY  variant: $VARIANT"
        echo "  model      : $MODEL"
        echo "  paths_path : $PATHS"
        echo "  log        : $LOG"
        echo "=========================================="

        python -u test_stretch.py \
            --data_dir "data/${CITY}" \
            --paths_path "$PATHS" \
            --model_suffix "${SUFFIX}" \
            --batch_size 32 \
            2>&1 | tee "$LOG"

        echo "Done: $CITY ($VARIANT)"
        echo ""
    done
done

echo "All tests complete!"
