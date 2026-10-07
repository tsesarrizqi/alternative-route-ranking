#!/bin/bash
# Train stretch model on all 11 cities
# Parameters match thesis configuration

CITIES=(
    "amsterdam"
    "bangalore"
    "london"
    "manchester"
    "manhattan"
    "paris"
    "rome"
    "seattle"
    "singapore"
    "sydney"
    "tokyo"
)

mkdir -p new_logs

for CITY in "${CITIES[@]}"; do
    LOG="new_logs/train_stretch_${CITY}_plateau_ub_2_dm_4_sim_07.log"
    echo "=========================================="
    echo "Training: $CITY  (log: $LOG)"
    echo "=========================================="
    python -u train_stretch.py \
        --data_dir data/$CITY \
        --paths_path paths_scored_step2_paths_raw_step1_${CITY}_ub_2_dm_4_sim_07.csv \
        --epochs 50 \
        --lr 0.001 \
        --batch_size 32 \
        --loss listmle \
        --hidden_dim 128 \
        --use_weighted \
        --n_bins 15 \
        --weight_power 0.6 \
        --train_alpha 1 \
        --val_alpha 0.05 \
        --epsilon 0.1 \
        --model_suffix _${CITY}_plateau_ub_2_dm_4_sim_07 2>&1 | tee "$LOG"
    echo "Done: $CITY"
done

echo "All cities done!"
