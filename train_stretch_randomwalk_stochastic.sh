#!/bin/bash
# Train stretch model on 5 cities x 2 variants (randomwalk, stochastic_mu15)

CITIES=(
    "london"
    "manhattan"
    "paris"
    "singapore"
    "tokyo"
)

VARIANTS=(
    "randomwalk"
    "stochastic_mu15"
)

mkdir -p new_logs

for VARIANT in "${VARIANTS[@]}"; do
    for CITY in "${CITIES[@]}"; do
        LOG="new_logs/train_stretch_${CITY}_${VARIANT}.log"
        echo "=========================================="
        echo "Training: $CITY  variant: $VARIANT  (log: $LOG)"
        echo "=========================================="
        python -u train_stretch.py \
            --data_dir data/$CITY \
            --paths_path paths_scored_step2_${CITY}_${VARIANT}.csv \
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
            --model_suffix _${CITY}_${VARIANT} 2>&1 | tee "$LOG"
        echo "Done: $CITY ($VARIANT)"
    done
done

echo "All cities and variants done!"
