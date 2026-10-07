#!/bin/bash
# Profile GPU/CPU usage during 1-epoch training of the plateau variant
# for all 11 cities. Results saved to profile_logs/.
#
# Usage:
#   chmod +x profile_training_plateau.sh
#   ./profile_training_plateau.sh

CITIES=("amsterdam" "bangalore" "london" "manchester" "manhattan" "paris" "rome" "seattle" "singapore" "sydney" "tokyo")
VARIANT="plateau_ub_2_dm_4_sim_07"
INTERVAL=1          # seconds between samples
BATCH_SIZE=32

mkdir -p profile_logs

for CITY in "${CITIES[@]}"; do
    case "$VARIANT" in
        plateau_ub_2_dm_4_sim_07)
            PATHS="paths_scored_step2_paths_raw_step1_${CITY}_ub_2_dm_4_sim_07.csv"
            ;;
    esac

    USAGE_CSV="profile_logs/usage_${CITY}_${VARIANT}.csv"
    TRAIN_LOG="profile_logs/train_${CITY}_${VARIANT}.log"

    echo "=========================================="
    echo "Profiling: $CITY  |  variant: $VARIANT"
    echo "  GPU/CPU log : $USAGE_CSV"
    echo "  Train log   : $TRAIN_LOG"
    echo "=========================================="

    # Start background monitor
    python -u monitor_usage.py --output "$USAGE_CSV" --interval "$INTERVAL" &
    MONITOR_PID=$!

    # Run 1-epoch training with the same hyperparams as the full run
    python -u train_stretch.py \
        --data_dir   data/$CITY \
        --paths_path "$PATHS" \
        --epochs     1 \
        --lr         0.001 \
        --batch_size $BATCH_SIZE \
        --loss       listmle \
        --hidden_dim 128 \
        --use_weighted \
        --n_bins     15 \
        --weight_power 0.6 \
        --train_alpha 1 \
        --val_alpha   0.05 \
        --epsilon     0.1 \
        --model_suffix _${CITY}_${VARIANT}_profile \
        2>&1 | tee "$TRAIN_LOG"

    # Stop monitor
    kill "$MONITOR_PID" 2>/dev/null
    wait "$MONITOR_PID" 2>/dev/null

    echo "Done: $CITY"
    echo ""
done

echo "All cities profiled."
echo ""
echo "Summary of peak GPU/CPU usage:"
echo "------------------------------------------------------------"
printf "%-12s  %8s  %10s  %8s  %8s\n" "City" "GPU%_max" "GMem_MB_max" "CPU%_max" "RAM_GB_max"
echo "------------------------------------------------------------"

for CITY in "${CITIES[@]}"; do
    CSV="profile_logs/usage_${CITY}_${VARIANT}.csv"
    if [ ! -f "$CSV" ]; then
        printf "%-12s  (no data)\n" "$CITY"
        continue
    fi
    # Use python to parse the CSV (handles None gracefully)
    python3 - <<EOF
import csv, sys

path = "$CSV"
gpu_util, gpu_mem, cpu_util, ram = [], [], [], []
with open(path) as f:
    for row in csv.DictReader(f):
        for lst, col in [(gpu_util, "gpu_util_pct"),
                         (gpu_mem,  "gpu_mem_used_mb"),
                         (cpu_util, "cpu_util_pct"),
                         (ram,      "ram_used_gb")]:
            try:
                lst.append(float(row[col]))
            except (ValueError, KeyError):
                pass

def mx(lst):
    return f"{max(lst):.1f}" if lst else "--"

print(f"{'$CITY':<12}  {mx(gpu_util):>8}  {mx(gpu_mem):>10}  {mx(cpu_util):>8}  {mx(ram):>8}")
EOF
done
echo "------------------------------------------------------------"
