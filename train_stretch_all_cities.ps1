# Train stretch model on all 11 cities
# Parameters match thesis configuration

$cities = @(
    "amsterdam",
    "bangalore",
    "london",
    "manchester",
    "manhattan",
    "paris",
    "rome",
    "seattle",
    "singapore",
    "sydney",
    "tokyo"
)

foreach ($city in $cities) {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Training: $city" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    python train_stretch.py `
        --data_dir data/$city `
        --paths_path paths_scored_step2_paths_raw_step1_${city}_ub_2_dm_4_sim_07.csv `
        --epochs 25 `
        --lr 0.001 `
        --batch_size 32 `
        --loss listmle `
        --hidden_dim 128 `
        --use_weighted `
        --n_bins 15 `
        --weight_power 0.6 `
        --train_alpha 1 `
        --val_alpha 0.05 `
        --epsilon 0.1 `
        --model_suffix _${city}_ub2_dm4
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Training failed for $city" -ForegroundColor Red
        exit 1
    }
    Write-Host "Done: $city" -ForegroundColor Green
}

Write-Host "All cities done!" -ForegroundColor Green
