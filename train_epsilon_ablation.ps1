# Epsilon sensitivity analysis: London KSP + Plateau x eps={0.01, 0.025, 0.15}
# Trains 6 models on rescored CSVs (eps=0.05 and eps=0.20 models already exist).
# All hyperparams match the main training (eps=0.1 baseline, already done).
# Note: eps=0.025 CSV is named eps0.03 (Python :.2f rounds 0.025 -> 0.03).
# Run rescore_epsilon_supplemental.ps1 first to generate the input CSVs.

$city = "london"

$runs = @(
    # KSP -- 3 new epsilons only (0.05 and 0.20 models already trained)
    @{ variant = "ksp";     eps = 0.01;  csv = "paths_scored_step2_london_ksp_eps0.01.csv";                                            suffix = "_london_ksp_eps0.01";    batch_size = 16 },
    @{ variant = "ksp";     eps = 0.025; csv = "paths_scored_step2_london_ksp_eps0.03.csv";                                            suffix = "_london_ksp_eps0.025";   batch_size = 16 },
    @{ variant = "ksp";     eps = 0.15;  csv = "paths_scored_step2_london_ksp_eps0.15.csv";                                            suffix = "_london_ksp_eps0.15";    batch_size = 16 },
    # Plateau -- 3 new epsilons only
    @{ variant = "plateau"; eps = 0.01;  csv = "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.01.csv"; suffix = "_london_plateau_eps0.01";  batch_size = 8 },
    @{ variant = "plateau"; eps = 0.025; csv = "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.03.csv"; suffix = "_london_plateau_eps0.025"; batch_size = 8 },
    @{ variant = "plateau"; eps = 0.15;  csv = "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.15.csv"; suffix = "_london_plateau_eps0.15";  batch_size = 8 }
)

foreach ($run in $runs) {
    $eps_str = $run.eps.ToString("0.00")
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Variant: $($run.variant)  |  Epsilon: $eps_str" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    python train_stretch.py `
        --data_dir data/$city `
        --paths_path $run.csv `
        --epochs 50 `
        --lr 0.001 `
        --batch_size $run.batch_size `
        --loss listmle `
        --hidden_dim 128 `
        --use_weighted `
        --n_bins 15 `
        --weight_power 0.6 `
        --train_alpha 1 `
        --val_alpha 0.05 `
        --epsilon $run.eps `
        --model_suffix $run.suffix

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Training failed for $($run.variant) eps=$eps_str" -ForegroundColor Red
        continue
    }
    Write-Host "Done training: $($run.variant) eps=$eps_str" -ForegroundColor Green

    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "TEST: $($run.variant)  |  Epsilon: $eps_str" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    python test_stretch.py `
        --data_dir data/$city `
        --paths_path $run.csv `
        --model_suffix $run.suffix `
        --batch_size 32

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Testing failed for $($run.variant) eps=$eps_str" -ForegroundColor Red
        continue
    }
    Write-Host "Done testing: $($run.variant) eps=$eps_str" -ForegroundColor Green
}

Write-Host ""
Write-Host "All 6 epsilon sensitivity runs (train + test) complete!" -ForegroundColor Green
