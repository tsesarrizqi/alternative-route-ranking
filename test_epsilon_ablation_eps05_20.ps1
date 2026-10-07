# Test epsilon ablation models: eps=0.05 and eps=0.20 on London
# Covers all 4 variants: KSP, Plateau, RandomWalk, Stochastic
# Models extracted from epsilon_ablation_results.tar.gz (sent by Farhan)

$city = "london"

$runs = @(
    @{ variant = "ksp";          eps = 0.05; csv = "paths_scored_step2_london_ksp_eps0.05.csv";                                            suffix = "_london_ksp_eps0.05" },
    @{ variant = "ksp";          eps = 0.20; csv = "paths_scored_step2_london_ksp_eps0.20.csv";                                            suffix = "_london_ksp_eps0.20" },
    @{ variant = "plateau";      eps = 0.05; csv = "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.05.csv"; suffix = "_london_plateau_eps0.05" },
    @{ variant = "plateau";      eps = 0.20; csv = "paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07_eps0.20.csv"; suffix = "_london_plateau_eps0.20" },
    @{ variant = "randomwalk";   eps = 0.05; csv = "paths_scored_step2_london_randomwalk_eps0.05.csv";                       suffix = "_london_randomwalk_eps0.05" },
    @{ variant = "randomwalk";   eps = 0.20; csv = "paths_scored_step2_london_randomwalk_eps0.20.csv";                       suffix = "_london_randomwalk_eps0.20" },
    @{ variant = "stochastic";   eps = 0.05; csv = "paths_scored_step2_london_stochastic_mu15_eps0.05.csv";                  suffix = "_london_stochastic_mu15_eps0.05" },
    @{ variant = "stochastic";   eps = 0.20; csv = "paths_scored_step2_london_stochastic_mu15_eps0.20.csv";                  suffix = "_london_stochastic_mu15_eps0.20" }
)

$passed = 0
$failed = 0

foreach ($run in $runs) {
    $eps_str = $run.eps.ToString("0.00")
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "TEST: $($run.variant)  |  Epsilon: $eps_str" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    python test_stretch.py `
        --data_dir data/$city `
        --paths_path $run.csv `
        --model_suffix $run.suffix `
        --batch_size 32

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Test failed for $($run.variant) eps=$eps_str" -ForegroundColor Red
        $failed++
    } else {
        Write-Host "Done: $($run.variant) eps=$eps_str" -ForegroundColor Green
        $passed++
    }
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Yellow
Write-Host "Summary: $passed passed, $failed failed" -ForegroundColor Yellow
Write-Host "Results in: data/$city/results/" -ForegroundColor Yellow
Write-Host "==========================================" -ForegroundColor Yellow
