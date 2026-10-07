# Rescoring for epsilon ablation: eps=0.01, 0.025, 0.15
# City: London | Variants: KSP, Plateau
# eps=0.05 and eps=0.20 already exist; only these 3 are new.
# Dijkstra runs once per variant, all 3 CSVs written in one pass.

$data_dir = "data/london"
$out_dir  = "data/london/graph"

$variants = @(
    @{ name = "ksp";     csv = "graph/paths_scored_step2_london_ksp.csv" },
    @{ name = "plateau"; csv = "graph/paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07.csv" }
)

foreach ($variant in $variants) {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Variant: $($variant.name)  (eps 0.01, 0.025, 0.15)" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    python rescore_stretch.py `
        --data_dir   $data_dir `
        --paths_path $variant.csv `
        --epsilons   0.01 0.025 0.15 `
        --output_dir $out_dir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: rescoring failed for $($variant.name)" -ForegroundColor Red
        exit 1
    }
    Write-Host "Done: $($variant.name)" -ForegroundColor Green
    Write-Host ""
}

Write-Host "Rescoring complete (eps=0.01, 0.025, 0.15 for both variants)." -ForegroundColor Green
