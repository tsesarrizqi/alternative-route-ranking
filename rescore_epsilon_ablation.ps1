# Generate re-scored CSVs for epsilon ablation study
# City: London | Variants: KSP, Plateau | Epsilons: 0.01, 0.025, 0.05, 0.15, 0.20
# Uses batch mode: Dijkstra computed ONCE per variant, reused for all epsilons.
# Output CSVs are saved alongside the originals in data/london/graph/
# Note: eps=0.10 is the baseline (original CSV, no suffix). Focus on lower epsilon range.

$data_dir  = "data/london"
$out_dir   = "data/london/graph"

$variants = @(
    @{ name = "ksp";     csv = "graph/paths_scored_step2_london_ksp.csv" },
    @{ name = "plateau"; csv = "graph/paths_scored_step2_paths_raw_step1_london_ub_2_dm_4_sim_07.csv" }
)

foreach ($variant in $variants) {
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Variant: $($variant.name)  (eps 0.05 and 0.2)" -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan

    # --epsilons batch mode: Dijkstra runs once, all CSVs written in one pass
    python rescore_stretch.py `
        --data_dir   $data_dir `
        --paths_path $variant.csv `
        --epsilons   0.01 0.025 0.05 0.15 0.20 `
        --output_dir $out_dir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: rescoring failed for $($variant.name)" -ForegroundColor Red
        exit 1
    }
    Write-Host "Done: $($variant.name)" -ForegroundColor Green
    Write-Host ""
}

Write-Host "All 10 rescored CSVs ready (2 variants x 5 epsilons)." -ForegroundColor Green
