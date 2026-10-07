# Evaluate all 5 end-to-end conditions for 5 cities x 4 variants x 2 top-k values (3 and 5)
# PowerShell equivalent to run evaluate_endtoend.py for each combination

$Cities = @("london","manhattan","paris","singapore","tokyo")
$TopKValues = @(3, 5)
$Variants = @(
    @{
        key         = "ksp"
        suffix_tpl  = "_{city}_ksp"
        paths_tpl   = "paths_scored_step2_{city}_ksp.csv"
        genorder_tpl= "paths_genorder_{city}_ksp.csv"
    },
    @{
        key         = "plateau"
        suffix_tpl  = "_{city}_plateau_ub_2_dm_4_sim_07"
        paths_tpl   = "paths_scored_step2_paths_raw_step1_{city}_ub_2_dm_4_sim_07.csv"
        genorder_tpl= "paths_genorder_{city}_plateau_ub_2_dm_4_sim_07.csv"
    },
    @{
        key         = "randomwalk"
        suffix_tpl  = "_{city}_randomwalk"
        paths_tpl   = "paths_scored_step2_{city}_randomwalk.csv"
        genorder_tpl= "paths_genorder_{city}_randomwalk.csv"
    },
    @{
        key         = "stochastic_mu15"
        suffix_tpl  = "_{city}_stochastic_mu15"
        paths_tpl   = "paths_scored_step2_{city}_stochastic_mu15.csv"
        genorder_tpl= "paths_genorder_{city}_stochastic_mu15.csv"
    }
)

New-Item -ItemType Directory -Force "eval_logs" | Out-Null

foreach ($City in $Cities) {
    foreach ($V in $Variants) {
        foreach ($TopK in $TopKValues) {
        $suffix   = $V.suffix_tpl   -replace "\{city\}", $City
        $paths    = $V.paths_tpl    -replace "\{city\}", $City
        $genorder = $V.genorder_tpl -replace "\{city\}", $City
        $log        = "eval_logs\endtoend${suffix}_top${TopK}.log"
        $predsCache = "data/$City/results/preds${suffix}.json"

        Write-Host "=========================================="
        Write-Host "City: $City  Variant: $($V.key)  top_k: $TopK"
        Write-Host "  model_suffix : $suffix"
        Write-Host "  paths_path   : $paths"
        Write-Host "  genorder     : $genorder"
        Write-Host "  preds_cache  : $predsCache"
        Write-Host "  log          : $log"
        Write-Host "=========================================="

        python -u evaluate_endtoend.py `
            --data_dir "data/$City" `
            --model_suffix $suffix `
            --paths_path $paths `
            --genorder_path $genorder `
            --top_k $TopK `
            --theta 0.5 `
            --delta 0.1 `
            --preds_cache $predsCache `
            2>&1 | Tee-Object -FilePath $log

        Write-Host "Done: $City ($($V.key)) top_k=$TopK"
        Write-Host ""
        }
    }
}

Write-Host "All evaluations complete!"
