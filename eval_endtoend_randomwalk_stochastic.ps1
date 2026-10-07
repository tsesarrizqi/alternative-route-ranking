# Evaluate end-to-end for 5 cities x 2 variants (randomwalk, stochastic_mu15)

$Cities = @("london","manhattan","paris","singapore","tokyo")
$Variants = @(
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
        $suffix   = $V.suffix_tpl   -replace "\{city\}", $City
        $paths    = $V.paths_tpl    -replace "\{city\}", $City
        $genorder = $V.genorder_tpl -replace "\{city\}", $City
        $log      = "eval_logs\endtoend${suffix}.log"

        Write-Host "=========================================="
        Write-Host "City: $City  Variant: $($V.key)"
        Write-Host "  model_suffix : $suffix"
        Write-Host "  paths_path   : $paths"
        Write-Host "  genorder     : $genorder"
        Write-Host "  log          : $log"
        Write-Host "=========================================="

        python -u evaluate_endtoend.py `
            --data_dir "data/$City" `
            --model_suffix $suffix `
            --paths_path $paths `
            --genorder_path $genorder `
            --top_k 3 `
            --theta 0.5 `
            --delta 0.1 `
            2>&1 | Tee-Object -FilePath $log

        Write-Host "Done: $City ($($V.key))"
        Write-Host ""
    }
}

Write-Host "All evaluations complete!"
