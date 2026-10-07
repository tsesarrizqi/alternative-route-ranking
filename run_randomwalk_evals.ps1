$ErrorActionPreference = 'Stop'
$cities = 'amsterdam','bangalore','london','manchester','manhattan','paris','rome','seattle','singapore','sydney','tokyo'
foreach ($c in $cities) {
    foreach ($k in 3,5) {
        Write-Output "=== Running $c top_k=$k ==="
        python -u evaluate_endtoend.py --data_dir "data/$c" --model_suffix "_${c}_randomwalk" --paths_path "paths_scored_step2_${c}_randomwalk.csv" --genorder_path "paths_genorder_${c}_randomwalk.csv" --top_k $k --theta 0.5 --delta 0.1
    }
}
Write-Output "=== Aggregating results ==="
python aggregate_results.py
