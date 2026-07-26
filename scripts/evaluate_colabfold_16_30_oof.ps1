param(
    [ValidateSet('pilot', 'full')]
    [string]$Scope = 'pilot',
    [string]$Python = 'D:\anaconda3\envs\dachuang-26\python.exe',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'

if ($Scope -eq 'pilot') {
    $Tag = 'pilot400'
} else {
    $Tag = 'full2034'
}
$CacheRoot = "data/af2_ensemble_16_30/cache_$Scope"
$Manifest = "$CacheRoot/conformer_manifest.csv"
$Augmented = "data/processed/train_chemical_sites_final_esm2_af2_16_30_$Tag.pt"
$Subset = "data/processed/train_chemical_sites_final_esm2_af2_16_30_${Tag}_subset.pt"
$SubsetFolds = "data/folds/af2_16_30_${Tag}_folds.csv"
$RunPrefix = "runs/af2_16_30_${Tag}_residual"
$BasePrefix = 'runs/chemical_site_ablation_include_same_residue'

& $Python -u -B -m ghxtox.conformer_features `
    --processed data/processed/train_chemical_sites_final_esm2.pt `
    --conformer-manifest $Manifest `
    --cache-root $CacheRoot `
    --output $Augmented `
    --pilot-output $Subset `
    --pilot-fold-manifest $SubsetFolds `
    --source-fold-manifest data/folds/train_cdhit080_fallback_folds.csv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

foreach ($Fold in 0..4) {
    & $Python -u -B -m ghxtox.train `
        --train $Subset `
        --fold-manifest $SubsetFolds `
        --fold $Fold `
        --config configs/conformer_ensemble_residual_calibrated.json `
        --output-dir "${RunPrefix}_fold$Fold" `
        --device $Device
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$OofDir = "${RunPrefix}_oof"
New-Item -ItemType Directory -Force -Path $OofDir | Out-Null

$BaseCheckpoints = 0..4 | ForEach-Object {
    "${BasePrefix}_fold$_/best_model.pt"
}
$CandidateCheckpoints = 0..4 | ForEach-Object {
    "${RunPrefix}_fold$_/best_model.pt"
}

# Compare on exactly the AF2-covered cohort. This is the prespecified gate:
# MCC must not decrease, and at least one of BACC/AUROC/AUPRC must improve.
& $Python -u -B -m ghxtox.oof `
    --checkpoints $BaseCheckpoints `
    --processed $Subset `
    --fold-manifest $SubsetFolds `
    --output-csv "$OofDir/covered_base_predictions.csv" `
    --output-json "$OofDir/covered_base_summary.json" `
    --metric mcc `
    --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -u -B -m ghxtox.oof `
    --checkpoints $CandidateCheckpoints `
    --processed $Subset `
    --fold-manifest $SubsetFolds `
    --output-csv "$OofDir/covered_candidate_predictions.csv" `
    --output-json "$OofDir/covered_candidate_summary.json" `
    --metric mcc `
    --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -B -m ghxtox.conformer_compare `
    --baseline "$OofDir/covered_base_predictions.csv" `
    --candidate "$OofDir/covered_candidate_predictions.csv" `
    --threshold 0.677819 `
    --bootstrap-iterations 1000 `
    --permutation-iterations 1000 `
    --output "$OofDir/covered_paired_vs_base.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -u -B -m ghxtox.oof `
    --checkpoints $CandidateCheckpoints `
    --processed $Augmented `
    --fold-manifest data/folds/train_cdhit080_fallback_folds.csv `
    --output-csv "$OofDir/full_predictions.csv" `
    --output-json "$OofDir/full_summary.json" `
    --metric mcc `
    --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -B -m ghxtox.conformer_compare `
    --baseline runs/3d_v2_default/oof_predictions.csv `
    --candidate "$OofDir/full_predictions.csv" `
    --threshold 0.677819 `
    --bootstrap-iterations 1000 `
    --permutation-iterations 1000 `
    --output "$OofDir/paired_vs_base.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Paired = Get-Content "$OofDir/covered_paired_vs_base.json" -Raw | ConvertFrom-Json
$MccDelta = [double]$Paired.metrics.mcc.delta_v2_minus_v1
$BaccDelta = [double]$Paired.metrics.balanced_accuracy.delta_v2_minus_v1
$AurocDelta = [double]$Paired.metrics.auroc.delta_v2_minus_v1
$AuprcDelta = [double]$Paired.metrics.auprc.delta_v2_minus_v1
$Retain = ($MccDelta -ge -1e-12) -and (
    ($BaccDelta -gt 0) -or ($AurocDelta -gt 0) -or ($AuprcDelta -gt 0)
)
$Decision = [ordered]@{
    scope = $Scope
    cohort = $Tag
    retained = $Retain
    gate = 'covered-cohort MCC non-decrease and at least one of BACC/AUROC/AUPRC improves'
    fixed_threshold = 0.677819
    deltas = [ordered]@{
        mcc = $MccDelta
        balanced_accuracy = $BaccDelta
        auroc = $AurocDelta
        auprc = $AuprcDelta
    }
    next_action = if ($Retain) {
        if ($Scope -eq 'pilot') {
            'run server full only after reviewing bootstrap and fold stability'
        } else {
            'retain as AF2 conformer candidate; external confirmation still required'
        }
    } else {
        'stop this route; keep only compact negative-ablation summaries'
    }
}
$Decision | ConvertTo-Json -Depth 6 |
    Set-Content "$OofDir/decision.json" -Encoding UTF8

if (-not $Retain) {
    Write-Host 'AF2 conformer candidate did not pass the prespecified OOF gate.'
    Write-Host "Decision: $OofDir/decision.json"
    exit 0
}

& $Python -B -m ghxtox.oof_fusion apply `
    --baseline "$OofDir/full_predictions.csv" `
    --expert runs/plm_textcnn_prott5_bsmote_multiseed/oof_predictions.csv `
    --output-dir "${RunPrefix}_prott5_frozen_oof" `
    --expert-weight 0.62 `
    --threshold 0.60
exit $LASTEXITCODE
