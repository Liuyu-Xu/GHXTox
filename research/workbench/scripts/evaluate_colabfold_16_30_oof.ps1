param(
    [ValidateSet('pilot', 'full')]
    [string]$Scope = 'pilot',
    [string]$Python = 'D:\anaconda3\envs\dachuang-26\python.exe',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [string]$Archive = ''
)

$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'

if ($Scope -eq 'pilot') {
    $Tag = 'pilot400'
    $ExpectedSamples = 400
} else {
    $Tag = 'full2034'
    $ExpectedSamples = 2034
}
$ExpectedConformers = $ExpectedSamples * 8
$CacheRoot = "data/af2_ensemble_16_30/cache_$Scope"
$Manifest = "$CacheRoot/conformer_manifest.csv"
$TaskManifest = "data/af2_ensemble_16_30/train_${Scope}_$ExpectedSamples.csv"
$Augmented = "data/processed/train_chemical_sites_final_esm2_af2_16_30_$Tag.pt"
$Subset = "data/processed/train_chemical_sites_final_esm2_af2_16_30_${Tag}_subset.pt"
$SubsetFolds = "data/folds/af2_16_30_${Tag}_folds.csv"
$RunPrefix = "runs/af2_16_30_${Tag}_residual"
$BasePrefix = 'runs/chemical_site_ablation_include_same_residue'

if ($Archive) {
    if (-not (Test-Path $Archive)) {
        throw "Archive not found: $Archive"
    }
    $Checksum = "$Archive.sha256"
    if (Test-Path $Checksum) {
        $ExpectedHash = ((Get-Content $Checksum -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
        $ActualHash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualHash -ne $ExpectedHash) {
            throw "SHA256 mismatch for $Archive"
        }
    }
    tar -xzf $Archive -C .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not (Test-Path $Manifest)) {
    throw "Conformer manifest not found: $Manifest"
}
$ManifestRows = Import-Csv $Manifest
$UniqueSamples = @($ManifestRows.sample_id | Sort-Object -Unique).Count
if ($UniqueSamples -ne $ExpectedSamples -or $ManifestRows.Count -ne $ExpectedConformers) {
    throw "Expected $ExpectedSamples samples/$ExpectedConformers conformers; found $UniqueSamples/$($ManifestRows.Count)."
}

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

& $Python -B -m ghxtox.conformer_cohort_analysis `
    --baseline "$OofDir/covered_base_predictions.csv" `
    --candidate "$OofDir/covered_candidate_predictions.csv" `
    --task-manifest $TaskManifest `
    --cache-root $CacheRoot `
    --threshold 0.677819 `
    --output "$OofDir/structure_cohort_analysis.json" `
    --sample-output "$OofDir/structure_sample_analysis.csv"
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

$BestEpochs = @(
    foreach ($Checkpoint in $CandidateCheckpoints) {
        & $Python -B -c @"
import torch
payload = torch.load(r'$Checkpoint', map_location='cpu', weights_only=False)
print(int(payload['epoch']))
"@
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
) | ForEach-Object { [int]$_ } | Sort-Object
$RefitEpochs = [int]$BestEpochs[[Math]::Floor($BestEpochs.Count / 2)]
$RefitDir = "runs/af2_16_30_${Tag}_residual_candidate"

& $Python -u -B -m ghxtox.train `
    --train $Subset `
    --config configs/conformer_ensemble_residual_calibrated.json `
    --initial-checkpoint runs/3d_v2_default/best_model.pt `
    --output-dir $RefitDir `
    --full-data-epochs $RefitEpochs `
    --device $Device
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -B -m ghxtox.oof_fusion apply `
    --baseline "$OofDir/full_predictions.csv" `
    --expert runs/plm_textcnn_prott5_bsmote_multiseed/oof_predictions.csv `
    --output-dir "${RunPrefix}_prott5_frozen_oof" `
    --expert-weight 0.62 `
    --threshold 0.60
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Retained AF2 conformer candidate: $RefitDir/best_model.pt"
Write-Host "Full-data refit epochs (median fold-best epoch): $RefitEpochs"
