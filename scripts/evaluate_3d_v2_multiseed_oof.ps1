param(
    [ValidateSet(42, 123, 2025)]
    [int]$Seed = 42,
    [ValidateSet('prepare', 'base', 'residual', 'all', 'aggregate')]
    [string]$Stage = 'all',
    [switch]$Resume,
    [string]$Device = 'cuda',
    [string]$Python = 'D:\anaconda\envs\pytorch_env\python.exe'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$env:PYTHONPATH = 'src'

if (-not (Test-Path -LiteralPath $Python)) {
    $resolvedPython = Get-Command python -ErrorAction Stop
    $Python = $resolvedPython.Source
}

$outputRoot = 'runs/3d_v2_multiseed'
$protocolPath = Join-Path $outputRoot 'protocol.json'

function Invoke-PythonChecked {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Prepare-Protocol {
    Invoke-PythonChecked @(
        '-B', '-m', 'ghxtox.three_d_multiseed', 'prepare',
        '--output-root', $outputRoot
    )
}

if (($Stage -eq 'prepare') -or (-not (Test-Path -LiteralPath $protocolPath))) {
    Prepare-Protocol
}
if ($Stage -eq 'prepare') {
    exit 0
}

if (($Seed -ne 42) -and ($Stage -ne 'aggregate')) {
    $referenceGatePath = Join-Path $outputRoot 'seed_42/reference_gate.json'
    if (-not (Test-Path -LiteralPath $referenceGatePath)) {
        throw 'Seed 42 reference gate is missing; seeds 123/2025 are not authorized.'
    }
    $referenceGate = Get-Content -LiteralPath $referenceGatePath -Raw -Encoding utf8 | ConvertFrom-Json
    if (-not $referenceGate.new_seeds_authorized) {
        throw 'Seed 42 reproduction did not pass; seeds 123/2025 are not authorized.'
    }
}

$seedRoot = Join-Path $outputRoot "seed_$Seed"
$baseConfig = Join-Path $outputRoot "configs/base_seed$Seed.json"
$residualConfig = Join-Path $outputRoot "configs/residual_seed$Seed.json"

if (($Stage -eq 'base') -or ($Stage -eq 'all')) {
    foreach ($fold in 0..4) {
        $foldOutput = Join-Path $seedRoot "base/fold$fold"
        $checkpoint = Join-Path $foldOutput 'best_model.pt'
        if ($Resume -and (Test-Path -LiteralPath $checkpoint)) {
            Write-Output "Reusing $checkpoint"
            continue
        }
        Invoke-PythonChecked @(
            '-u', '-B', '-m', 'ghxtox.train',
            '--train', 'data/processed/train_chemical_sites_final_esm2.pt',
            '--fold-manifest', 'data/folds/train_cdhit080_fallback_folds.csv',
            '--fold', "$fold",
            '--config', $baseConfig,
            '--output-dir', $foldOutput,
            '--device', $Device
        )
    }
    $baseCheckpoints = 0..4 | ForEach-Object { Join-Path $seedRoot "base/fold$_/best_model.pt" }
    $baseOofArguments = @(
        '-u', '-B', '-m', 'ghxtox.oof',
        '--checkpoints'
    ) + @($baseCheckpoints) + @(
        '--processed', 'data/processed/train_chemical_sites_final_esm2.pt',
        '--fold-manifest', 'data/folds/train_cdhit080_fallback_folds.csv',
        '--output-csv', (Join-Path $seedRoot 'base/oof_predictions.csv'),
        '--output-json', (Join-Path $seedRoot 'base/oof_summary.json'),
        '--metric', 'mcc',
        '--device', $Device
    )
    Invoke-PythonChecked -Arguments $baseOofArguments
}

if (($Stage -eq 'residual') -or ($Stage -eq 'all')) {
    foreach ($fold in 0..4) {
        $baseCheckpoint = Join-Path $seedRoot "base/fold$fold/best_model.pt"
        if (-not (Test-Path -LiteralPath $baseCheckpoint)) {
            throw "Missing matching seed-$Seed base checkpoint: $baseCheckpoint"
        }
        $foldOutput = Join-Path $seedRoot "residual/fold$fold"
        $checkpoint = Join-Path $foldOutput 'best_model.pt'
        if ($Resume -and (Test-Path -LiteralPath $checkpoint)) {
            Write-Output "Reusing $checkpoint"
            continue
        }
        Invoke-PythonChecked @(
            '-u', '-B', '-m', 'ghxtox.train',
            '--train', 'data/processed/train_chemical_sites_final_esm2_pepflow_pilot250.pt',
            '--fold-manifest', 'data/folds/pepflow_pilot250_folds.csv',
            '--fold', "$fold",
            '--config', $residualConfig,
            '--output-dir', $foldOutput,
            '--device', $Device
        )
    }
    $residualCheckpoints = 0..4 | ForEach-Object { Join-Path $seedRoot "residual/fold$_/best_model.pt" }
    $candidateOof = Join-Path $seedRoot 'residual/oof_predictions.csv'
    $residualOofArguments = @(
        '-u', '-B', '-m', 'ghxtox.oof',
        '--checkpoints'
    ) + @($residualCheckpoints) + @(
        '--processed', 'data/processed/train_chemical_sites_final_esm2_pepflow.pt',
        '--fold-manifest', 'data/folds/train_cdhit080_fallback_folds.csv',
        '--output-csv', $candidateOof,
        '--output-json', (Join-Path $seedRoot 'residual/oof_summary.json'),
        '--metric', 'mcc',
        '--device', $Device
    )
    Invoke-PythonChecked -Arguments $residualOofArguments
    if ($Seed -eq 42) {
        Invoke-PythonChecked @(
            '-B', '-m', 'ghxtox.three_d_multiseed', 'compare-reference',
            '--reference-oof', 'runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv',
            '--candidate-oof', $candidateOof,
            '--output-json', (Join-Path $seedRoot 'reference_gate.json')
        )
    }
}

if ($Stage -eq 'aggregate') {
    Invoke-PythonChecked @(
        '-B', '-m', 'ghxtox.three_d_multiseed', 'aggregate',
        '--reference-oof', 'runs/conformer_ensemble_residual_oof/full_calibrated_predictions.csv',
        '--quality-csv', 'runs/three_d_team_handoff/oof_3d_quality.csv',
        '--seed-oof',
        '42=runs/3d_v2_multiseed/seed_42/residual/oof_predictions.csv',
        '123=runs/3d_v2_multiseed/seed_123/residual/oof_predictions.csv',
        '2025=runs/3d_v2_multiseed/seed_2025/residual/oof_predictions.csv',
        '--output-csv', (Join-Path $outputRoot 'ensemble/oof_predictions.csv'),
        '--output-summary', (Join-Path $outputRoot 'ensemble/summary.json'),
        '--output-decision', (Join-Path $outputRoot 'ensemble/decision.json')
    )
}
