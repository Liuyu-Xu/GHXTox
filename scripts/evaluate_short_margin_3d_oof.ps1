param(
    [ValidateSet('verify', 'train', 'evaluate', 'aggregate', 'all')]
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
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$outputRoot = 'runs/short_margin_3d'
$protocol = Join-Path $outputRoot 'protocol.json'
$config = 'configs/conformer_ensemble_short_margin_fixed12.json'

function Invoke-PythonChecked {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Invoke-PythonChecked @('-B', '-m', 'ghxtox.short_margin_3d', 'verify', '--protocol', $protocol)
if ($Stage -eq 'verify') {
    exit 0
}

if (($Stage -eq 'train') -or ($Stage -eq 'all')) {
    foreach ($fold in 0..4) {
        $foldOutput = Join-Path $outputRoot "fold$fold"
        $checkpoint = Join-Path $foldOutput 'best_model.pt'
        if ($Resume -and (Test-Path -LiteralPath $checkpoint)) {
            Write-Output "Reusing $checkpoint"
            continue
        }
        Invoke-PythonChecked @(
            '-u', '-B', '-m', 'ghxtox.train',
            '--train', 'data/processed/train_chemical_sites_final_esm2_pepflow_pilot250.pt',
            '--fold-manifest', 'data/folds/pepflow_pilot250_folds.csv',
            '--fold', "$fold", '--config', $config,
            '--output-dir', $foldOutput, '--device', $Device
        )
    }
}

if (($Stage -eq 'evaluate') -or ($Stage -eq 'all')) {
    $checkpoints = 0..4 | ForEach-Object {
        $checkpoint = Join-Path $outputRoot "fold$_/best_model.pt"
        if (-not (Test-Path -LiteralPath $checkpoint)) {
            throw "Missing Phase 29 fold checkpoint: $checkpoint"
        }
        $checkpoint
    }
    $candidateDir = Join-Path $outputRoot 'candidate'
    New-Item -ItemType Directory -Force -Path $candidateDir | Out-Null
    $oofArgs = @('-u', '-B', '-m', 'ghxtox.oof', '--checkpoints') + @($checkpoints) + @(
        '--processed', 'data/processed/train_chemical_sites_final_esm2_pepflow.pt',
        '--fold-manifest', 'data/folds/train_cdhit080_fallback_folds.csv',
        '--output-csv', (Join-Path $candidateDir 'oof_predictions.csv'),
        '--output-json', (Join-Path $candidateDir 'oof_summary.json'),
        '--fixed-threshold', '0.677819', '--device', $Device
    )
    Invoke-PythonChecked -Arguments $oofArgs
}

if (($Stage -eq 'aggregate') -or ($Stage -eq 'all')) {
    $candidate = Join-Path $outputRoot 'candidate/oof_predictions.csv'
    if (-not (Test-Path -LiteralPath $candidate)) {
        throw "Missing Phase 29 candidate OOF: $candidate"
    }
    Invoke-PythonChecked @(
        '-B', '-m', 'ghxtox.short_margin_3d', 'aggregate',
        '--protocol', $protocol, '--candidate-oof', $candidate,
        '--output-dir', (Join-Path $outputRoot 'aggregate')
    )
}
