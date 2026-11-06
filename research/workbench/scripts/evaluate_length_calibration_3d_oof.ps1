param(
    [string]$Python = 'D:\anaconda\envs\pytorch_env\python.exe'
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot
$env:PYTHONPATH = 'src'

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

& $Python -u -B -m ghxtox.length_conditioned_calibration_3d `
    --protocol runs/length_calibration_3d/protocol.json `
    --output-root runs/length_calibration_3d
if ($LASTEXITCODE -ne 0) {
    throw "Phase 31 failed with exit code ${LASTEXITCODE}."
}
