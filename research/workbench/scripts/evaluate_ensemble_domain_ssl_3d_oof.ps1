param(
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

$arguments = @(
    '-u', '-B', '-m', 'ghxtox.ensemble_domain_ssl_3d',
    '--protocol', 'runs/ensemble_domain_ssl_3d/protocol.json',
    '--output-root', 'runs/ensemble_domain_ssl_3d',
    '--device', $Device
)
if ($Resume) {
    $arguments += '--resume'
}

& $Python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Phase 30 failed with exit code $LASTEXITCODE."
}
