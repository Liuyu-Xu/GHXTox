param(
    [string]$Python = "D:\anaconda\envs\pytorch_env\python.exe",
    [string]$Device = "auto",
    [switch]$VerifyOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

$arguments = @(
    "-m", "ghxtox.partial_unfreeze_3d",
    "--protocol", "runs/partial_unfreeze_3d/protocol.json",
    "--device", $Device,
    "--resume"
)
if ($VerifyOnly) {
    $arguments += "--verify-only"
}
& $Python @arguments
exit $LASTEXITCODE
