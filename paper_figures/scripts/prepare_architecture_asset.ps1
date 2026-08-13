param(
    [string]$Source = "paper_figures/assets/fig1_ghxtox_architecture_blue_orange.png",
    [string]$Destination = "paper_figures/output/manuscript_figures/fig1_ghxtox_architecture_blue_orange.png"
)

$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..\..")).Path
$sourcePath = (Resolve-Path -LiteralPath (Join-Path $root $Source)).Path
$destinationPath = Join-Path $root $Destination
$destinationDirectory = Split-Path -Parent $destinationPath
New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force

$sourceHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $sourcePath).Hash
$destinationHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $destinationPath).Hash
if ($sourceHash -ne $destinationHash) {
    throw "Architecture asset checksum mismatch after copy."
}

Write-Output "Prepared $destinationPath"
