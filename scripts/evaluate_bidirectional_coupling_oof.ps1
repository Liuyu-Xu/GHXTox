param(
    [string]$Python = 'D:\anaconda\envs\pytorch_env\python.exe',
    [ValidateSet('cuda', 'cpu')]
    [string]$Device = 'cuda',
    [switch]$Resume
)

$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'

$FeatureCache = 'data/processed/train_prott5_structure_oof_adapter_multisource.pt'
$CheckpointPattern = 'runs/prott5_structure_chemical_cross_attention_reproduced_oof/fold{fold}/best_model.pt'
$OutputDir = 'runs/prott5_structure_bidirectional_agreement_oof'
$SummaryPath = "$OutputDir/summary.json"

$RequiredPaths = @($FeatureCache)
$RequiredPaths += 0..4 | ForEach-Object {
    $CheckpointPattern.Replace('{fold}', [string]$_)
}
foreach ($RequiredPath in $RequiredPaths) {
    if (-not (Test-Path $RequiredPath)) {
        throw "Required input not found: $RequiredPath"
    }
}

$Arguments = @(
    '-u',
    '-B',
    '-m',
    'ghxtox._legacy.cross_modal_adapter',
    'train-oof',
    '--feature-cache',
    $FeatureCache,
    '--output-dir',
    $OutputDir,
    '--device',
    $Device,
    '--epochs',
    '40',
    '--batch-size',
    '64',
    '--learning-rate',
    '3e-4',
    '--weight-decay',
    '5e-4',
    '--patience',
    '7',
    '--threshold',
    '0.60',
    '--structure-dropout',
    '0.25',
    '--delta-l1',
    '0.01',
    '--seed',
    '42',
    '--use-chemical-sites',
    '--bidirectional-coupling',
    '--initial-checkpoint-pattern',
    $CheckpointPattern,
    '--freeze-base-adapter',
    '--bidirectional-only-loss'
)
if ($Resume) {
    $Arguments += '--resume'
}

& $Python @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Summary = Get-Content $SummaryPath -Raw | ConvertFrom-Json
$Mcc = [double]$Summary.adapter.mcc
$Auprc = [double]$Summary.adapter.auprc
$Brier = [double]$Summary.adapter.brier
$FoldWins = [int]$Summary.fold_mcc_wins
$FoldLosses = @(
    $Summary.fold_reports | Where-Object { [double]$_.mcc_delta -lt -1e-12 }
).Count

# These values were frozen before this candidate was evaluated.
$Gate = [ordered]@{
    mcc_strictly_above = 0.871184
    auprc_at_least = 0.953639
    brier_at_most = 0.044464
    minimum_fold_mcc_wins = 3
    maximum_fold_mcc_losses = 1
}
$Retained = (
    ($Mcc -gt $Gate.mcc_strictly_above) -and
    ($Auprc -ge $Gate.auprc_at_least) -and
    ($Brier -le $Gate.brier_at_most) -and
    ($FoldWins -ge $Gate.minimum_fold_mcc_wins) -and
    ($FoldLosses -le $Gate.maximum_fold_mcc_losses)
)
$Decision = [ordered]@{
    candidate = 'confidence/agreement-gated bidirectional residue coupling'
    retained = $Retained
    test1_test2_used = $false
    fixed_protocol = [ordered]@{
        seed = 42
        epochs = 40
        batch_size = 64
        learning_rate = 0.0003
        weight_decay = 0.0005
        patience = 7
        threshold = 0.60
        structure_dropout = 0.25
        delta_l1 = 0.01
        base_adapter = 'frozen fold-specific chemical cross-attention default'
    }
    gate = $Gate
    observed = [ordered]@{
        mcc = $Mcc
        auprc = $Auprc
        brier = $Brier
        fold_mcc_wins = $FoldWins
        fold_mcc_losses = $FoldLosses
    }
    next_action = if ($Retained) {
        'freeze architecture; refit at median OOF best epoch, then evaluate Test1/Test2 once'
    } else {
        'retain as an ablation; do not refit or inspect Test1/Test2'
    }
}
$Decision | ConvertTo-Json -Depth 6 |
    Set-Content "$OutputDir/decision.json" -Encoding UTF8

Write-Host "OOF decision saved to $OutputDir/decision.json"
Write-Host "Retained: $Retained"
