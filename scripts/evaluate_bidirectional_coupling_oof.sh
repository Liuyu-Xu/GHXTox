#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
FEATURE_CACHE="data/processed/train_prott5_structure_oof_adapter_multisource.pt"
CHECKPOINT_PATTERN="runs/prott5_structure_chemical_cross_attention_reproduced_oof/fold{fold}/best_model.pt"
OUTPUT_DIR="runs/prott5_structure_bidirectional_agreement_oof"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

required=("$FEATURE_CACHE")
for fold in 0 1 2 3 4; do
    required+=("${CHECKPOINT_PATTERN/\{fold\}/$fold}")
done
for path in "${required[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Required input not found: $path" >&2
        exit 1
    fi
done

args=(
    -u -B -m ghxtox._legacy.cross_modal_adapter train-oof
    --feature-cache "$FEATURE_CACHE"
    --output-dir "$OUTPUT_DIR"
    --device "$DEVICE"
    --epochs 40
    --batch-size 64
    --learning-rate 3e-4
    --weight-decay 5e-4
    --patience 7
    --threshold 0.60
    --structure-dropout 0.25
    --delta-l1 0.01
    --seed 42
    --use-chemical-sites
    --bidirectional-coupling
    --initial-checkpoint-pattern "$CHECKPOINT_PATTERN"
    --freeze-base-adapter
    --bidirectional-only-loss
)
if [[ "${1:-}" == "--resume" ]]; then
    args+=(--resume)
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--resume]" >&2
    exit 2
fi

"$PYTHON" "${args[@]}"

OUTPUT_DIR="$OUTPUT_DIR" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["OUTPUT_DIR"])
summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
adapter = summary["adapter"]
fold_losses = sum(
    float(report["mcc_delta"]) < -1e-12 for report in summary["fold_reports"]
)
gate = {
    "mcc_strictly_above": 0.871184,
    "auprc_at_least": 0.953639,
    "brier_at_most": 0.044464,
    "minimum_fold_mcc_wins": 3,
    "maximum_fold_mcc_losses": 1,
}
observed = {
    "mcc": float(adapter["mcc"]),
    "auprc": float(adapter["auprc"]),
    "brier": float(adapter["brier"]),
    "fold_mcc_wins": int(summary["fold_mcc_wins"]),
    "fold_mcc_losses": fold_losses,
}
retained = (
    observed["mcc"] > gate["mcc_strictly_above"]
    and observed["auprc"] >= gate["auprc_at_least"]
    and observed["brier"] <= gate["brier_at_most"]
    and observed["fold_mcc_wins"] >= gate["minimum_fold_mcc_wins"]
    and observed["fold_mcc_losses"] <= gate["maximum_fold_mcc_losses"]
)
decision = {
    "candidate": "confidence/agreement-gated bidirectional residue coupling",
    "retained": retained,
    "test1_test2_used": False,
    "fixed_protocol": {
        "seed": 42,
        "epochs": 40,
        "batch_size": 64,
        "learning_rate": 0.0003,
        "weight_decay": 0.0005,
        "patience": 7,
        "threshold": 0.60,
        "structure_dropout": 0.25,
        "delta_l1": 0.01,
        "base_adapter": "frozen fold-specific chemical cross-attention default",
    },
    "gate": gate,
    "observed": observed,
    "next_action": (
        "freeze architecture; refit at median OOF best epoch, then evaluate "
        "Test1/Test2 once"
        if retained
        else "retain as an ablation; do not refit or inspect Test1/Test2"
    ),
}
(output / "decision.json").write_text(
    json.dumps(decision, indent=2) + "\n", encoding="utf-8"
)
print(f"OOF decision saved to {output / 'decision.json'}")
print(f"Retained: {retained}")
PY
