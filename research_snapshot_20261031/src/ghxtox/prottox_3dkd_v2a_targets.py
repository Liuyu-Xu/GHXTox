"""Build V2-A fold-safe residual targets from frozen v1 fold bases + teachers."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from ghxtox.models.tristat_head import TriStatMLPHead
from ghxtox.prottox_3d_distill_targets import sha256_file
from ghxtox.prottox_tsk import sequence_sha256


SEEDS = (20260905, 20260906)


def _load_v1_fold_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != "ProtTox-3D-Distill" or payload.get("variant") != "mlp_teacher_kd":
        raise ValueError(f"V1 fold checkpoint mismatch: {path}")
    config = payload["config"]
    model = TriStatMLPHead(int(config["input_dim"]), hidden_dim=int(config["hidden_dim"]), dropout=float(config["dropout"]))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload["feature_mean"], payload["feature_std"], payload["metadata"], payload


def build_v2a_targets(cache_path, fold_template, teacher_target_dir, output_dir, seeds=SEEDS, device_name="cuda"):
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    records = cache["records"]
    if len(records) != 6387:
        raise ValueError("V2-A targets require the audited 6387-record cache")
    features = torch.cat(
        (
            torch.stack([torch.as_tensor(record["tristat"], dtype=torch.float32) for record in records]),
            torch.stack([torch.as_tensor(record["physchem"], dtype=torch.float32) for record in records]),
        ),
        dim=1,
    )
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_oof_base_probabilities = [None] * len(records)
    record_positions = {str(record["sample_id"]): index for index, record in enumerate(records)}
    for fold in range(5):
        fold_models = []
        fold_checkpoint_shas = {}
        for seed in seeds:
            path = Path(str(fold_template).format(seed=seed, k=fold))
            fold_models.append(_load_v1_fold_checkpoint(path))
            fold_checkpoint_shas[str(seed)] = sha256_file(path)
        fold_logits = []
        fold_probabilities = []
        for model, mean, std, _metadata, _payload in fold_models:
            logits = model((features - mean) / std.clamp_min(1e-6))
            fold_logits.append(logits)
            fold_probabilities.append(torch.sigmoid(logits))
        base_logit_all = torch.logit((fold_probabilities[0] + fold_probabilities[1]) / 2.0).clamp(-50.0, 50.0)
        base_probability_all = torch.sigmoid(base_logit_all)
        teacher_path = Path(str(teacher_target_dir)).resolve() / f"teacher_outer{fold}_targets.csv"
        teacher_rows = {}
        with teacher_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                teacher_rows[str(row["sample_id"])] = row
        if len(teacher_rows) != len(records) - sum(1 for record in records if int(record["fold"]) == fold):
            raise ValueError(f"Teacher outer-{fold} row count mismatch")
        rows_out = []
        for record_index, record in enumerate(records):
            if int(record["fold"]) == fold:
                continue
            teacher = teacher_rows.get(str(record["historical_sample_id"]))
            if teacher is None:
                raise ValueError(f"Missing teacher target for {record['historical_sample_id']} fold {fold}")
            position = record_positions[str(record["sample_id"])]
            if position != record_index:
                raise RuntimeError("Cache order changed unexpectedly")
            rows_out.append(
                {
                    "source_index": position,
                    "sample_id": record["sample_id"],
                    "historical_sample_id": record["historical_sample_id"],
                    "sequence_hash": record["sequence_sha256"],
                    "outer_fold": fold,
                    "teacher_sample_role": teacher["teacher_sample_role"],
                    "z_base_logit": f"{float(base_logit_all[position]):.9g}",
                    "z_base_probability": f"{float(base_probability_all[position]):.9g}",
                    "z_teacher_logit": teacher["teacher_logit"],
                    "z_teacher_probability": teacher["teacher_probability"],
                    "residual_target_logit": f"{float(teacher['teacher_logit']) - float(base_logit_all[position]):.9g}",
                    "v1_checkpoint_sha256_seed20260905": fold_checkpoint_shas["20260905"],
                    "v1_checkpoint_sha256_seed20260906": fold_checkpoint_shas["20260906"],
                    "teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
                }
            )
        csv_path = output / f"v2a_residual_targets_outer{fold}.csv"
        fieldnames = list(rows_out[0])
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_out)
        summaries.append({"outer_fold": fold, "rows": len(rows_out), "file": csv_path.name, "sha256": sha256_file(csv_path)})
        for index, record in enumerate(records):
            if int(record["fold"]) == fold:
                all_oof_base_probabilities[index] = float(base_probability_all[index])
    if any(value is None for value in all_oof_base_probabilities):
        raise RuntimeError("V1 fold base OOF coverage incomplete")
    oof_csv = output / "v1_fold_base_oof_probabilities.csv"
    with oof_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "historical_sample_id", "sequence_hash", "label", "fold", "probability"])
        for record, probability in zip(records, all_oof_base_probabilities, strict=True):
            writer.writerow([record["sample_id"], record["historical_sample_id"], record["sequence_sha256"], record["label"], record["fold"], f"{probability:.9g}"])
    summary = {
        "artifact_type": "prottox_3dkd_v2a_targets",
        "version": "v2a-1",
        "targets": summaries,
        "v1_fold_base_oof": {"file": oof_csv.name, "sha256": sha256_file(oof_csv), "rows": len(records)},
        "selection_scope": "training OOF only; Test1/Test2 excluded",
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--fold-template", required=True)
    parser.add_argument("--teacher-target-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_v2a_targets(args.cache, args.fold_template, args.teacher_target_dir, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
