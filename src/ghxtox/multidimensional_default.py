# -*- coding: utf-8 -*-
"""Current public default: GHXTox-Multidimensional-Fusion (combination A 1D + 3D).

Two entry points:

  * assemble -- build the fused probability from existing per-sample 1D/3D
    probability CSVs (CPU-only; used to reproduce Test1/Test2 outputs).

  * predict  -- end-to-end inference for (possibly new) sequences:
        1D: LoRA-adapted ProtT5 + bidirectional Mamba, 3 full-train seeds
            averaged, then OOF-fitted Platt calibration -> p_1D
        3D: pLDDT-aware multi-scale structure expert on the ESMFold/chemical-site
            cache -> p_3D (structure_probability)
        fusion: p = w * p_1D + (1-w) * p_3D, decision = p >= threshold
        (w = 0.72, threshold = 0.4702; consistent/OOF-selected on the current
        3D-expert OOF -- see runs/lora_mamba_full/protocol_audit.json)

Requires a GPU for the 1D (ProtT5) and 3D (EGNN/ESM2) forward passes; the
assemble path is CPU-only.
"""
from __future__ import annotations
import argparse, csv, json, os
import numpy as np


def _logit(p):
    x = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(x) - np.log1p(-x)


def fit_platt(logits, labels):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(np.asarray(logits, dtype=np.float64).reshape(-1, 1), np.asarray(labels).astype(int))
    a = float(model.coef_[0][0]); b = float(model.intercept_[0])
    if not (np.isfinite(a) and np.isfinite(b)):
        return 1.0, 0.0
    return a, b


def _load_csv_pairs(path, key):
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            out[str(r["sample_id"])] = (float(r[key]), int(r["label"]))
    return out


def _metrics(labels, prob, thr):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 matthews_corrcoef, roc_auc_score,
                                 average_precision_score, brier_score_loss,
                                 precision_score, recall_score, f1_score)
    pred = (prob >= thr).astype(int)
    tp = int(np.sum((pred == 1) & (labels == 1))); tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0))); fn = int(np.sum((pred == 0) & (labels == 1)))
    sn = tp / (tp + fn) if tp + fn else 0.0
    sp = tn / (tn + fp) if tn + fp else 0.0
    return {"accuracy": float(accuracy_score(labels, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "sensitivity": float(recall_score(labels, pred, zero_division=0)),
            "specificity": float(sp),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, pred)),
            "auroc": float(roc_auc_score(labels, prob)),
            "auprc": float(average_precision_score(labels, prob)),
            "brier": float(brier_score_loss(labels, prob))}


def _write_predictions(output_csv, ids, labels, p1d, p3d, fused, thr, w):
    pred = (fused >= thr).astype(int)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["source_index", "sample_id", "label", "p_1d", "p_3d",
                     "toxicity_probability", "prediction", "decision_threshold", "weight_1d"])
        for i, sid in enumerate(ids):
            wr.writerow([i, sid, int(labels[i]), f"{p1d[i]:.10g}", f"{p3d[i]:.10g}",
                         f"{fused[i]:.10g}", int(pred[i]), f"{thr:.10g}", f"{w:.6g}"])


def assemble(weight, threshold, cal_1d, structure_csv, output_csv, output_metrics=None):
    p1 = _load_csv_pairs(cal_1d, "toxicity_probability")
    p3 = _load_csv_pairs(structure_csv, "structure_probability")
    ids = [k for k in p1 if k in p3]
    labels = np.array([p1[i][1] for i in ids])
    a = np.array([p1[i][0] for i in ids]); b = np.array([p3[i][0] for i in ids])
    fused = weight * a + (1.0 - weight) * b
    _write_predictions(output_csv, ids, labels, a, b, fused, threshold, weight)
    m = _metrics(labels, fused, threshold)
    if output_metrics:
        with open(output_metrics, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
    return {"n": len(ids), "metrics": m}


def _predict_1d_seed(records, checkpoint, config, device, tokenizer):
    import torch
    from ghxtox.lora_mamba_1d import (_load_plm, build_model, predict_probs,
                                      LoraMambaDataset, LoraMambaCollator)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = {**config, **payload.get("config", {})}
    plm = _load_plm(cfg, device)
    model = build_model(plm, cfg, device)
    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)
    if unexpected:
        raise ValueError("Unexpected checkpoint keys: " + str(unexpected[:5]))
    collator = LoraMambaCollator(tokenizer, max_length=int(cfg.get("max_length", 128)))
    return predict_probs(model, LoraMambaDataset(records), collator, device,
                         int(cfg.get("batch_size", 16)), bool(cfg.get("fp16", True)))


def _fit_platt_from_oof(oof_jsons, manifest):
    import csv as _csv
    man_ids = [str(r["sample_id"]) for r in _csv.DictReader(open(manifest, encoding="utf-8-sig"))]
    probs = np.zeros((len(oof_jsons), len(man_ids)), dtype=np.float64)
    labels = None
    for j, path in enumerate(oof_jsons):
        payload = json.load(open(path, encoding="utf-8"))
        p = np.asarray(payload["oof"], dtype=np.float64)
        assert len(p) == len(man_ids), "OOF length != manifest length"
        probs[j] = p
        if labels is None:
            labels = np.asarray(payload["labels"])
    mean = probs.mean(axis=0)
    return fit_platt(_logit(mean), labels)


def predict(processed_path, adapter_path, config_1d, checkpoints, oof_jsons, manifest,
            structure_checkpoint, weight, threshold, device_name, batch_size,
            output_csv, output_metrics=None):
    import torch
    from transformers import T5Tokenizer
    from ghxtox.hybrid_default import _predict_structure
    from ghxtox.lora_mamba_1d import load_test_records
    config = json.load(open(config_1d, encoding="utf-8-sig"))
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = T5Tokenizer.from_pretrained(str(config.get("plm", "Rostlab/prot_t5_xl_half_uniref50-enc")), do_lower_case=False)
    records = load_test_records(adapter_path)
    seed_probs = [_predict_1d_seed(records, ckpt, config, device, tokenizer) for ckpt in checkpoints]
    p_ens = np.mean(seed_probs, axis=0)
    a, b = _fit_platt_from_oof(oof_jsons, manifest)
    p_1d = 1.0 / (1.0 + np.exp(-(a * _logit(p_ens) + b)))
    structure = _predict_structure(processed_path, structure_checkpoint, device=device, batch_size=batch_size)
    p3_map = dict(zip(structure["sample_ids"], structure["probabilities"]))
    ids = [r.sample_id for r in records]
    labels = np.array([r.label for r in records])
    p3 = np.array([p3_map.get(sid, float("nan")) for sid in ids])
    if np.isnan(p3).any():
        raise ValueError("3D probabilities missing for some samples; ensure structure cache covers all inputs.")
    fused = weight * p_1d + (1.0 - weight) * p3
    _write_predictions(output_csv, ids, labels, p_1d, p3, fused, threshold, weight)
    m = _metrics(labels, fused, threshold)
    if output_metrics:
        with open(output_metrics, "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
    return {"n": len(ids), "weight_1d": weight, "threshold": threshold, "metrics": m}


def main():
    ap = argparse.ArgumentParser(description="GHXTox-Multidimensional-Fusion (current default)")
    sub = ap.add_subparsers(dest="command", required=True)
    pa = sub.add_parser("assemble", help="assemble fusion from existing 1D/3D probability CSVs (CPU)")
    pa.add_argument("--cal-1d", required=True)
    pa.add_argument("--structure-csv", required=True)
    pa.add_argument("--weight", type=float, default=0.72)
    pa.add_argument("--threshold", type=float, default=0.47022256531284556)
    pa.add_argument("--output-csv", required=True)
    pa.add_argument("--output-metrics", default=None)
    pp = sub.add_parser("predict", help="end-to-end inference on (possibly new) sequences (GPU)")
    pp.add_argument("--processed", required=True)
    pp.add_argument("--adapter", required=True)
    pp.add_argument("--config-1d", default="configs/lora_mamba_1d_autodl.json")
    pp.add_argument("--checkpoints", nargs="+", required=True)
    pp.add_argument("--oof-jsons", nargs="+", required=True)
    pp.add_argument("--manifest", default="data/folds/train_cdhit080_fallback_folds.csv")
    pp.add_argument("--structure-checkpoint", default="runs/3d_v2_conformer_ensemble_candidate/best_model.pt")
    pp.add_argument("--weight", type=float, default=0.72)
    pp.add_argument("--threshold", type=float, default=0.47022256531284556)
    pp.add_argument("--device", default="auto")
    pp.add_argument("--batch-size", type=int, default=16)
    pp.add_argument("--output-csv", required=True)
    pp.add_argument("--output-metrics", default=None)
    args = ap.parse_args()
    if args.command == "assemble":
        res = assemble(args.weight, args.threshold, args.cal_1d, args.structure_csv, args.output_csv, args.output_metrics)
        print("assemble n=%d BACC=%.4f MCC=%.4f AUROC=%.4f AUPRC=%.4f" % (
            res["n"], res["metrics"]["balanced_accuracy"], res["metrics"]["mcc"],
            res["metrics"]["auroc"], res["metrics"]["auprc"]))
    else:
        res = predict(args.processed, args.adapter, args.config_1d, args.checkpoints, args.oof_jsons,
                      args.manifest, args.structure_checkpoint, args.weight, args.threshold,
                      args.device, args.batch_size, args.output_csv, args.output_metrics)
        print("predict n=%d BACC=%.4f MCC=%.4f AUROC=%.4f AUPRC=%.4f" % (
            res["n"], res["metrics"]["balanced_accuracy"], res["metrics"]["mcc"],
            res["metrics"]["auroc"], res["metrics"]["auprc"]))


if __name__ == "__main__":
    main()
