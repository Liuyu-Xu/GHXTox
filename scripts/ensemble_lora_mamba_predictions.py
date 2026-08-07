"""Average exactly three aligned public LoRA-BiMamba seed predictions."""
import argparse, csv
from pathlib import Path

import numpy as np

from ghxtox.lora_mamba_1d import PUBLIC_1D_SEEDS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if len(args.inputs) != len(PUBLIC_1D_SEEDS):
        raise ValueError(f"exactly {len(PUBLIC_1D_SEEDS)} seed prediction files are required")
    order = None
    reference = None
    members = []
    for path in args.inputs:
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        ids = [str(row["sample_id"]) for row in rows]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f"{path}: empty or duplicate sample IDs")
        current = [(str(row.get("sequence", "")), int(row["label"])) for row in rows]
        if order is None:
            order, reference = ids, current
        elif ids != order or current != reference:
            raise ValueError(f"{path}: sample order, sequence, or labels differ across seeds")
        values = np.asarray([float(row["toxicity_probability"]) for row in rows])
        if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{path}: invalid probabilities")
        members.append(values)
    mean = np.mean(np.stack(members), axis=0)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "sequence", "label", "toxicity_probability"])
        w.writeheader()
        for index, sample_id in enumerate(order):
            sequence, label = reference[index]
            w.writerow({"sample_id": sample_id, "sequence": sequence, "label": label,
                        "toxicity_probability": f"{mean[index]:.9g}"})
    print("ensemble written:", args.output, "samples:", len(order))

if __name__ == "__main__":
    main()
