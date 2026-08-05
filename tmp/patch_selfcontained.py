p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()

# 1) remove ghxtox.* imports
old_imports = "from ghxtox.folds import load_fold_indices\nfrom ghxtox.utils import save_json, resolve_device\n"
assert old_imports in s
s = s.replace(old_imports, "", 1)

# 2) insert local helpers after imports (before the LoRA section header)
anchor = "# ---------------------------------------------------------------------------\n# LoRA\n# ---------------------------------------------------------------------------"
helpers = '''# ---------------------------------------------------------------------------
# Local helpers (self-contained; no dependency on the rest of the ghxtox package)
# ---------------------------------------------------------------------------
def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested in {"gpu", "cuda"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
        return torch.device("cuda")
    return torch.device(requested)


def _save_json(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _load_fold_indices(
    manifest: str | Path, validation_fold: int, expected_size: int
) -> tuple[list[int], list[int]]:
    """Load train/validation indices from a fixed fold manifest (source_index -> fold)."""
    assignments: dict[int, int] = {}
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            assignments[int(row["source_index"])] = int(row["fold"])
    if set(assignments) != set(range(expected_size)):
        raise ValueError(
            f"Fold manifest contains {len(assignments)} indices, expected exactly {expected_size}."
        )
    validation = [index for index, fold in assignments.items() if fold == validation_fold]
    training = [index for index, fold in assignments.items() if fold != validation_fold]
    if not training or not validation:
        raise ValueError(f"Fold {validation_fold} produced an empty train or validation subset.")
    return training, validation


'''
assert anchor in s
s = s.replace(anchor, helpers + anchor, 1)

# 3) replace usages
s = s.replace("load_fold_indices(fold_manifest, fold, count)", "_load_fold_indices(fold_manifest, fold, count)")
s = s.replace("save_json(result, out_dir / f\"oof_seed{args.seed}.json\")", "_save_json(result, out_dir / f\"oof_seed{args.seed}.json\")")
s = s.replace("device = resolve_device(args.device)", "device = _resolve_device(args.device)")

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched self-contained")
