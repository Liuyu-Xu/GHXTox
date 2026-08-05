p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()

helper = '''def _trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """State dict restricted to trainable parameters (LoRA + head), excluding the frozen PLM."""
    keys = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    return {name: value for name, value in model.state_dict().items() if name in keys}


'''
anchor = "# ---------------------------------------------------------------------------\n# Commands"
assert anchor in s
s = s.replace(anchor, helper + anchor, 1)

old = '            torch.save({"model_state": model.state_dict(), "config": config, "fold": fold, "seed": seed}, checkpoint_path)'
new = '            torch.save({"model_state": _trainable_state_dict(model), "config": config, "fold": fold, "seed": seed}, checkpoint_path)'
assert old in s
s = s.replace(old, new, 1)

old = '    torch.save({"model_state": model.state_dict(), "config": config, "seed": seed}, output)'
new = '    torch.save({"model_state": _trainable_state_dict(model), "config": config, "seed": seed}, output)'
assert old in s
s = s.replace(old, new, 1)

old = '    model.load_state_dict(payload["model_state"])'
new = '    missing, unexpected = model.load_state_dict(payload["model_state"], strict=False)\n    if unexpected:\n        raise ValueError(f"Unexpected checkpoint keys: {unexpected[:5]}")'
assert old in s
s = s.replace(old, new, 1)

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched checkpoint size")
