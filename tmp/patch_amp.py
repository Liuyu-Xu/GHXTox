p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()

# 1) add compatibility helpers after the _resolve_device block
anchor = '''def _save_json(payload: dict, path: str | Path) -> None:'''
compat = '''try:  # torch >= 2.3
    from torch.amp import GradScaler as _GradScaler

    def _amp_autocast(enabled: bool):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)

except (ImportError, AttributeError):  # older torch (<= 2.2)
    from torch.cuda.amp import GradScaler as _GradScaler

    def _amp_autocast(enabled: bool):
        return torch.cuda.amp.autocast(enabled=enabled)


'''
assert anchor in s
s = s.replace(anchor, compat + anchor, 1)

# 2) type annotation
s = s.replace("scaler: torch.amp.GradScaler | None,", "scaler: _GradScaler | None,")

# 3) autocast usages
s = s.replace('with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):',
              'with _amp_autocast(use_amp):')

# 4) GradScaler creations
s = s.replace('scaler = torch.amp.GradScaler("cuda", enabled=use_amp)', 'scaler = _GradScaler("cuda", enabled=use_amp)')

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched amp compat")
