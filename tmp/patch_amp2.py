p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()

old = '''try:  # torch >= 2.3
    from torch.amp import GradScaler as _GradScaler

    def _amp_autocast(enabled: bool):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)

except (ImportError, AttributeError):  # older torch (<= 2.2)
    from torch.cuda.amp import GradScaler as _GradScaler

    def _amp_autocast(enabled: bool):
        return torch.cuda.amp.autocast(enabled=enabled)'''
new = '''try:  # torch >= 2.3
    from torch.amp import GradScaler as _GradScaler

    def _make_scaler(enabled: bool):
        return _GradScaler("cuda", enabled=enabled)

    def _amp_autocast(enabled: bool):
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)

except (ImportError, AttributeError):  # older torch (<= 2.2); GradScaler takes no device arg
    from torch.cuda.amp import GradScaler as _GradScaler

    def _make_scaler(enabled: bool):
        return _GradScaler(enabled=enabled)

    def _amp_autocast(enabled: bool):
        return torch.cuda.amp.autocast(enabled=enabled)'''
assert old in s
s = s.replace(old, new, 1)

s = s.replace('scaler = _GradScaler("cuda", enabled=use_amp)', 'scaler = _make_scaler(use_amp)')

open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched scaler factory")
