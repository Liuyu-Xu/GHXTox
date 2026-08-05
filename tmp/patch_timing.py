p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old = "import numpy as np\nimport torch"
new = "import time\n\nimport numpy as np\nimport torch"
assert old in s
s = s.replace(old, new, 1)
old = "        use_amp = bool(config.get(\"fp16\", True)) and device.type == \"cuda\"\n        scaler = torch.amp.GradScaler(\"cuda\", enabled=use_amp)\n        epochs = int(config.get(\"epochs\", 10)) if max_epochs is None else max_epochs\n        for _ in range(epochs):\n            train_epoch(model, train_loader, optimizer, loss_fn, device, use_amp, scaler)"
new = "        use_amp = bool(config.get(\"fp16\", True)) and device.type == \"cuda\"\n        scaler = torch.amp.GradScaler(\"cuda\", enabled=use_amp)\n        epochs = int(config.get(\"epochs\", 10)) if max_epochs is None else max_epochs\n        fold_start = time.time()\n        for _ in range(epochs):\n            train_epoch(model, train_loader, optimizer, loss_fn, device, use_amp, scaler)\n        timing[f\"fold{fold}_train_seconds\"] = float(time.time() - fold_start)"
assert old in s
s = s.replace(old, new, 1)
old = '''        \"oof\": oof.tolist(),
        \"labels\": labels.tolist(),
        \"folds\": fold_assignments,
    }'''
new = '''        \"oof\": oof.tolist(),
        \"labels\": labels.tolist(),
        \"folds\": fold_assignments,
        \"timing_seconds\": timing,
    }'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched timing")
