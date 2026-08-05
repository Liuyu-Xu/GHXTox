import os
paths = [
    r"src/ghxtox/lora_mamba_1d.py",
    r"scripts/make_lora_mamba_pilot_manifest.py",
    r"scripts/run_lora_mamba_pilot.ps1",
    r"scripts/refreeze_lora_mamba_fusion.py",
    r"configs/lora_mamba_1d.json",
    r"tmp/patch_lora.py",
    r"tmp/patch_mamba.py",
    r"tmp/patch_freeze.py",
    r"tmp/patch_timing.py",
    r"tmp/smoke_logic.py",
    r"tmp/smoke_logic2.py",
    r"tmp/dl_prott5.py",
    r"tmp/dl_prott5_aux.py",
    r"tmp/dl_prott5_model.py",
]
for p in paths:
    if not os.path.exists(p):
        continue
    data = open(p, "rb").read()
    if data.startswith(b"\xef\xbb\xbf"):
        open(p, "wb").write(data[3:])
        print("stripped BOM:", p)
print("done")
