import torch
for ds in ("test1", "test2"):
    src = f"data/processed/{ds}_prott5_structure_adapter.pt"
    out = f"data/processed/{ds}_lora_mamba_light.pt"
    payload = torch.load(src, map_location="cpu", weights_only=False)
    light = {
        "sample_ids": list(payload["sample_ids"]),
        "sequences": list(payload["sequences"]),
        "labels": payload["labels"],
        "metadata": {"source": src, "subset": "sample_ids/sequences/labels only for 1D inference"},
    }
    torch.save(light, out)
    import os
    print(out, "n =", len(light["sample_ids"]), "size MB =", round(os.path.getsize(out) / 1e6, 2))
