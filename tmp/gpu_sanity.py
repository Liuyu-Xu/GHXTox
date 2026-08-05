import os, sys, time
sys.path.insert(0, os.path.abspath("src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import T5EncoderModel, T5Tokenizer
from ghxtox.lora_mamba_1d import (LoraMamba1D, LoraMambaDataset, LoraMambaCollator,
                                  load_train_records, _load_plm, build_model)
from torch.utils.data import DataLoader

path = "data/models/prot_t5_xl_half_uniref50-enc"
tok = T5Tokenizer.from_pretrained(path, do_lower_case=False)
config = {"plm": path, "token_dim": 1024, "d_model": 192, "d_state": 16, "d_conv": 4,
          "expand": 2, "n_layers": 2, "lora_rank": 8, "lora_alpha": 16.0, "lora_dropout": 0.1,
          "dropout": 0.1, "max_length": 128, "batch_size": 4, "learning_rate": 3e-4,
          "weight_decay": 0.01, "epochs": 3, "fp16": True, "grad_checkpoint": False}

records = load_train_records("data/folds/lora_mamba_pilot600_folds.csv")[:100]
torch.manual_seed(42)
plm = _load_plm(config, torch.device("cuda"))
model = build_model(plm, config, torch.device("cuda"))
model.train()
loader = DataLoader(LoraMambaDataset(records), batch_size=4, shuffle=True, collate_fn=LoraMambaCollator(tok))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4569/1818).cuda())
scaler = torch.amp.GradScaler("cuda", enabled=True)
t0 = time.time()
for epoch in range(3):
    losses = []
    for batch in loader:
        b = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(b["input_ids"], b["attention_mask"], b["length"])
            loss = loss_fn(logits, b["label"])
        losses.append(float(loss))
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    print(f"epoch {epoch}: loss={sum(losses)/len(losses):.4f} finite={all(torch.isfinite(torch.tensor(l)).item() for l in losses)}", flush=True)
print("total %.1fs" % (time.time()-t0))
# final forward check
model.eval()
with torch.no_grad():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        b = next(iter(DataLoader(LoraMambaDataset(records), batch_size=8, shuffle=False, collate_fn=LoraMambaCollator(tok))))
        bb = {k: v.cuda() if torch.is_tensor(v) else v for k, v in b.items()}
        logits = model(bb["input_ids"], bb["attention_mask"], bb["length"])
        probs = torch.sigmoid(logits)
        print("eval probs finite:", torch.isfinite(probs).all().item(), "range:", float(probs.min()), float(probs.max()))
print("SANITY DONE")
