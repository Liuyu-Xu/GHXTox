import os, sys
sys.path.insert(0, os.path.abspath("src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import T5EncoderModel, T5Tokenizer
from ghxtox.lora_mamba_1d import LoraMamba1D

torch.manual_seed(0)
path = "data/models/prot_t5_xl_half_uniref50-enc"
tok = T5Tokenizer.from_pretrained(path, do_lower_case=False)
plm = T5EncoderModel.from_pretrained(path, torch_dtype=torch.float16).to("cuda")
plm.eval()

def check(name, t):
    print(f"{name}: finite={torch.isfinite(t).all().item()} min={float(t.min())} max={float(t.max())}")

seqs = ["APAKRWFGHEECTYWLGPCEVDDTCCSASCESKFCGLW", "DCCIIAGCPFGCTICC"]
enc = tok([" ".join(s) for s in seqs], add_special_tokens=True, padding=True, return_tensors="pt")
ids, attn = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
lengths = torch.tensor([len(s) for s in seqs]).cuda()

with torch.autocast(device_type="cuda", dtype=torch.float16):
    hidden = plm(input_ids=ids, attention_mask=attn).last_hidden_state
check("plm hidden", hidden.float())

model = LoraMamba1D(plm, token_dim=1024, d_model=192, d_state=16, d_conv=4, expand=2,
                    n_layers=2, lora_rank=8, lora_alpha=16, lora_dropout=0.1).to("cuda")
model.eval()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    h2 = model.plm(input_ids=ids, attention_mask=attn).last_hidden_state
    check("plm hidden (wrapped)", h2.float())
    x = model.proj(h2)
    check("proj", x.float())
    positions = torch.arange(h2.shape[1], device=h2.device)[None, :]
    valid = (positions < lengths[:, None]).float()
    for i, layer in enumerate(model.mamba_layers):
        x = layer(x, valid)
        check("mamba layer %d" % i, x.float())
    pooled = (x * valid.unsqueeze(-1)).sum(1) / lengths.clamp(min=1).unsqueeze(-1).float()
    check("pooled", pooled.float())
    pooled = model.pool_norm(pooled)
    check("pool_norm", pooled.float())
    logits = model.classifier(pooled).squeeze(-1)
    check("logits", logits.float())
    probs = torch.sigmoid(logits)
    check("probs", probs.float())
print("DEBUG DONE")
