import os, sys, time
sys.path.insert(0, os.path.abspath("src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import T5EncoderModel, T5Tokenizer
from ghxtox.lora_mamba_1d import LoraMamba1D

path = "data/models/prot_t5_xl_half_uniref50-enc"
tok = T5Tokenizer.from_pretrained(path, do_lower_case=False)
plm = T5EncoderModel.from_pretrained(path, torch_dtype=torch.float16).to("cuda")
plm.eval()
model = LoraMamba1D(plm, token_dim=1024, d_model=192, d_state=16, d_conv=4, expand=2,
                    n_layers=2, lora_rank=8, lora_alpha=16, lora_dropout=0.1).to("cuda")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable params:", trainable, flush=True)

seqs = [
    "APAKRWFGHEECTYWLGPCEVDDTCCSASCESKFCGLW",
    "DCCIIAGCPFGCTICC",
    "CKGKGAPCTRLMYDCCHGSCSSSKGRCG",
    "LAKRADICQPGKTSQRACET",
    "RSVCSNGCRPKPFGGCSC",
] * 4  # 20 samples

for batch_size in (4, 8):
    enc = tok([" ".join(s) for s in seqs[:batch_size]], add_special_tokens=True, padding=True, return_tensors="pt")
    ids, attn = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
    lengths = torch.tensor([len(s) for s in seqs[:batch_size]]).cuda()
    labels = torch.rand(batch_size).cuda()
    # warmup
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(ids, attn, lengths)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    # timed
    t0 = time.time()
    n = 5
    for _ in range(n):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(ids, attn, lengths)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"batch={batch_size}: {dt*1000:.0f} ms/step ({dt/batch_size*1000:.0f} ms/sample), max mem {mem:.2f} GB", flush=True)
    torch.cuda.reset_peak_memory_stats()
print("GPU BENCH DONE")
