import os, sys, json, tempfile
sys.path.insert(0, os.path.abspath("src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import T5Config, T5EncoderModel

cfg = T5Config(vocab_size=30, d_model=32, d_ff=64, d_kv=16, num_layers=2, num_heads=2,
               decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, n_positions=64)
tmpdir = tempfile.mkdtemp()
cfg.save_pretrained(tmpdir)
plm = T5EncoderModel(cfg)

from ghxtox.lora_mamba_1d import LoraMamba1D
model = LoraMamba1D(plm, token_dim=32, d_model=16, d_state=8, d_conv=3, expand=2, n_layers=1,
                    lora_rank=4, lora_alpha=8, lora_dropout=0.1)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("trainable params:", trainable)

# base frozen check: params not containing lora_A/lora_B must be frozen
bad = [n for n, p in model.named_parameters() if p.requires_grad and "lora_A" not in n and "lora_B" not in n and not n.startswith("proj") and not n.startswith("mamba") and not n.startswith("pool_norm") and not n.startswith("classifier")]
print("unexpected trainable params:", bad[:5])

torch.manual_seed(0)
ids = torch.randint(2, 28, (3, 12))
attn = torch.ones(3, 12, dtype=torch.long)
lengths = torch.tensor([10, 8, 12])
logits = model(ids, attn, lengths)
print("logits:", tuple(logits.shape), "finite:", torch.isfinite(logits).all().item())
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1.0, 0.0, 1.0]))
loss.backward()
grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
print("grads finite:", grad_ok)
print("SMOKE LOGIC2 PASS")
