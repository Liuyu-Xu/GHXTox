import os, sys, json, tempfile
sys.path.insert(0, os.path.abspath("src"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import T5Config, T5EncoderModel, T5TokenizerFast

# Tiny random T5 for logic validation (no pretrained weights needed)
cfg = T5Config(vocab_size=30, d_model=32, d_ff=64, d_kv=16, num_layers=2, num_heads=2,
               decoder_start_token_id=0, eos_token_id=1, pad_token_id=0, n_positions=64)
tmpdir = tempfile.mkdtemp()
cfg.save_pretrained(tmpdir)
plm = T5EncoderModel(cfg)

from ghxtox.lora_mamba_1d import LoraMamba1D, wrap_encoder_lora
n_wrapped = wrap_encoder_lora(plm.encoder, rank=4, alpha=8, dropout=0.1)
print("wrapped linears:", n_wrapped)

model = LoraMamba1D(plm, token_dim=32, d_model=16, d_state=8, d_conv=3, expand=2, n_layers=1,
                    lora_rank=4, lora_alpha=8, lora_dropout=0.1)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print("trainable params:", trainable, "total:", total)

# forward/backward on random inputs
torch.manual_seed(0)
ids = torch.randint(2, 28, (3, 12))
attn = torch.ones(3, 12, dtype=torch.long)
lengths = torch.tensor([10, 8, 12])
logits = model(ids, attn, lengths)
print("logits shape:", tuple(logits.shape), "finite:", torch.isfinite(logits).all().item())
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([1.0, 0.0, 1.0]))
loss.backward()
grad_ok = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
print("backward OK, grads finite:", grad_ok)

# verify base PLM params are frozen
base_frozen = all(not p.requires_grad for name, p in plm.named_parameters() if "lora" not in name)
print("base PLM frozen (except LoRA):", base_frozen)
print("SMOKE LOGIC PASS")
