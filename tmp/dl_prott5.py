import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import T5EncoderModel, T5Tokenizer
t0 = time.time()
print("loading tokenizer...", flush=True)
tok = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
print("loading model...", flush=True)
model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc", torch_dtype=torch.float16)
model.eval()
print("done in %.1fs" % (time.time()-t0), flush=True)
print("num params:", sum(p.numel() for p in model.parameters()), flush=True)
