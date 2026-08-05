import os, sys, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from transformers import T5EncoderModel, T5Tokenizer
path = "data/models/prot_t5_xl_half_uniref50-enc"
t0 = time.time()
tok = T5Tokenizer.from_pretrained(path, do_lower_case=False)
print("tokenizer OK in %.1fs" % (time.time()-t0), flush=True)
seq = "APAKRWFGHEECTYWLGPCEVDDTCCSASCESKFCGLW"
enc = tok([" ".join(seq)], add_special_tokens=True, padding=True, return_tensors="pt")
print("token ids shape:", tuple(enc["input_ids"].shape), "n tokens:", int(enc["attention_mask"].sum()))
t0 = time.time()
model = T5EncoderModel.from_pretrained(path, torch_dtype=torch.float16)
print("model loaded in %.1fs, params=%d" % (time.time()-t0, sum(p.numel() for p in model.parameters())), flush=True)
