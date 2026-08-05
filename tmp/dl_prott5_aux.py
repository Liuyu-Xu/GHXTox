import subprocess, os
outdir = r"data/models/prot_t5_xl_half_uniref50-enc"
os.makedirs(outdir, exist_ok=True)
base = "https://hf-mirror.com/Rostlab/prot_t5_xl_half_uniref50-enc/resolve/main"
for name in ["spiece.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"]:
    dest = os.path.join(outdir, name)
    if os.path.exists(dest):
        print("skip", name)
        continue
    r = subprocess.run(["curl.exe", "-sSL", "--max-time", "60", "-o", dest, base + "/" + name])
    print(name, "exit", r.returncode, "size", os.path.getsize(dest) if os.path.exists(dest) else 0)
