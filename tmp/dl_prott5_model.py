import subprocess, sys, os
out = r"data/models/prot_t5_xl_half_uniref50-enc/pytorch_model.bin"
os.makedirs(os.path.dirname(out), exist_ok=True)
url = "https://hf-mirror.com/Rostlab/prot_t5_xl_half_uniref50-enc/resolve/main/pytorch_model.bin"
proc = subprocess.run(["curl.exe", "-sSL", "-C", "-", "--retry", "5", "--retry-delay", "2", "-o", out, url])
print("curl exit", proc.returncode)
print("final size MB", round(os.path.getsize(out)/1e6, 1))
