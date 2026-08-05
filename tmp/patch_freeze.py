p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old = """    \"\"\"Wrap every nn.Linear in the encoder in-place with LoRALinear.  Returns the number wrapped.\"\"\"
    wrapped = 0"""
new = """    \"\"\"Wrap every nn.Linear in the encoder in-place with LoRALinear.  Returns the number wrapped.\"\"\"
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    wrapped = 0"""
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched freeze")
