p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old_init = "        self.D = nn.Parameter(torch.ones(self.d_inner))"
new_init = "        self.D = nn.Parameter(torch.ones(self.d_inner))\n        self.out_proj = nn.Linear(self.d_inner, self.d_model)"
assert old_init in s
s = s.replace(old_init, new_init, 1)
old_ret = "        return y * z + self.D.unsqueeze(0).unsqueeze(0) * x_masked"
new_ret = "        merged = y * z + self.D.unsqueeze(0).unsqueeze(0) * x_masked\n        return self.out_proj(merged)"
assert old_ret in s
s = s.replace(old_ret, new_ret, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched mamba out_proj")
