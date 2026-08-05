p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old = "        self.A = nn.Parameter(torch.rand(self.d_inner, self.d_state) - 0.5)"
new = "        self.A_log = nn.Parameter(torch.empty(self.d_inner, self.d_state))\n        nn.init.uniform_(self.A_log, a=-5.0, b=-1.0)"
assert old in s
s = s.replace(old, new, 1)
old = "        y = selective_scan_loop(x_masked.float(), dt_masked.float(), self.A.float(),\n                                B_scan.float(), C_scan.float())                # (B, L, D)"
new = "        A_scan = -torch.exp(self.A_log.float())                                  # (D, N), negative eigenvalues\n        y = selective_scan_loop(x_masked.float(), dt_masked.float(), A_scan,\n                                B_scan.float(), C_scan.float())                # (B, L, D)"
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched A init")
