p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
old = '''        dt = nn.functional.softplus(self.dt_proj(dt))
        # mask padded positions so they do not inject information into the state
        mask = valid.unsqueeze(-1)                                             # (B, L, 1)
        x_masked = x * mask
        dt_masked = dt * mask
        y = selective_scan_loop(x_masked, dt_masked, self.A, B_scan, C_scan)   # (B, L, D)
        merged = y * z + self.D.unsqueeze(0).unsqueeze(0) * x_masked
        return self.out_proj(merged)'''
new = '''        dt = nn.functional.softplus(self.dt_proj(dt)).clamp(max=20.0)
        # mask padded positions so they do not inject information into the state
        mask = valid.unsqueeze(-1)                                             # (B, L, 1)
        x_masked = x * mask
        dt_masked = dt * mask
        # run the scan in fp32 to avoid exp() overflow in fp16, then cast back
        y = selective_scan_loop(x_masked.float(), dt_masked.float(), self.A.float(),
                                B_scan.float(), C_scan.float())                # (B, L, D)
        y = y.to(x.dtype)
        merged = y * z + self.D.unsqueeze(0).unsqueeze(0) * x_masked
        return self.out_proj(merged)'''
assert old in s
s = s.replace(old, new, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched fp32 scan")
