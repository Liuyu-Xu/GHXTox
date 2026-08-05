p = "src/ghxtox/lora_mamba_1d.py"
s = open(p, encoding="utf-8").read()
anchor = "    def forward(self, hidden: torch.Tensor) -> torch.Tensor:"
prop = '''    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

'''
assert anchor in s, "anchor not found"
s = s.replace(anchor, prop + anchor, 1)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("patched")
