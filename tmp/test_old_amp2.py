import torch
# exactly the fallback branch for older torch:
from torch.cuda.amp import GradScaler as _GradScaler
def _make_scaler(enabled):
    return _GradScaler(enabled=enabled)
def _amp_autocast(enabled):
    return torch.cuda.amp.autocast(enabled=enabled)

x = torch.randn(64, 128, device="cuda")
w = torch.nn.Linear(128, 1).cuda()
opt = torch.optim.AdamW(w.parameters(), lr=1e-3)
scaler = _make_scaler(True)
for _ in range(3):
    opt.zero_grad()
    with _amp_autocast(True):
        y = w(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y.squeeze(-1), torch.rand(64).cuda())
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
print("old-amp fallback path OK, loss:", float(loss))
