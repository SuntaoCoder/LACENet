
import torch
import torch.nn as nn
import copy
from typing import Optional

class ModelEMA(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        device: Optional[torch.device] = None,
        fp32_params: bool = False,
        copy_buffers: bool = True,
        warmup_steps: int = 500,
    ):
        super().__init__()
        # deep copy the model to create the EMA shadow
        self.module = copy.deepcopy(model).eval()
        if device is not None:
            self.module.to(device)

        # EMA params are not trainable; optional fp32 master copy
        for p in self.module.parameters():
            p.requires_grad_(False)
            if fp32_params:
                p.data = p.data.float()

        # settings
        self.decay = float(decay)
        self.copy_buffers = copy_buffers
        self.num_updates = 0
        self.warmup_steps = int(warmup_steps)

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.num_updates += 1

        # decay warmup: start smaller, ramp up to target decay
        d = self.decay
        if self.warmup_steps > 0 and self.num_updates <= self.warmup_steps:
            # increases from ~1/warmup_steps up to 1, then capped by self.decay
            d = min(self.decay, (1.0 + self.num_updates) / (self.warmup_steps + self.num_updates))

        # EMA for parameters
        ema_params = dict(self.module.named_parameters())
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            mp = ema_params.get(name, None)
            if mp is None:
                continue  # defensive: param not found (wrappers, etc.)
            src = p.detach()
            if mp.dtype != src.dtype:
                src = src.to(mp.dtype)
            mp.mul_(d).add_(src, alpha=1.0 - d)

        # Copy buffers (e.g., BatchNorm running stats)
        if self.copy_buffers:
            ema_bufs = dict(self.module.named_buffers())
            for name, b in model.named_buffers():
                mb = ema_bufs.get(name, None)
                if mb is None:
                    continue
                src = b
                if mb.dtype != b.dtype:
                    src = b.to(mb.dtype)
                mb.copy_(src)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def save_ckpt(
    path,
    ema,
    model=None,
    optimizer=None,
    scaler=None,
    scheduler=None,
    epoch=0,
    step=0,
    best_mae=None,
    best_counter=0,
):
    ckpt = {
        "ema": ema.module.state_dict(),                   # EMA weights/buffers
        "model": model.state_dict() if model else None,   # raw training weights
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scaler": scaler.state_dict() if scaler else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "step": step,
        "ema_decay": getattr(ema, "decay", None),
        "ema_updates": getattr(ema, "num_updates", 0),
        "ema_warmup_steps": getattr(ema, "warmup_steps", 500),
        "best_mae": best_mae,
        "best_counter": best_counter,
    }
    torch.save(ckpt, path)

def load_ckpt_for_resume(path, model, device):
    ckpt = torch.load(path, map_location=device)

    # 1) restore base model (keeps optimizer graph continuity)
    model_state = ckpt.get("model", None)
    if model_state is None:
        model_state = ckpt["ema"]
    model.load_state_dict(model_state)

    # 2) rebuild EMA from the just-loaded model, then load EMA weights
    ema_decay = ckpt.get("ema_decay", 0.999)
    ema_warmup_steps = ckpt.get("ema_warmup_steps", 500)
    ema = ModelEMA(model, decay=ema_decay, device=device, warmup_steps=ema_warmup_steps)
    ema.module.load_state_dict(ckpt["ema"])
    ema.num_updates = ckpt.get("ema_updates", 0)

    return ema


def load_training_state(path, model, optimizer=None, scheduler=None, device=None, ema_decay=0.999, warmup_steps=500):
    ckpt = torch.load(path, map_location=device)

    model_state = ckpt.get("model", None)
    if model_state is None:
        model_state = ckpt["ema"]
    model.load_state_dict(model_state)

    ema = ModelEMA(
        model,
        decay=ckpt.get("ema_decay", ema_decay),
        device=device,
        warmup_steps=ckpt.get("ema_warmup_steps", warmup_steps),
    )
    if ckpt.get("ema", None) is not None:
        ema.module.load_state_dict(ckpt["ema"])
    ema.num_updates = ckpt.get("ema_updates", 0)

    if optimizer is not None and ckpt.get("optimizer", None) is not None:
        optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler", None) is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    return ema, ckpt
