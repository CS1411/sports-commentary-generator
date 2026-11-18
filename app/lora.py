"""LoRA from scratch (no peft): a frozen layer plus a trainable low-rank detour."""
import torch
import torch.nn as nn


def _dims(base):
    "nn.Linear has in/out_features; GPT-2's Conv1D stores its weight transposed as [d_in, d_out]."
    if hasattr(base, "in_features"):
        return base.in_features, base.out_features, False
    d_in, d_out = base.weight.shape
    return d_in, d_out, True


class LoRALinear(nn.Module):
    """Frozen base layer + trainable detour B@A, scaled by alpha/r.

    A is small random and B is zero, so the detour starts as an exact no-op and the wrapped layer
    behaves identically to the base until training moves B.
    """

    def __init__(self, base, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        d_in, d_out, self.transposed = _dims(base)
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        detour = (self.dropout(x) @ self.A.T) @ self.B.T
        return self.base(x) + detour * self.scale

    def merged(self):
        "Fold the detour into the frozen weight and return the plain layer."
        with torch.no_grad():
            delta = (self.B @ self.A) * self.scale  # [d_out, d_in]
            self.base.weight += delta.T if self.transposed else delta
        return self.base


def add_lora(model, r=8, alpha=16, dropout=0.05):
    """Freeze the whole GPT-2 model, then wrap the fused Q/K/V projection (c_attn) of every block.

    Freeze first, wrap second: the new A and B are then the only tensors that still want a gradient.
    """
    for p in model.parameters():
        p.requires_grad = False
    for block in model.transformer.h:
        block.attn.c_attn = LoRALinear(block.attn.c_attn, r=r, alpha=alpha, dropout=dropout)
    return model


def adapter_state(model):
    "Just the adapter: the A and B matrices of every LoRALinear."
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}


def count_params(model):
    "(trainable, total) parameter counts."
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def merge_lora(model):
    "Fold every adapter into the base weights and remove the wrappers: a plain GPT-2 again, no extra computation."
    for block in model.transformer.h:
        if isinstance(block.attn.c_attn, LoRALinear):
            block.attn.c_attn = block.attn.c_attn.merged()
    return model
