# rwkv_separator_v7_bi.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

# --- Import official RWKV-v7 components (preferred path from the author).
#     Fallback to a local module name if your project lays it out differently.
try:
    from RWKV.RWKV_v7.train_temp.src.model import RWKV_Tmix_x070, RWKV_CMix_x070
except Exception as e:
    try:
        # local fallback (adjust if your local filename differs)
        from rwkv_orig_model import RWKV_Tmix_x070, RWKV_CMix_x070  # noqa: F401
    except Exception as e2:
        raise ImportError(
            "Unable to import RWKV_Tmix_x070 / RWKV_CMix_x070 from the official repo or local fallback.\n"
            "Make sure the RWKV-v7 (x070) source is available in PYTHONPATH."
        ) from e

# ----------------------- Configs -----------------------

@dataclass
class V7Args:
    """Minimal arg set required by RWKV_Tmix_x070 / RWKV_CMix_x070."""
    n_embd: int                 # internal channel dim
    n_layer: int                # number of stacked layers
    dim_att: int                # timemix inner dim (use n_embd)
    head_size_a: int            # head size must divide n_embd
    my_testing: str = "x070"    # ensure x070 logic in the kernels
    head_size_divisor: int = 64 # kept default-compatible
    pre_ffn: int = 0
    my_pos_emb: int = 0

@dataclass
class SeparatorV7Config:
    in_dim: int                 # latent channel dim from DAC (e.g., 1024)
    layers: int = 6
    head_size_a: int = 64
    hidden_dim: Optional[int] = None   # auto -> round to multiple of head_size_a
    dir_drop_p: float = 0.0            # direction dropout prob
    use_mask: bool = True              # mask + residual heads
    enforce_bf16: bool = True          # activations in bf16 at fused ops

# ----------------------- Layers -----------------------

class V7Layer(nn.Module):
    """One x070 layer: PreLN -> TimeMix -> +res -> PreLN -> ChannelMix -> +res, with v_first plumbing."""
    def __init__(self, args: V7Args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(args.n_embd)      # params kept in fp32
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.tmix = RWKV_Tmix_x070(args, layer_id)
        self.cmix = RWKV_CMix_x070(args, layer_id)

    @staticmethod
    def _to_bf16_contig(t: torch.Tensor) -> torch.Tensor:
        # Cast to bf16 only for the fused kernels; keep contiguous for CUDA.
        if t.is_cuda and t.dtype is not torch.bfloat16:
            t = t.to(torch.bfloat16)
        return t.contiguous()

    def forward(self, x: torch.Tensor, v_first: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Keep LN in fp32 for stability. Cast to bf16 only at fused TimeMix/ChannelMix boundaries.
        with torch.cuda.amp.autocast(enabled=False):
            # --- TimeMix ---
            x1 = self.ln1(x.float())
            x1 = self._to_bf16_contig(x1)
            h, v_first = self.tmix(x1, v_first)   # [B,T,C]
            x = x + h

            # --- ChannelMix ---
            x2 = self.ln2(x.float())
            x2 = self._to_bf16_contig(x2)
            x = x + self.cmix(x2)

        return x, v_first

class V7Core(nn.Module):
    """Uni-directional stack of V7Layer."""
    def __init__(self, args: V7Args):
        super().__init__()
        self.layers = nn.ModuleList([V7Layer(args, i) for i in range(args.n_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_first: Optional[torch.Tensor] = None
        for lyr in self.layers:
            x, v_first = lyr(x, v_first)
        return x

class BiV7Core(nn.Module):
    """Bidirectional wrapper with optional Direction Dropout.
       If dropped, runs only forward or backward path; otherwise fuses both."""
    def __init__(self, args: V7Args, dir_drop_p: float = 0.0):
        super().__init__()
        self.dir_drop_p = float(dir_drop_p)
        self.fwd = V7Core(args)
        self.bwd = V7Core(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.dir_drop_p > 0.0:
            u = torch.rand((), device=x.device)
            if u < 0.5 * self.dir_drop_p:
                return self.fwd(x)
            if u < self.dir_drop_p:
                xb = torch.flip(x, dims=[1]).contiguous()
                xb = self.bwd(xb)
                return torch.flip(xb, dims=[1]).contiguous()

        xf = self.fwd(x)
        xb = torch.flip(x, dims=[1]).contiguous()
        xb = self.bwd(xb)
        xb = torch.flip(xb, dims=[1]).contiguous()
        return 0.5 * (xf + xb)

# ------------------------ Heads ------------------------

class SimpleSnake(nn.Module):
    """Scalar-parameterized Snake for [B,T,C] tensors (matches Codec/DAC-style nonlinearity)."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.alpha + 1e-9
        return x + (1.0 / a) * torch.sin(a * x) ** 2

# ----------------------- Separator -----------------------

class RWKVv7Separator(nn.Module):
    """
    Conv1x1 down (C -> H) -> Bi V7 core (L layers) -> Conv1x1 up (H -> C) -> 2 heads (mask+resid).
    H is adjusted so that H % head_size_a == 0 (required by v7 kernels).
    """
    def __init__(self, cfg: SeparatorV7Config):
        super().__init__()

        # Keep the kernel env in sync with the configuration (compile-once safety).
        os.environ.setdefault("RWKV_MY_TESTING", "x070")
        os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
        os.environ["RWKV_HEAD_SIZE_A"] = str(cfg.head_size_a)

        C = int(cfg.in_dim)
        if cfg.hidden_dim is None:
            # pick >= C//2 and round to multiple of head_size_a
            Happrox = max(C // 2, cfg.head_size_a)
            H = (Happrox + cfg.head_size_a - 1) // cfg.head_size_a * cfg.head_size_a
        else:
            H = int(cfg.hidden_dim)
            if H % cfg.head_size_a != 0:
                H = (H + cfg.head_size_a - 1) // cfg.head_size_a * cfg.head_size_a  # round up safely

        self.cfg = cfg
        self.in_dim = C
        self.hid_dim = H

        # 1x1 conv over channels = Linear
        self.down = nn.Linear(C, H)

        # RWKV-v7 core
        v7args = V7Args(
            n_embd=H,
            n_layer=cfg.layers,
            dim_att=H,
            head_size_a=cfg.head_size_a,
            my_testing="x070",
        )
        self.core = BiV7Core(v7args, dir_drop_p=cfg.dir_drop_p)

        self.up = nn.Linear(H, C)

        # Heads (residual + mask)
        head_hidden = max(128, C // 2)
        act = SimpleSnake(1.0)
        self.head_r1 = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, head_hidden), act, nn.Linear(head_hidden, C))
        self.head_r2 = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, head_hidden), act, nn.Linear(head_hidden, C))

        self.use_mask = bool(cfg.use_mask)
        if self.use_mask:
            # Predict per-source logits then softmax over the 2-class "source" axis
            self.head_m = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2 * C))
        else:
            self.head_m = None

    def _pre_core_cast(self, x: torch.Tensor) -> torch.Tensor:
        # The fused kernels are optimized for bf16 on CUDA. Cast activations (not params).
        if self.cfg.enforce_bf16 and x.is_cuda and x.dtype is not torch.bfloat16:
            x = x.to(torch.bfloat16)
        return x.contiguous()

    def forward(self, z_mix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z_mix: [B,T,C] float{32|16|bf16}
        Returns dict with pred1, pred2, and optionally mask1/2 + resid1/2.
        """
        assert z_mix.dim() == 3, f"Expected [B,T,C], got {tuple(z_mix.shape)}"
        B, T, C = z_mix.shape
        assert C == self.in_dim, f"Input C={C} != configured C={self.in_dim}. Re-cache or reconfigure."

        # Down-project
        x = self.down(z_mix)
        x = self._pre_core_cast(x)

        # Sanity for kernel: hid_dim must be multiple of head_size_a
        HC = x.shape[-1]
        assert (HC % self.cfg.head_size_a) == 0, f"HC={HC} not divisible by head_size_a={self.cfg.head_size_a}"

        # Core (bf16 activations if CUDA), LN kept fp32 inside V7Layer
        if self.cfg.enforce_bf16 and x.is_cuda:
            with torch.cuda.amp.autocast("cuda", dtype=torch.bfloat16):
                h = self.core(x)
        else:
            h = self.core(x)

        # Up-project (compute likely happens in the autocast dtype)
        h = self.up(h)

        # Residual heads
        r1 = self.head_r1(h)
        r2 = self.head_r2(h)

        if self.use_mask:
            # [B,T,2C] -> [B,T,C,2] -> softmax over 2
            logits = self.head_m(h).reshape(B, T, C, 2)
            m = torch.softmax(logits, dim=-1)
            m1, m2 = m[..., 0], m[..., 1]
            y1 = m1 * z_mix + r1
            y2 = m2 * z_mix + r2
            return {
                "pred1": y1, "pred2": y2,
                "mask1": m1, "mask2": m2,
                "resid1": r1, "resid2": r2,
            }
        else:
            y1 = z_mix + r1
            y2 = z_mix + r2
            return {
                "pred1": y1, "pred2": y2,
                "resid1": r1, "resid2": r2,
            }

# ----------------------- Smoke test -----------------------
if __name__ == "__main__":
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")

    B, T, C = 2, 320, 128
    cfg = SeparatorV7Config(in_dim=C, layers=4, head_size_a=64, dir_drop_p=0.3, use_mask=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RWKVv7Separator(cfg).to(device)
    x = torch.randn(B, T, C, device=device)
    out = model(x)
    for k, v in out.items():
        print(k, tuple(v.shape), v.dtype)
