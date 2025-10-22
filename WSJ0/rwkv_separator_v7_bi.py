import os
from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import official RWKV-v7 components from the author's file
from RWKV.RWKV_v7.train_temp.src.model import RWKV_Tmix_x070, RWKV_CMix_x070

# ----------------------- Configs -----------------------

@dataclass
class V7Args:
    # minimal arg set required by RWKV_Tmix_x070 / RWKV_CMix_x070
    n_embd: int                            # DAC latent dim=1024
    n_layer: int                           # Number of stacked V7 layers
    dim_att: int                           # Timemix dimensionality (Data dimension inside separator)
    head_size_a: int
    my_testing: str = "x070"               # ensure x070 logic
    head_size_divisor: int = 64            # used in some inits (keep default-compatible)
    pre_ffn: int = 0
    my_pos_emb: int = 0                    # Positional encoding (disabled as v7 handles temporal mixing internally)

@dataclass
class SeparatorV7Config:
    in_dim: int                            # Latent channel dimension=1024 
    layers: int = 6                        # Number of RWKV blocks
    head_size_a: int = 64                  # must divide hidden_dim
    hidden_dim: Optional[int] = None       # if None, auto-adjust to nearest multiple of head_size_a
    dropout: float = 0.0                   # Generic dropout; handled inside v7 blocks via author's design
    dir_drop_p: float = 0.0                # direction dropout in Bi wrapper
    use_mask: bool = True                  # mask+residual heads
    enforce_bf16: bool = True              # cast activations to bf16 for kernel path

# ----------------------- Layers -----------------------

class V7Layer(nn.Module):
    """One x070 layer: PreLN -> Tmix -> +res -> PreLN -> CMix -> +res, with v_first plumbing."""
    def __init__(self, args: V7Args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(args.n_embd)   # keep these in fp32 params
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.tmix = RWKV_Tmix_x070(args, layer_id)
        self.cmix = RWKV_CMix_x070(args, layer_id)

    def _to_bf16_contig(self, t: torch.Tensor) -> torch.Tensor:
        # cast only for fused kernels
        if t.dtype is not torch.bfloat16:
            t = t.to(torch.bfloat16)
        return t.contiguous()

    def forward(self, x: torch.Tensor, v_first: Optional[torch.Tensor]):
        # Keep LN in fp32 (more stable), fuse ops in bf16
        with torch.cuda.amp.autocast(enabled=False):
            # --- TimeMix branch ---
            x1 = self.ln1(x.float())                 # LN in fp32
            x1 = self._to_bf16_contig(x1)            # cast for fused tmix
            h, v_first = self.tmix(x1, v_first)      # [B,T,C]

            x = x + h                                 # first residual

            # --- ChannelMix branch ---
            x2 = self.ln2(x.float())                 # LN on UPDATED x
            x2 = self._to_bf16_contig(x2)            # cast for fused cmix
            x = x + self.cmix(x2)                    # second residual

        return x, v_first

class V7Core(nn.Module):
    """Stack of V7Layer (uni-directional)."""
    def __init__(self, args: V7Args):
        super().__init__()
        self.layers = nn.ModuleList([V7Layer(args, i) for i in range(args.n_layer)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_first = None
        for i, lyr in enumerate(self.layers):
            x, v_first = lyr(x, v_first)
        return x

class BiV7Core(nn.Module):
    """Bidirectional wrapper: run forward and reversed, then fuse. Supports Direction Dropout."""
    def __init__(self, args: V7Args, dir_drop_p: float = 0.0):
        super().__init__()
        self.dir_drop_p = float(dir_drop_p)
        self.fwd = V7Core(args)
        self.bwd = V7Core(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.dir_drop_p > 0.0:
            u = torch.rand((), device=x.device)
            if u < 0.5 * self.dir_drop_p:
                # forward-only
                return self.fwd(x)
            elif u < self.dir_drop_p:
                # backward-only
                xb = torch.flip(x, dims=[1]).contiguous()
                xb = self.bwd(xb)
                return torch.flip(xb, dims=[1]).contiguous()

        # fuse both directions
        xf = self.fwd(x)
        xb = torch.flip(x, dims=[1]).contiguous()
        xb = self.bwd(xb)
        xb = torch.flip(xb, dims=[1]).contiguous()
        return 0.5 * (xf + xb)

# ------------------------ Helpers ------------------------

class SimpleSnake(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha)

    def forward(self, x):
        return x + (1.0 / self.alpha) * torch.sin(self.alpha * x) ** 2

# ----------------------- Separator -----------------------

class RWKVv7Separator(nn.Module):
    """
    Conv1x1 down (C -> H') -> Bi V7 core (L layers) -> Conv1x1 up (H' -> C) -> 2 heads (mask+resid).
    H' is adjusted so that H' is divisible by head_size_a (required by v7 kernel).
    """
    def __init__(self, cfg: SeparatorV7Config):
        super().__init__()
        C = cfg.in_dim
        if cfg.hidden_dim is None:
            Happrox = max(C // 2, cfg.head_size_a)
            H = int(round(Happrox / cfg.head_size_a) * cfg.head_size_a)
        else:
            H = cfg.hidden_dim
            # auto-round up to the next multiple (safer for kernels)
            m = cfg.head_size_a
            if H % m != 0:
                H = ((H + m - 1) // m) * m

        self.cfg = cfg
        self.down = nn.Linear(C, H)
        # Build v7 args for the core
        v7args = V7Args(
            n_embd=H,
            n_layer=cfg.layers,
            dim_att=H,
            head_size_a=cfg.head_size_a,
            my_testing="x070"
        )
        # Ensure env toggles
#       os.environ.setdefault("RWKV_MY_TESTING", "x070")

        self.core = BiV7Core(v7args, dir_drop_p=cfg.dir_drop_p)
        # after creating self.core in RWKVv7Separator.__init__
#       if self.cfg.enforce_bf16 and torch.cuda.is_available():
#           self.core = self.core.to(torch.bfloat16)    # cast Tmix/CMix LNs & linears to bf16
            # ensure downstream-created LNs/Linears are also bf16
#           for m in self.core.modules():
#               if isinstance(m, (nn.LayerNorm, nn.Linear)):
#                   m.to(torch.bfloat16)

        self.up = nn.Linear(H, C)

        # Heads
        head_hidden = max(128, C // 2)
        self.head_r1 = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, head_hidden), SimpleSnake(1.0), nn.Linear(head_hidden, C))
        self.head_r2 = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, head_hidden), SimpleSnake(1.0), nn.Linear(head_hidden, C))
        if cfg.use_mask:
            self.head_m = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, 2*C))
#           self.head_m1 = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C))
#           self.head_m2 = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C))
        else:
            self.head_m = None
#           self.head_m1 = self.head_m2 = None

    def forward(self, z_mix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z_mix: [B,T,C] float32/float16/bfloat16
        Returns pred1, pred2 (and masks/resids if enabled).
        """
        assert z_mix.dim() == 3, f"Expected [B,T,C], got {z_mix.shape}"
        B,T,C = z_mix.shape

        x = self.down(z_mix)

        # The official kernel is fastest in bf16; cast activations if desired.
        if self.cfg.enforce_bf16 and z_mix.device.type == "cuda":
            x = x.to(torch.bfloat16)
        
        x = x.contiguous()  # important for the CUDA kernel

        HC = x.shape[-1]
        assert (HC % self.cfg.head_size_a) == 0, f"[pre-core] HC={HC} must be divisible by head_size_a={self.cfg.head_size_a}"
        
        # Run the core in bf16 to avoid fp16 autocast corrupting the fused kernel inputs
        if self.cfg.enforce_bf16 and x.is_cuda:
            # Either use bf16 autocast explicitly...
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                h = self.core(x)
        else:
            # ...or, if you prefer to be extra strict, disable autocast:
            # with torch.cuda.amp.autocast(enabled=False):
            #     h = self.core(x)
            h = self.core(x)

        if h.dtype != z_mix.dtype:
            h = h.to(z_mix.dtype)

        h = self.up(h)

        r1 = self.head_r1(h)
        r2 = self.head_r2(h)
#        if self.head_m1 is not None:
#           m1 = torch.sigmoid(self.head_m1(h))
#           m2 = torch.sigmoid(self.head_m2(h))
        if self.head_m is not None:
            logits = self.head_m(h).view(B, T, C, 2)
            m = logits.softmax(dim=-1)
            m1, m2 = m[...,0], m[...,1]
            y1 = m1 * z_mix + r1
            y2 = m2 * z_mix + r2
            return {"pred1": y1, "pred2": y2, "mask1": m1, "mask2": m2, "resid1": r1, "resid2": r2}
        else:
            y1 = z_mix + r1
            y2 = z_mix + r2
            return {"pred1": y1, "pred2": y2, "resid1": r1, "resid2": r2}

# ----------------------- Smoke test -----------------------
if __name__ == "__main__":
    # Set env to use x070 path
    os.environ["RWKV_MY_TESTING"] = "x070"
    B,T,C = 2, 320, 128
    cfg = SeparatorV7Config(in_dim=C, layers=4, head_size_a=64, dir_drop_p=0.3)
    model = RWKVv7Separator(cfg).cuda() if torch.cuda.is_available() else RWKVv7Separator(cfg)
    x = torch.randn(B,T,C, device=next(model.parameters()).device)
    out = model(x)
    for k,v in out.items():
        print(k, tuple(v.shape), v.dtype)
