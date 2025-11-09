# -*- coding: utf-8 -*-
"""
Drop-in replacement for rwkv_separator_DS.py (FULL implementation)

What this script does
- Uses the **full** RWKV‑v7 parameterization & projections (aligned with model_original.py)
- Implements the **bidirectional concat+flip** single‑launch CUDA path like your working rwkv7.py
- Keeps RWKV‑v7 post‑kernel pieces: GroupNorm on x (ln_x), the (r⊙k)·v additive term with r_k, and the output gate g
- Corrects time‑shift to true 1D shift on [B,T,C] (no ZeroPad2d misuse)
- Keeps LayerNorm/GroupNorm in **fp32**; wraps the CUDA call in **bf16 autocast** (toggle via cfg.enforce_bf16)
- Pads T to multiple of 16; enforces C mod 64 = 0 (x070 kernels)
- Separation heads end with **Snake** activation (DAC / CodecFormer‑EL family)

Public API
- build_rwkv7_separator(...)-> RWKVv7Separator
- Forward: x:[B,T,C] (DAC latents) → y:[B,T,num_sources,C]

"""
from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Snake activation (DAC / CodecFormer family)
# -----------------------------------------------------------------------------
try:
    from dac.nn.layers import Snake1d  # type: ignore
except Exception:
    class Snake1d(nn.Module):
        def __init__(self, channels: int, alpha: float = 1.0):
            super().__init__()
            self.alpha = nn.Parameter(torch.ones(1, 1, channels) * alpha)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            a = self.alpha
            return x + (torch.sin(a * x) ** 2) / a.clamp_min(1e-4)


# Channels-last compatible Snake activation wrapper
class SnakeChannelsLast(nn.Module):
    """
    Snake activation adapter for channels-last format [B, T, C].
    
    The standard Snake1d expects [B, C, T] format (Conv1d-style),
    but we're working with [B, T, C] (Linear/LayerNorm-style).
    This adapter handles the transpose internally.
    """
    def __init__(self, channels):
        super().__init__()
        self.snake = Snake1d(channels)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C] tensor
        Returns:
            [B, T, C] tensor with Snake activation applied
        """
        # Transpose to [B, C, T] for Snake1d
        x = x.transpose(1, 2)  # [B, T, C] -> [B, C, T]
        
        # Apply Snake activation  
        x = self.snake(x)  # [B, C, T]
        
        # Transpose back to [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        
        return x

# -----------------------------------------------------------------------------
# Import the fused RWKV‑v7 CUDA runner (multiple fallbacks)
# -----------------------------------------------------------------------------
RUN_CUDA = None
_run_import_err: Optional[str] = None
try:
    from rwkv7_cuda_bindings import RUN_CUDA_RWKV7g as RUN_CUDA  # type: ignore
except Exception as e1:
    try:
        from rwkv7 import RUN_CUDA_RWKV7g as RUN_CUDA  # type: ignore
    except Exception as e2:
        try:
            from rwkv.ops import RUN_CUDA_RWKV7g as RUN_CUDA  # type: ignore
        except Exception as e3:
            _run_import_err = f"Could not import RUN_CUDA_RWKV7g: {e1} | {e2} | {e3}"
            RUN_CUDA = None

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def time_shift_1(x: torch.Tensor) -> torch.Tensor:
    """Temporal shift for [B,T,C]: prepend one zero frame, drop last."""
    return torch.cat([x.new_zeros(x.size(0), 1, x.size(2)), x[:, :-1, :]], dim=1)


def pad_T_to_multiple(x: torch.Tensor, m: int) -> Tuple[torch.Tensor, int]:
    T = x.size(1)
    rem = T % m
    if rem == 0:
        return x, 0
    pad = m - rem
    return F.pad(x, (0, 0, 0, pad)), pad

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class SeparatorV7Config:
    n_embd: int = 512
    n_layer: int = 8
    head_size_a: int = 64            # x070: 64‑aligned
    enforce_bf16: bool = True        # bf16 autocast on CUDA
    num_sources: int = 2             # number of speakers
    head_hidden: int = 256
    head_mode: str = "residual"       # "residual" or "mask"

# -----------------------------------------------------------------------------
# FULL RWKV‑v7 TimeMix/ChannelMix parameters (aligned with model_original.py)
# We reproduce the parameterization & math (x_r/x_w/... w1,w2,w0; a1,a2,a0; v1,v2,v0; g1,g2; k_k,k_a; r_k)
# Then we call the fused CUDA kernel in a **bidirectional** manner (concat+flip single launch) and finish with
# ln_x (GroupNorm), the additive (r⊙k)·v term with r_k, and the output gate g.
# -----------------------------------------------------------------------------
class Full_Tmix_Params(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size: int):
        super().__init__()
        self.layer_id = layer_id
        self.n_embd = n_embd
        self.head_size = head_size
        assert n_embd % head_size == 0
        H = n_embd // head_size
        N = head_size
        C = n_embd

        ratio_0_to_1 = layer_id / max(1, (n_layer - 1))
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))

        ddd = torch.ones(1, 1, C)
        for i in range(C):
            ddd[0, 0, i] = i / C

        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        def ortho_init(x, scale):
            with torch.no_grad():
                shape = x.shape
                if len(shape) == 2:
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                elif len(shape) == 3:
                    gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                    for i in range(shape[0]):
                        nn.init.orthogonal_(x[i], gain=gain * scale)
                else:
                    raise RuntimeError("bad shape for ortho_init")
                return x

        www = torch.zeros(C)
        zigzag = torch.zeros(C)
        linear = torch.zeros(C)
        for n in range(C):
            linear[n] = n / (C - 1) - 0.5
            zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag[n] = zigzag[n] * abs(zigzag[n])
            www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

        D_DECAY_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
        self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

        D_AAA_LORA = max(32, int(round((2.5 * (C ** 0.5)) / 32) * 32))
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
        self.a0 = nn.Parameter(torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4)

        D_MV_LORA = max(32, int(round((1.7 * (C ** 0.5)) / 32) * 32))
        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

        D_GATE_LORA = max(32, int(round((5 * (C ** 0.5)) / 32) * 32))
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
        self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(n_embd // head_size, head_size) - 0.04)  # [H,N]

        # Projections & norms
        self.receptance = nn.Linear(C, C, bias=False)
        self.key        = nn.Linear(C, C, bias=False)
        self.value      = nn.Linear(C, C, bias=False)
        self.output     = nn.Linear(C, C, bias=False)
        self.ln_x       = nn.GroupNorm(n_embd // head_size, C, eps=64e-5)

        # Init per original design
        self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
        self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.output.weight.data.zero_()

    @torch.no_grad()
    def _sanity(self):
        pass

    def project(self, x: torch.Tensor, v_first: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute r,w,k,v,a,g,kk & updated v_first using the full RWKV‑v7 formulae."""
        B, T, C = x.shape
        # time context features
        xx = time_shift_1(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        H = self.n_embd // self.head_size
        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), p=2.0, dim=-1).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        return r, w, k, v, a, g, kk, v_first

# -----------------------------------------------------------------------------
# Bidirectional TimeMix (full): concat+flip single CUDA, then ln_x, (r⊙k)·v term, output*g
# -----------------------------------------------------------------------------
class BiTimeMixFull(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size_a: int):
        super().__init__()
        self.params = Full_Tmix_Params(n_embd, n_layer, layer_id, head_size=head_size_a)
        self.n_embd = n_embd
        self.head_size = head_size_a
        self.n_head = n_embd // head_size_a
        # depthwise head‑gate to fuse fwd/bwd (rwkv7.py style)
        self.head_gate = nn.Conv1d(n_embd, self.n_head, kernel_size=1, groups=self.n_head, bias=False)

    def forward(self, x: torch.Tensor, v_first: torch.Tensor, *, bf16_enabled: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if RUN_CUDA is None:
            raise RuntimeError(_run_import_err or "RUN_CUDA_RWKV7g is not available")
        B, T, C = x.shape
        assert C % 64 == 0, "n_embd must be multiple of 64 for x070 kernels"
        assert T % 16 == 0, "T must be multiple of 16 (CHUNK_LEN)"

        # full projections per original RWKV‑v7
        r, w, k, v, a, g, kk, v_first = self.params.project(x.float(), v_first)

        # concat+flip single CUDA launch (works in bf16 under autocast)
        use_cuda = x.is_cuda
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda and bf16_enabled):
            rf, rb = r, torch.flip(r, dims=[1])
            wf, wb = w, torch.flip(w, dims=[1])
            kf, kb = k, torch.flip(k, dims=[1])
            vf, vb = v, torch.flip(v, dims=[1])
            af, ab = -kk, torch.flip(-kk, dims=[1])
            bf, bb = (kk * a), torch.flip(kk * a, dims=[1])

            y_cat = RUN_CUDA(
                torch.cat([rf, rb], dim=0).contiguous(),
                torch.cat([wf, wb], dim=0).contiguous(),
                torch.cat([kf, kb], dim=0).contiguous(),
                torch.cat([vf, vb], dim=0).contiguous(),
                torch.cat([af, ab], dim=0).contiguous(),
                torch.cat([bf, bb], dim=0).contiguous(),
            )  # [2B,T,C]

            y_f, y_b = torch.chunk(y_cat.view(2 * B, T, C), chunks=2, dim=0)

            # head‑wise fuse forward/backward (sigmoid gate from local time‑features)
            xx = time_shift_1(x.float()) - x.float()
            gate = torch.sigmoid(self.head_gate(xx.transpose(1, 2))).transpose(1, 2)  # [B,T,H]
            y = (
                gate.unsqueeze(-1) * y_f.view(B, T, self.n_head, -1)
                + (1.0 - gate).unsqueeze(-1) * torch.flip(y_b, dims=[1]).view(B, T, self.n_head, -1)
            ).contiguous().view(B, T, C)

        # Post‑kernel pieces as in RWKV‑v7
        y = self.params.ln_x(y.view(B * T, C)).view(B, T, C)
        # add (r⊙k)·v using forward r,k,v (keeps original modeling intent)
        y = y + ((r.view(B, T, self.n_head, -1) * k.view(B, T, self.n_head, -1) * self.params.r_k)
                 .sum(dim=-1, keepdim=True) * v.view(B, T, self.n_head, -1)).view(B, T, C)
        y = self.params.output(y * g)
        return y, v_first

# -----------------------------------------------------------------------------
# ChannelMix (full, with corrected time‑shift)
# -----------------------------------------------------------------------------
class Full_CMix(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int):
        super().__init__()
        self.n_embd = n_embd
        C = n_embd
        ratio_1_to_almost0 = 1.0 - (layer_id / max(1, n_layer))
        ddd = torch.ones(1, 1, C)
        for i in range(C):
            ddd[0, 0, i] = i / C
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4))
        self.key = nn.Linear(C, C * 4, bias=False)
        self.value = nn.Linear(C * 4, C, bias=False)
        self.key.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.value.weight.data.zero_()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = time_shift_1(x.float()) - x.float()
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)

# -----------------------------------------------------------------------------
# Blocks & Core
# -----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, layer_id: int, head_size_a: int, bf16: bool):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.tmix = BiTimeMixFull(n_embd, n_layer, layer_id, head_size_a)
        self.cmix = Full_CMix(n_embd, n_layer, layer_id)
        self.bf16 = bf16
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(n_embd)
    def forward(self, x: torch.Tensor, v_first: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layer_id == 0:
            x = self.ln0(x)
        x_attn, v_first = self.tmix(self.ln1(x), v_first, bf16_enabled=self.bf16)
        x = x + x_attn
        x = x + self.cmix(self.ln2(x))
        return x, v_first

class V7Core(nn.Module):
    def __init__(self, n_embd: int, n_layer: int, head_size_a: int, bf16: bool):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(n_embd, n_layer, i, head_size_a, bf16) for i in range(n_layer)
        ])
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v_first = torch.empty_like(x)
        for blk in self.blocks:
            x, v_first = blk(x, v_first)
        return x

# -----------------------------------------------------------------------------
# Separation head with Snake final activation
# -----------------------------------------------------------------------------
class SeparationHead(nn.Module):
    def __init__(self, dim: int, hidden: int, num_sources: int, mode: str = "residual"):
        super().__init__()
        assert mode in ("residual", "mask")
        self.mode = mode
        self.num_sources = num_sources
        self.pre = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
        )
        self.post = nn.Sequential(
            nn.LayerNorm(hidden),
            SnakeChannelsLast(hidden),
            nn.Linear(hidden, num_sources * dim, bias=False),
        )
    def forward(self, x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        y = self.post(h).view(x.size(0), x.size(1), self.num_sources, -1)  # [B,T,S,C]
        if self.mode == "mask":
            m = torch.sigmoid(y)
            return m * x_ref.unsqueeze(2)
        return x_ref.unsqueeze(2) + y

# -----------------------------------------------------------------------------
# Top‑level separator
# -----------------------------------------------------------------------------
class RWKVv7Separator(nn.Module):
    def __init__(self, cfg: SeparatorV7Config):
        super().__init__()
        self.cfg = cfg
        C = cfg.n_embd
        assert C % cfg.head_size_a == 0, "n_embd must be multiple of head_size_a"
        self.core = V7Core(C, cfg.n_layer, cfg.head_size_a, cfg.enforce_bf16)
        self.head = SeparationHead(C, cfg.head_hidden, cfg.num_sources, mode=cfg.head_mode)

    @staticmethod
    def _shape_checks(x: torch.Tensor) -> Tuple[int, int, int]:
        assert x.dim() == 3, "Expect [B,T,C]"
        B, T, C = x.shape
        assert C % 64 == 0, "n_embd must be multiple of 64 for x070 kernels"
        return B, T, C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = self._shape_checks(x)
        # pad T to multiple of 16 for the CUDA kernel
        x_pad, pad = pad_T_to_multiple(x, 16)
        h = self.core(x_pad)
        if pad:
            h = h[:, :-pad, :]
        y = self.head(h, x)
        return y  # [B,T,S,C]

# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def build_rwkv7_separator(
    n_embd: int,
    n_layer: int,
    num_sources: int = 2,
    *,
    head_size_a: int = 64,
    head_hidden: int = 256,
    head_mode: str = "residual",
    enforce_bf16: bool = True,
) -> RWKVv7Separator:
    cfg = SeparatorV7Config(
        n_embd=n_embd,
        n_layer=n_layer,
        head_size_a=head_size_a,
        enforce_bf16=enforce_bf16,
        num_sources=num_sources,
        head_hidden=head_hidden,
        head_mode=head_mode,
    )
    model = RWKVv7Separator(cfg)
    if RUN_CUDA is None:
        warnings.warn(_run_import_err or "RUN_CUDA_RWKV7g missing — forward() will raise",
                      RuntimeWarning)
    return model

# -----------------------------------------------------------------------------
# Smoke test (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    B, T, C = 2, 160, 512  # T multiple of 16, C multiple of 64
    x = torch.randn(B, T, C)
    m = build_rwkv7_separator(n_embd=C, n_layer=2, num_sources=2, enforce_bf16=False)
    try:
        y = m(x)
        print("OK:", tuple(y.shape))
    except Exception as e:
        print("Forward error:", e)
