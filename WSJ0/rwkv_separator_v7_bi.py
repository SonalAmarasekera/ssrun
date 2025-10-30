# rwkv_separator_v7_bi_codecformer.py
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from RWKV.RWKV_v7.train_temp.src.model import RWKV_Tmix_x070, RWKV_CMix_x070

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
    num_spks: int = 2                   # number of speakers
    enforce_bf16: bool = True          # activations in bf16 at fused ops

# ----------------------- Layers -----------------------
class V7Layer(nn.Module):
    """One x070 layer: PreLN -> TimeMix -> +res -> PreLN -> ChannelMix -> +res, with v_first plumbing."""

    def __init__(self, args, layer_id: int):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        # Keep LN weights in fp32 for numerical stability
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        self.tmix = RWKV_Tmix_x070(args, layer_id)
        self.cmix = RWKV_CMix_x070(args, layer_id)

    @staticmethod
    def _to_bf16_contig(t: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is bf16 and contiguous (required for fused CUDA kernels)."""
        if t.is_cuda and t.dtype != torch.bfloat16:
            t = t.to(torch.bfloat16)
        return t.contiguous()

    def forward(self, x: torch.Tensor, v_first: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for a single RWKV-v7 layer.
        - Keeps LN in fp32.
        - Ensures TimeMix and ChannelMix receive bf16 tensors.
        - Prevents mixed-precision mismatch for CUDA kernel inputs (w,q,k,v,z,b).
        """
        use_cuda = x.is_cuda

        # --- TimeMix ---
        # LayerNorm in fp32 for stability
        x1 = self.ln1(x.float())
        # Convert to bf16 before feeding to fused kernel
        x1 = self._to_bf16_contig(x1)

        # Run TimeMix in bf16 (disable autocast interference)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
            h, v_first = self.tmix(x1, v_first)
        x = x + h  # residual connection

        # --- ChannelMix ---
        x2 = self.ln2(x.float())
        x2 = self._to_bf16_contig(x2)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_cuda):
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

# ---------------------- Helper --------------------------
def pad_to_chunk(x, chunk_len=16):
    B, T, C = x.shape
    pad = (chunk_len - (T % chunk_len)) % chunk_len
    if pad > 0:
        pad_tensor = torch.zeros(B, pad, C, dtype=x.dtype, device=x.device)
        x = torch.cat([x, pad_tensor], dim=1)
    return x

# ----------------------- Separator -----------------------

class RWKVv7Separator(nn.Module):
    """
    CodecFormer-style separation with RWKV-v7 backbone.
    Conv1x1 down (C -> H) -> Bi V7 core (L layers) -> Conv1x1 up (H -> C) -> Gated separation heads.
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
        self.num_spks = cfg.num_spks

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

        # CodecFormer-style gated separation heads
        # Initial projection to speaker-specific feature bases
        self.masker = nn.Conv1d(C, C * self.num_spks, 1, bias=False)
        nn.utils.weight_norm(self.masker)
        
        # Gated output components (per speaker)
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C, C, 1, bias=False),
                SimpleSnake(1.0)
            ) for _ in range(self.num_spks)
        ])
        
        self.output_gate_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(C, C, 1, bias=False),
                nn.Sigmoid()
            ) for _ in range(self.num_spks)
        ])
        
        # Final activation
        self.activation = SimpleSnake(1.0)

    def _pre_core_cast(self, x: torch.Tensor) -> torch.Tensor:
        # The fused kernels are optimized for bf16 on CUDA. Cast activations (not params).
        if self.cfg.enforce_bf16 and x.is_cuda and x.dtype is not torch.bfloat16:
            x = x.to(torch.bfloat16)
        return x.contiguous()

    def forward(self, z_mix: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z_mix: [B,T,C] float{32|16|bf16}
        Returns dict with separated sources in CodecFormer style.
        """
        assert z_mix.dim() == 3, f"Expected [B,T,C], got {tuple(z_mix.shape)}"
        B, T0, C = z_mix.shape
        assert C == self.in_dim, f"Input C={C} != configured C={self.in_dim}. Re-cache or reconfigure."

        # Down-project to hidden (H). Keep kernel contract: H % head_size_a == 0
        x = self.down(z_mix)                          # [B, T0, H]
        x = self._pre_core_cast(x)                    # if you keep bf16 activations for the core
        H = x.shape[-1]
        assert (H % self.cfg.head_size_a) == 0, f"HC={H} not divisible by head_size_a={self.cfg.head_size_a}"

        # Pad T up to CHUNK_LEN=16 for the fused kernel, then run the core
        x = pad_to_chunk(x, chunk_len=16)             # -> [B, T_pad, H], T_pad >= T0 and T_pad % 16 == 0

        # Core: enforce bf16 autocast ONLY for the kernel path (LN stays fp32 inside V7Layer)
        if self.cfg.enforce_bf16 and x.is_cuda:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = self.core(x)                      # [B, T_pad, H]
        else:
            h = self.core(x)

        # Trim back to the original time length BEFORE heads / up-proj
        if h.size(1) != T0:
            h = h[:, :T0, :].contiguous()             # [B, T0, H]

        # Up-project back to codec latent dim C
        h = self.up(h)                                # [B, T0, C]

        # (Optional) align dtype with input for residual math
        if h.dtype != z_mix.dtype:
            h = h.to(z_mix.dtype)

        # CodecFormer-style separation
        # Convert to [B, C, T] for conv1d operations
        h_conv = h.transpose(1, 2).contiguous()       # [B, C, T0]
        
        # Generate speaker-specific feature bases
        masks = self.masker(h_conv)                   # [B, C * num_spks, T0]
        
        # Reshape for parallel processing across speakers
        B, CT, L = masks.shape
        masks = masks.view(B * self.num_spks, -1, L)  # [B * num_spks, C, T0]
        
        # Apply gated transformation to all speakers in parallel
        all_sources = []
        for i in range(self.num_spks):
            # Get the slice for this speaker
            speaker_slice = masks[i::self.num_spks]   # [B, C, T0] for speaker i
            
            # Apply gated output: content * gate
            content = self.output_heads[i](speaker_slice)
            gate = self.output_gate_heads[i](speaker_slice)
            source_out = content * gate
            source_out = self.activation(source_out)
            all_sources.append(source_out)
        
        # Stack and reshape to final output format
        sources_stacked = torch.stack(all_sources, dim=0)  # [num_spks, B, C, T0]
        sources_stacked = sources_stacked.transpose(0, 1)  # [B, num_spks, C, T0]
        sources_stacked = sources_stacked.transpose(2, 3)  # [B, num_spks, T0, C]
        
        # Convert to same format as original output
        pred1 = sources_stacked[:, 0, :, :]  # [B, T0, C]
        pred2 = sources_stacked[:, 1, :, :]  # [B, T0, C]
        
        return {
            "pred1": pred1, 
            "pred2": pred2,
            # For compatibility, return empty masks and residuals
            "mask1": torch.zeros_like(pred1),
            "mask2": torch.zeros_like(pred2),
            "resid1": torch.zeros_like(pred1),
            "resid2": torch.zeros_like(pred2),
        }

# ----------------------- Smoke test -----------------------
if __name__ == "__main__":
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")

    B, T, C = 2, 320, 128
    cfg = SeparatorV7Config(in_dim=C, layers=4, head_size_a=64, dir_drop_p=0.3, num_spks=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RWKVv7Separator(cfg).to(device)
    x = torch.randn(B, T, C, device=device)
    out = model(x)
    for k, v in out.items():
        print(k, tuple(v.shape), v.dtype)
