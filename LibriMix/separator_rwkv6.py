import torch
import torch.nn as nn
from dac.nn.layers import Snake1d as Snake           # ✅ single import
from rwkv6_wrapper import RWKV6Wrapper               # wrapper above


class RWKV6Separator(nn.Module):
    def __init__(self, codec, depth=4, n_spk=2, down_ratio=2):
        super().__init__()
        self.codec = codec.eval().requires_grad_(False)

        # 1️⃣ Detect latent width from the codec
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 16000, device=next(codec.parameters()).device)
            latent_dim = self.codec.encode(dummy)[0].shape[1]   # 1024 for 16-kHz DAC

        hidden_dim = latent_dim // down_ratio
        self.down   = nn.Conv1d(latent_dim, hidden_dim, 1)
        self.rwkv   = RWKV6Wrapper(depth=depth, dim=hidden_dim)
        self.up     = nn.Conv1d(hidden_dim, latent_dim, 1)

        # 2️⃣ Per-speaker heads
        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(latent_dim, latent_dim, 1),
                Snake(latent_dim)
            )
            for _ in range(n_spk)
        ])

    # --------------------------------------------------------
    # forward
    # --------------------------------------------------------
    def forward(self, mix_wave, state=None):
        """
        mix_wave : [B, 1, T]  – raw waveform
        Returns   : list[[B, C, F]], new_state
        """
        # Keep codec in FP32 even inside autocast ➜ avoid Half / Float mismatch
        z = self.codec.encode(mix_wave.float())[0]        # [B,C,F], FP32
        z = z.to(self.down.weight.dtype)                  # cast to model dtype (fp16 or fp32)

        y = self.down(z)
        y, state = self.rwkv(y, state)
        y = self.up(y)

        latents = [head(y) for head in self.head]         # list length = n_spk
        return latents, state
