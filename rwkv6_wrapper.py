# models/rwkv6_wrapper.py
import torch, torch.nn as nn
from rwkv6_layer_block import RWKV6LayerBlock
from rwkv6_block_config_map import RWKV6BlockConfigMap

#class RWKV6Wrapper(nn.Module):
"""
    A stack of L RWKV6LayerBlock modules that accepts / returns
    [B, C, T] latents (codec format) and keeps streaming state.

    def __init__(self, depth=4, dim=64):
        super().__init__()
        cfg = RWKV6BlockConfigMap(
            num_hidden_layers=depth,
            hidden_size=dim
        )
        self.blocks = nn.ModuleList([
            RWKV6LayerBlock(cfg.new_block_config_map(layer_id=i))
            for i in range(depth)
        ])
"""
class RWKV6Wrapper(nn.Module):
    def __init__(self, depth=4, dim=64):
        super().__init__()
        cfg = RWKV6BlockConfigMap(
            num_hidden_layers=depth,
            hidden_size=dim,
            dtype="float32"          # ← ensure all sub-modules stay FP32
        )
        self.blocks = nn.ModuleList([
            RWKV6LayerBlock(cfg.new_block_config_map(layer_id=i))
            for i in range(depth)
        ])

    def forward(self, x, states=None):
        # x : [B, C, T] → [B, T, C]
        x = x.permute(0, 2, 1)
        if states is None:
            states = [
                (
                    torch.zeros(x.size(0), self.blocks[0].configMap.hidden_size,  device=x.device, dtype=x.dtype),      # shift
                    torch.zeros(x.size(0), self.blocks[0].configMap.hidden_size // 64, 64, 64, device=x.device, dtype=x.dtype), # wkv
                    torch.zeros(x.size(0), self.blocks[0].configMap.hidden_size,  device=x.device, dtype=x.dtype)       # shift cmix
                ) for _ in self.blocks
            ]
        new_states = [None] * len(self.blocks)
        for i, blk in enumerate(self.blocks):
            x, new_states[i] = blk(x, states[i])
        y = x.permute(0, 2, 1)   # back to [B, C, T]
        return y, new_states
