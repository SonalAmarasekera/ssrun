# collate_latents.py
from __future__ import annotations
from typing import List, Dict
import torch

def collate_rwkv_latents(batch: List[Dict], pad_value: float = 0.0) -> Dict[str, torch.Tensor]:
    """
    Pads variable-length [T,C] latents into:
      z_mix : [B,T_max,C]
      z_s1  : [B,T_max,C] (if present)
      z_s2  : [B,T_max,C] (if present)
      mask  : [B,T_max] (True for valid)
    Also returns fps (int), sr (int) lists for reference.
    """
    B = len(batch)
    C = batch[0]["z_mix"].shape[1]
    T_max = max(x["z_mix"].shape[0] for x in batch)

    z_mix = torch.full((B, T_max, C), pad_value, dtype=torch.float32)
    mask  = torch.zeros((B, T_max), dtype=torch.bool)

    has_targets = "z_s1" in batch[0] and "z_s2" in batch[0]
    z_s1 = z_s2 = None
    if has_targets:
        z_s1 = torch.full((B, T_max, C), pad_value, dtype=torch.float32)
        z_s2 = torch.full((B, T_max, C), pad_value, dtype=torch.float32)

    utt_ids, fps_list, sr_list = [], [], []
    for i, ex in enumerate(batch):
        zi = ex["z_mix"]; Ti = zi.shape[0]
        z_mix[i, :Ti] = zi
        mask[i, :Ti]  = True
        if has_targets:
            z_s1[i, :Ti] = ex["z_s1"]
            z_s2[i, :Ti] = ex["z_s2"]
        utt_ids.append(ex["utt_id"])
        fps_list.append(int(ex["fps"]))
        sr_list.append(int(ex["sr"]))

    out = {
        "z_mix": z_mix,          # [B,T_max,C]
        "mask":  mask,           # [B,T_max]
        "utt_id": utt_ids,
        "fps": torch.tensor(fps_list, dtype=torch.int32),
        "sr":  torch.tensor(sr_list,  dtype=torch.int32),
    }
    if has_targets:
        out["z_s1"] = z_s1
        out["z_s2"] = z_s2
    return out
