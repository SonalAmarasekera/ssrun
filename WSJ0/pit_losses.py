
# pit_losses.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F

def _masked_mse(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    # compute numerically in fp32 for stability; keep on same device
    diff = (a - b).to(torch.float32)  # [B,T,C]
    if mask is not None:
        m = mask.unsqueeze(-1).to(diff.dtype)
        diff = diff * m
        denom = m.sum()
        return (diff.square().sum() / torch.clamp_min(denom, 1.0))
    else:
        return diff.square().mean()

def pit_latent_mse_2spk(preds: Dict[str, torch.Tensor],
                        tgt1: torch.Tensor,
                        tgt2: torch.Tensor,
                        pad_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[int,int], Dict[str, torch.Tensor]]:
    y1, y2 = preds["pred1"], preds["pred2"]
    L_12 = _masked_mse(y1, tgt1, pad_mask) + _masked_mse(y2, tgt2, pad_mask)
    L_21 = _masked_mse(y1, tgt2, pad_mask) + _masked_mse(y2, tgt1, pad_mask)
    if L_12 <= L_21:
        return L_12, (0,1), {"L_12": L_12.detach(), "L_21": L_21.detach()}
    else:
        return L_21, (1,0), {"L_12": L_12.detach(), "L_21": L_21.detach()}

def reorder_by_perm(preds: Dict[str, torch.Tensor], perm: Tuple[int,int]) -> Tuple[torch.Tensor, torch.Tensor]:
    y = [preds["pred1"], preds["pred2"]]
    return y[perm[0]], y[perm[1]]

def latent_mse_embedding_loss(y: torch.Tensor, z: torch.Tensor, pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
    # Conformer/CodecFormer-EL style latent regression, adapted to [B,T,C] with padding mask
    return _masked_mse(y, z, pad_mask)

def total_separator_loss(preds: Dict[str, torch.Tensor],
                         z_s1: torch.Tensor,
                         z_s2: torch.Tensor,
                         pad_mask: Optional[torch.Tensor],
                         lambda_residual_l2: float = 1e-4,
                         lambda_mask_entropy: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Tuple[int,int]]:
    pit_loss, perm, extras = pit_latent_mse_2spk(preds, z_s1, z_s2, pad_mask)

    reg_loss = 0.0
    for key in ("resid1", "resid2"):
        r = preds.get(key, None)
        if r is not None:
            if pad_mask is not None:
                m = pad_mask.unsqueeze(-1).to(r.dtype)
                reg_loss = reg_loss + (r.square() * m).sum() / torch.clamp_min(m.sum(), 1.0)
            else:
                reg_loss = reg_loss + r.square().mean()
    reg_loss = reg_loss * float(lambda_residual_l2)

    ent_loss = 0.0
    if lambda_mask_entropy > 0.0 and ("mask1" in preds and "mask2" in preds):
        for key in ("mask1", "mask2"):
            m = preds[key].clamp(1e-6, 1-1e-6)
            H = -(m*torch.log(m) + (1-m)*torch.log(1-m))
            if pad_mask is not None:
                pm = pad_mask.unsqueeze(-1).to(H.dtype)
                ent_loss = ent_loss + (H * pm).sum() / torch.clamp_min(pm.sum(), 1.0)
            else:
                ent_loss = ent_loss + H.mean()
        ent_loss = ent_loss * float(lambda_mask_entropy)

    total = pit_loss + reg_loss + ent_loss
    logs = {
        "loss/pit_latent": pit_loss.detach(),
        "loss/reg_residual": torch.tensor(reg_loss).detach(),
        "loss/mask_entropy": torch.tensor(ent_loss).detach(),
        "loss/total": total.detach(),
        **extras
    }
    return total, logs, perm
