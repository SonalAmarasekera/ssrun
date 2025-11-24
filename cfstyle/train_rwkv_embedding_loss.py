#!/usr/bin/env python3
"""
RWKV-v7 + DAC training script with Embedding Loss (CodecFormer-EL style).

Pipeline:
  1. Encode Mixture -> z_mix_q (Latents)
  2. RWKV Separator -> est_latents
  3. Encode Sources -> src_q (Ground Truth Latents)
  4. Loss = PIT(MSE(est_latents, src_q))

This avoids the expensive DAC Decoder step during training.

Usage:
python train_rwkv_embedding_loss.py --train_csv train.csv --valid_csv valid.csv --sample_rate 16000 --epochs 100
"""

import argparse
import csv
import math
import os
from typing import List, Dict, Tuple
from itertools import permutations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# --- RWKV v7 CUDA settings ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")

# IMPORT YOUR MODEL HERE
# Ensure rwkv_separator_grouped.py is in the same folder
from rwkv_separator_grouped import build_rwkv7_separator 
from codecformer3 import DACWrapper

# =========================
#   LOSS FUNCTIONS
# =========================

def pit_mse_embedding_loss(est_latents: torch.Tensor, true_latents: torch.Tensor) -> torch.Tensor:
    """
    Permutation Invariant Training (PIT) with MSE Loss on Latent Embeddings.
    Args:
        est_latents:  [Batch, S, C, T]
        true_latents: [Batch, S, C, T]
    """
    B, S, C, T = est_latents.shape
    assert true_latents.shape == est_latents.shape
    
    # Optimization for 2 speakers (hardcoded for speed)
    if S == 2:
        # est: [B, 2, C, T]
        e1, e2 = est_latents[:, 0], est_latents[:, 1]
        s1, s2 = true_latents[:, 0], true_latents[:, 1]
        
        # Permutation 1: 1->1, 2->2
        loss1 = F.mse_loss(e1, s1, reduction='none').mean(dim=(1,2)) + \
                F.mse_loss(e2, s2, reduction='none').mean(dim=(1,2))
        
        # Permutation 2: 1->2, 2->1
        loss2 = F.mse_loss(e1, s2, reduction='none').mean(dim=(1,2)) + \
                F.mse_loss(e2, s1, reduction='none').mean(dim=(1,2))
        
        # Min over permutations
        min_loss, _ = torch.min(torch.stack([loss1, loss2], dim=1), dim=1)
        return min_loss.mean()

    # Generic S > 2
    perms = list(permutations(range(S)))
    loss_perms = []
    for p in perms:
        est_p = est_latents[:, p, :, :]
        mse = F.mse_loss(est_p, true_latents, reduction='none').mean(dim=(1,2,3))
        loss_perms.append(mse)
    
    min_loss, _ = torch.min(torch.stack(loss_perms, dim=1), dim=1)
    return min_loss.mean()

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """SI-SDR for Validation monitoring only."""
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)
    dot = (est_zm * ref_zm).sum(dim=-1, keepdim=True)
    ref_energy = (ref_zm ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = dot / ref_energy * ref_zm
    e_noise = est_zm - s_target
    ratio = ((s_target ** 2).sum(dim=-1) + eps) / ((e_noise ** 2).sum(dim=-1) + eps)
    return 10 * torch.log10(ratio + eps)

def pit_si_sdr_loss(est, ref):
    """PIT wrapper for SI-SDR (Validation only)."""
    B, S, T = est.shape
    if S != 2: return torch.tensor(0.0) # Fallback
    
    sdr11 = si_sdr(est[:,0], ref[:,0])
    sdr22 = si_sdr(est[:,1], ref[:,1])
    perm1 = -(sdr11 + sdr22) / 2
    
    sdr12 = si_sdr(est[:,0], ref[:,1])
    sdr21 = si_sdr(est[:,1], ref[:,0])
    perm2 = -(sdr12 + sdr21) / 2
    
    return torch.min(perm1, perm2).mean()

# =========================
#   DATASET
# =========================

class Wsj02MixDataset(Dataset):
    def __init__(self, csv_path: str, sample_rate: int = 16000, segment_seconds: float = 3.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
        self.rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("mix_path"): self.rows.append(row)

    def __len__(self): return len(self.rows)

    def _load(self, path):
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1: data = data.T.mean(dim=0)
        wav = torch.from_numpy(data).unsqueeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav

    def __getitem__(self, idx):
        row = self.rows[idx]
        mix = self._load(row["mix_path"])
        s1  = self._load(row["s1_path"])
        s2  = self._load(row["s2_path"])
        
        T = min(mix.size(-1), s1.size(-1), s2.size(-1))
        mix, s1, s2 = mix[..., :T], s1[..., :T], s2[..., :T]
        
        seg = self.segment_samples
        if T > seg:
            start = torch.randint(0, T - seg + 1, (1,)).item()
            mix = mix[..., start:start+seg]
            s1  = s1[..., start:start+seg]
            s2  = s2[..., start:start+seg]
        elif T < seg:
            pad = seg - T
            mix = F.pad(mix, (0, pad))
            s1  = F.pad(s1, (0, pad))
            s2  = F.pad(s2, (0, pad))
            
        sources = torch.stack([s1, s2], dim=0) # [2, 1, T]
        return {"mix": mix, "sources": sources}

def collate_fn(batch):
    T_max = max([b["mix"].shape[-1] for b in batch])
    # Front padding for Warm-up (helps RWKV)
    pad_front = 1600 # 100ms at 16k
    
    mix_list, src_list = [], []
    for b in batch:
        mix, src = b["mix"], b["sources"]
        pad_back = T_max - mix.shape[-1]
        # Pad: (Front, Back)
        mix = F.pad(mix, (pad_front, pad_back))
        src = F.pad(src, (pad_front, pad_back)) 
        mix_list.append(mix)
        src_list.append(src)
        
    return torch.stack(mix_list), torch.stack(src_list)

# =========================
#   TRAINING
# =========================

def train_one_epoch(epoch, model, dac, loader, optimizer, device, grad_clip, writer):
    model.train()
    dac.model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Train Ep {epoch}", leave=False)
    
    for i, (mix, sources) in enumerate(pbar):
        mix, sources = mix.to(device), sources.to(device)
        optimizer.zero_grad()
        
        # 1. Encode Mixture -> Latents
        with torch.no_grad():
            mix_enc, _ = dac.get_encoded_features(mix)
            mix_q, _, _, _, _ = dac.get_quantized_features(mix_enc)
        
        # Input to model: [B, T, C]
        z_mix = mix_q.permute(0, 2, 1).contiguous()
        
        # 2. RWKV Separation -> [B, T, S, C]
        sep_lat = model(z_mix) 
        
        # Prepare for Loss: [B, S, C, T]
        est_latents = sep_lat.permute(0, 2, 3, 1).contiguous()
        
        # 3. Encode Sources -> Target Latents
        B, S, _, T_wav = sources.shape
        src_flat = sources.view(B*S, 1, T_wav)
        
        with torch.no_grad():
            src_enc, _ = dac.get_encoded_features(src_flat)
            # Use Quantized Latents as target (Bounded & Stable)
            src_q, _, _, _, _ = dac.get_quantized_features(src_enc)
            
        # Reshape to [B, S, C, T]
        target_latents = src_q.view(B, S, -1, src_q.shape[-1])
        
        # 4. Embedding Loss (PIT-MSE)
        # NOTE: We must account for padding if lengths differ, 
        # but collate makes them equal T_lat.
        loss = pit_mse_embedding_loss(est_latents, target_latents)
        
        loss.backward()
        if grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
        if writer and i % 10 == 0:
            step = (epoch-1)*len(loader) + i
            writer.add_scalar("loss/train_step", loss.item(), step)
            
    return total_loss / len(loader)

@torch.no_grad()
def validate(epoch, model, dac, loader, device):
    model.eval()
    dac.model.eval()
    mse_loss_accum = 0.0
    sdr_loss_accum = 0.0
    
    # We validate on a subset to save time if needed
    for mix, sources in tqdm(loader, desc="Validating", leave=False):
        mix, sources = mix.to(device), sources.to(device)
        
        # --- Latent Pass ---
        mix_enc, orig_len = dac.get_encoded_features(mix)
        mix_q, _, _, _, _ = dac.get_quantized_features(mix_enc)
        z_mix = mix_q.permute(0, 2, 1).contiguous()
        
        sep_lat = model(z_mix) # [B, T, S, C]
        
        # --- MSE Loss Check ---
        est_latents = sep_lat.permute(0, 2, 3, 1)
        B, S, _, T_wav = sources.shape
        src_enc, _ = dac.get_encoded_features(sources.view(B*S, 1, T_wav))
        src_q, _, _, _, _ = dac.get_quantized_features(src_enc)
        target_latents = src_q.view(B, S, -1, src_q.shape[-1])
        
        mse = pit_mse_embedding_loss(est_latents, target_latents)
        mse_loss_accum += mse.item()
        
        # --- SI-SDR Check (Decode for monitoring only) ---
        # Decode estimates
        # sep_lat is [B, T, S, C]. Need [B, C, T] per source.
        est_wavs = []
        for s in range(S):
            z_s = sep_lat[:, :, s, :].permute(0, 2, 1) # [B, C, T]
            wav = dac.get_decoded_signal(z_s, orig_len)
            est_wavs.append(wav)
        est_wavs = torch.stack(est_wavs, dim=1).squeeze(2) # [B, S, T]
        
        # Decode targets (Oracle)
        src_wav = dac.get_decoded_signal(src_q, None).view(B, S, -1)
        # Trim padding
        min_len = min(est_wavs.shape[-1], src_wav.shape[-1])
        sdr = pit_si_sdr_loss(est_wavs[..., :min_len], src_wav[..., :min_len])
        sdr_loss_accum += sdr.item() # This is negative SI-SDR
        
    return mse_loss_accum / len(loader), sdr_loss_accum / len(loader)

# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--valid_csv", type=str, required=True)
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n_layer", type=int, default=4)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--resume", type=str, default=None)
    
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(os.path.join(args.save_dir, "tb"))
    
    # Load DAC (Frozen)
    dac = DACWrapper(DAC_sample_rate=16000, Freeze=True)
    dac.model.to(device)
    
    # Determine Latent Dim
    dummy = torch.randn(1, 1, 16000).to(device)
    with torch.no_grad():
        z, _ = dac.get_encoded_features(dummy)
    C_lat = z.shape[1]
    print(f"[INFO] DAC Latent Dim: {C_lat}")
    
    # Build Model (Grouped Bi-RWKV)
    model = build_rwkv7_separator(
        n_embd=C_lat,
        n_layer=args.n_layer,
        head_mode="mask", # Important for stability
        enforce_bf16=True
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Resume
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optim'])
        start_epoch = ckpt['epoch'] + 1
        print(f"[INFO] Resumed from ep {start_epoch-1}")

    # Load Data
    train_loader = DataLoader(
        Wsj02MixDataset(args.train_csv, args.sample_rate),
        batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    valid_loader = DataLoader(
        Wsj02MixDataset(args.valid_csv, args.sample_rate),
        batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    # Loop
    best_val = float('inf')
    for ep in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(ep, model, dac, train_loader, optimizer, device, 5.0, writer)
        val_mse, val_sdr = validate(ep, model, dac, valid_loader, device)
        
        print(f"[Ep {ep}] Train MSE: {train_loss:.4f} | Val MSE: {val_mse:.4f} | Val SI-SDR: {-val_sdr:.2f} dB")
        
        writer.add_scalar("loss/val_mse", val_mse, ep)
        writer.add_scalar("loss/val_sdr", -val_sdr, ep)
        
        if val_mse < best_val:
            best_val = val_mse
            torch.save({
                'epoch': ep,
                'model': model.state_dict(),
                'optim': optimizer.state_dict(),
                'config': vars(args)
            }, os.path.join(args.save_dir, "best_model.pt"))

if __name__ == "__main__":
    main()