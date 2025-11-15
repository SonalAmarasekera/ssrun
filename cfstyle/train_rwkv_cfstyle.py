#!/usr/bin/env python3
"""
RWKV-v7 + DAC training script (CodecFormer-style) on WSJ0-2mix.

Pipeline:
  waveforms (mix, s1, s2)
    → DAC encoder (frozen)
    → quantized latents z_mix_q [B, C, T_lat]
    → RWKVv7Separator(z_mix_q^T)  # [B, T_lat, C] → [B, T_lat, S, C]
    → DAC decoder per source
    → waveforms hat_s1, hat_s2
    → PIT SI-SDR loss in waveform domain

Dependencies:
  - torch
  - torchaudio
  - numpy
  - soundfile (if you prefer it over torchaudio.load)
  - codecformer3.DACWrapper   (from your original CodecFormer code)
  - rwkv_separator_Claudemod.build_rwkv7_separator  :contentReference[oaicite:1]{index=1}
"""

import argparse
import csv
import math
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

from codecformer.codecformer3 import DACWrapper   # make sure this is importable
from rwkv_separator_Claudemod import build_rwkv7_separator  # your RWKV v7 separator


# =========================
#   DATASET
# =========================

class Wsj02MixDataset(Dataset):
    """
    Simple WSJ0-2mix dataset loader using a CSV file.

    CSV format (header row required):
        mix_path,s1_path,s2_path
    Each row points to three mono WAV files with the same sample rate.
    """

    def __init__(self, csv_path: str, sample_rate: int = 8000):
        super().__init__()
        self.sample_rate = sample_rate
        self.rows: List[Dict[str, str]] = []

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("mix_path") or not row.get("s1_path") or not row.get("s2_path"):
                    continue
                self.rows.append(row)

        if not self.rows:
            raise RuntimeError(f"No valid rows found in CSV: {csv_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load_mono(self, path: str) -> torch.Tensor:
        """Load audio as [1, T] mono waveform at expected sample rate."""
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # mixdown to mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        return wav  # [1, T]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        mix = self._load_mono(row["mix_path"])   # [1, Tm]
        s1  = self._load_mono(row["s1_path"])    # [1, T1]
        s2  = self._load_mono(row["s2_path"])    # [1, T2]

        # Ensure same length (crop to min length)
        T = min(mix.size(-1), s1.size(-1), s2.size(-1))
        mix = mix[..., :T]
        s1  = s1[..., :T]
        s2  = s2[..., :T]

        # Stack sources: [S, 1, T]
        sources = torch.stack([s1, s2], dim=0)

        return {
            "mix": mix,         # [1, T]
            "sources": sources  # [2, 1, T]
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad all examples in the batch to the max time length.

    Returns:
      mix:     [B, 1, T_max]
      sources: [B, S, 1, T_max]
    """
    # batch[i]["mix"] : [1, T_i]
    # batch[i]["sources"] : [S, 1, T_i]

    # 1) find max length in this batch
    lengths = [b["mix"].shape[-1] for b in batch]
    T_max = max(lengths)

    mix_list = []
    sources_list = []

    for b in batch:
        mix = b["mix"]         # [1, T]
        sources = b["sources"] # [S, 1, T]
        T = mix.shape[-1]
        pad_T = T_max - T

        if pad_T > 0:
            # pad last dimension (time) on the right
            mix = F.pad(mix, (0, pad_T))                 # [1, T_max]
            sources = F.pad(sources, (0, pad_T))         # [S, 1, T_max]

        mix_list.append(mix)
        sources_list.append(sources)

    mix = torch.stack(mix_list, dim=0)          # [B, 1, T_max]
    sources = torch.stack(sources_list, dim=0)  # [B, S, 1, T_max]

    return mix, sources

# =========================
#   SI-SDR + PIT
# =========================

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-invariant SDR (SI-SDR) in dB.

    est, ref: [B, T]
    Returns: [B] (per-example SI-SDR)
    """
    # zero-mean
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)

    # projection of est onto ref
    dot = (est_zm * ref_zm).sum(dim=-1, keepdim=True)
    ref_energy = (ref_zm ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = dot / ref_energy * ref_zm

    e_noise = est_zm - s_target

    s_target_energy = (s_target ** 2).sum(dim=-1) + eps
    e_noise_energy = (e_noise ** 2).sum(dim=-1) + eps

    ratio = s_target_energy / e_noise_energy
    return 10 * torch.log10(ratio + eps)


def pit_si_sdr_loss(est_sources: torch.Tensor,
                    true_sources: torch.Tensor) -> torch.Tensor:
    """
    2-speaker PIT SI-SDR loss.

    est_sources:  [B, S, T]
    true_sources: [B, S, T]
    S must be 2 here.
    Returns a scalar loss (mean over batch).
    """
    assert est_sources.ndim == 3 and true_sources.ndim == 3
    B, S, T = est_sources.shape
    assert S == 2, "This simple PIT implementation assumes 2 speakers."

    est1 = est_sources[:, 0, :]  # [B, T]
    est2 = est_sources[:, 1, :]
    s1   = true_sources[:, 0, :]
    s2   = true_sources[:, 1, :]

    # permutation 1: (est1→s1, est2→s2)
    sdr11 = si_sdr(est1, s1)  # [B]
    sdr22 = si_sdr(est2, s2)  # [B]
    loss_perm1 = -0.5 * (sdr11 + sdr22)  # [B]

    # permutation 2: (est1→s2, est2→s1)
    sdr12 = si_sdr(est1, s2)
    sdr21 = si_sdr(est2, s1)
    loss_perm2 = -0.5 * (sdr12 + sdr21)  # [B]

    # take the better (lower loss ⇒ higher SI-SDR)
    loss = torch.minimum(loss_perm1, loss_perm2)
    return loss.mean()


# =========================
#   TRAINING LOOP
# =========================

def train_one_epoch(
    epoch: int,
    model: nn.Module,
    dac: DACWrapper,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
) -> float:
    model.train()
    dac.model.eval()  # frozen encoder/decoder
    total_loss = 0.0
    num_batches = 0

    for mix, sources in loader:
        mix = mix.to(device)             # [B, 1, T]
        sources = sources.to(device)     # [B, S, 1, T]
        B, S, _, T = sources.shape
        assert S == 2, "This script assumes 2 speakers."

        optimizer.zero_grad()

        # ---------- DAC encode mixture ----------
        # DACWrapper expects [B, 1, T]
        with torch.no_grad():
            mix_enc, orig_len = dac.get_encoded_features(mix)  # [B, C_lat, T_lat], int
            # Quantize (as in CodecFormer); decoder accepts continuous too
            mix_q, _, _, _, _ = dac.get_quantized_features(mix_enc)  # [B, C_lat, T_lat]

        # Convert to [B, T_lat, C_lat] for RWKV
        z_mix = mix_q.permute(0, 2, 1).contiguous()  # [B, T_lat, C_lat]

        # ---------- RWKV-v7 separation in latent domain ----------
        # Your model: [B,T,C] -> [B,T,S,C]
        sep_lat = model(z_mix)                       # [B, T_lat, S, C_lat]

        # Reformat per speaker for decoder: [B, C_lat, T_lat]
        sep_lat = sep_lat.permute(0, 2, 3, 1).contiguous()  # [B, S, C, T_lat]

        # ---------- DAC decode each speaker ----------
        est_wavs = []
        for s_idx in range(S):
            z_s = sep_lat[:, s_idx, :, :]  # [B, C, T_lat]
            # DACWrapper.get_decoded_signal expects [B, C, T_lat]
            # Use original mixture length for all examples (they are cropped to same T)
            wav_hat = dac.get_decoded_signal(z_s, orig_len)  # [B, 1, T']
            # Make sure T' matches mix length; DACWrapper already trims/pads to original_length
            est_wavs.append(wav_hat)  # list of [B,1,T]

        # Stack to [B, S, T]
        est_sources = torch.stack(est_wavs, dim=1).squeeze(2)  # [B, S, T]
        true_sources = sources.squeeze(2)                      # [B, S, T]

        # ---------- PIT SI-SDR loss in waveform space ----------
        loss = pit_si_sdr_loss(est_sources, true_sources)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def validate(
    epoch: int,
    model: nn.Module,
    dac: DACWrapper,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    dac.model.eval()
    total_loss = 0.0
    num_batches = 0

    for mix, sources in loader:
        mix = mix.to(device)         # [B, 1, T]
        sources = sources.to(device) # [B, S, 1, T]
        B, S, _, T = sources.shape
        assert S == 2

        # DAC encode
        mix_enc, orig_len = dac.get_encoded_features(mix)   # [B, C, T_lat]
        mix_q, _, _, _, _ = dac.get_quantized_features(mix_enc)
        z_mix = mix_q.permute(0, 2, 1).contiguous()         # [B, T_lat, C]

        # RWKV-v7 separation
        sep_lat = model(z_mix)                              # [B, T_lat, S, C]
        sep_lat = sep_lat.permute(0, 2, 3, 1).contiguous()  # [B, S, C, T_lat]

        # DAC decode each speaker
        est_wavs = []
        for s_idx in range(S):
            z_s = sep_lat[:, s_idx, :, :]                   # [B, C, T_lat]
            wav_hat = dac.get_decoded_signal(z_s, orig_len) # [B,1,T]
            est_wavs.append(wav_hat)

        est_sources = torch.stack(est_wavs, dim=1).squeeze(2)  # [B,S,T]
        true_sources = sources.squeeze(2)                      # [B,S,T]

        loss = pit_si_sdr_loss(est_sources, true_sources)
        total_loss += float(loss.item())
        num_batches += 1

    return total_loss / max(1, num_batches)


# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True,
                    help="CSV with mix_path,s1_path,s2_path for training set.")
    ap.add_argument("--valid_csv", type=str, required=True,
                    help="CSV with mix_path,s1_path,s2_path for validation set.")
    ap.add_argument("--sample_rate", type=int, default=8000,
                    help="Input sample rate (will be resampled to DAC SR internally).")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--dac_model_type", type=str, default="16khz",
                    help="DAC model type for DACWrapper (typically 16kHz).")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="./checkpoints_rwkv_dac")
    ap.add_argument("--n_layer", type=int, default=4,
                    help="Number of RWKV-v7 layers.")
    ap.add_argument("--head_mode", type=str, default="residual",
                    choices=["residual", "mask"],
                    help="RWKV separation head mode.")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # Datasets and loaders
    train_ds = Wsj02MixDataset(args.train_csv, sample_rate=args.sample_rate)
    valid_ds = Wsj02MixDataset(args.valid_csv, sample_rate=args.sample_rate)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # DAC wrapper (pretrained Descript Audio Codec)
    # This uses the internal DAC sample rate (e.g. 16 kHz) and resamples from your input SR.
    dac = DACWrapper(input_sample_rate=args.sample_rate, DAC_model_path=None, DAC_sample_rate=16000, Freeze=True)
    dac.model.to(device)
    dac.dac_sampler.to(device)
    dac.org_sampler.to(device)

    # Infer latent dimension C by encoding one batch
    mix_example, _ = next(iter(train_loader))
    mix_example = mix_example.to(device)           # [B,1,T]
    with torch.no_grad():
        z_enc, _ = dac.get_encoded_features(mix_example)  # [B,C,T_lat]
    _, C_lat, _ = z_enc.shape
    print(f"[INFO] DAC latent channels: {C_lat}")

    # RWKV-v7 separator
    model = build_rwkv7_separator(
        n_embd=C_lat,
        n_layer=args.n_layer,
        num_sources=2,
        head_mode=args.head_mode,
        enforce_bf16=True,  # you can toggle this off for debugging
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] RWKV-v7 separator parameters: total={num_params:,}, trainable={num_trainable:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            epoch, model, dac, train_loader, optimizer, device, grad_clip=args.grad_clip
        )
        val_loss = validate(
            epoch, model, dac, valid_loader, device
        )

        print(f"[EPOCH {epoch:03d}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f} (PIT SI-SDR)")

        # Remember: lower loss = better (since it's -SI-SDR)
        if val_loss < best_val:
            best_val = val_loss
            ckpt_path = os.path.join(args.save_dir, f"best_epoch{epoch:03d}_loss{val_loss:.4f}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": vars(args),
                    "latent_channels": C_lat,
                },
                ckpt_path,
            )
            print(f"  ✅ Saved new best checkpoint: {ckpt_path}")

    print(f"[DONE] Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
