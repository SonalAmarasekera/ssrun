#!/usr/bin/env python3
"""
Time-domain RWKV-v7 separator training script on WSJ0-2mix.

Pipeline:
  waveforms (mix, s1, s2)
    → TimeDomainRWKVSeparator (RWKV core in feature space)
    → estimated waveforms hat_s1, hat_s2
    → PIT SI-SDR loss in waveform domain vs clean sources

Usage example:
  python train_rwkv_time.py \
    --train_csv train_min.csv \
    --valid_csv dev_min.csv \
    --sample_rate 16000 \
    --epochs 50 \
    --device "cuda" \
    --n_layer 8 \
    --n_embd 256 \
    --head_mode "mask" \
    --lr_scheduler \
    --early_stop \
    --batch_size 16
"""

import argparse
import csv
import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio

import soundfile as sf

from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from tqdm.auto import tqdm

# --- RWKV v7 CUDA settings (must be set before importing RWKV model) ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
# -----------------------------------------------------------------------

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

    def __init__(self, csv_path: str, sample_rate: int = 16000, segment_seconds: float = 3.0):
        super().__init__()
        self.sample_rate = sample_rate
        self.segment_samples = int(segment_seconds * sample_rate)
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
        data, sr = sf.read(path, dtype="float32")  # data: [T] or [T, C]
        if data.ndim == 1:
            wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
        else:
            wav = torch.from_numpy(data.T)  # [C, T]
            wav = wav.mean(dim=0, keepdim=True)  # [1, T]

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        return wav  # [1, T]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        mix = self._load_mono(row["mix_path"])   # [1, Tm]
        s1  = self._load_mono(row["s1_path"])    # [1, T1]
        s2  = self._load_mono(row["s2_path"])    # [1, T2]

        # 1) align lengths
        T = min(mix.size(-1), s1.size(-1), s2.size(-1))
        mix = mix[..., :T]
        s1  = s1[..., :T]
        s2  = s2[..., :T]

        # 2) random crop or pad to fixed segment length
        seg = self.segment_samples
        if T > seg:
            start = torch.randint(0, T - seg + 1, (1,)).item()
            end = start + seg
            mix = mix[..., start:end]
            s1  = s1[..., start:end]
            s2  = s2[..., start:end]
            T = seg
        elif T < seg:
            pad = seg - T
            mix = F.pad(mix, (0, pad))
            s1  = F.pad(s1,  (0, pad))
            s2  = F.pad(s2,  (0, pad))
            T = seg

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
            mix = F.pad(mix, (0, pad_T))         # [1, T_max]
            sources = F.pad(sources, (0, pad_T)) # [S, 1, T_max]

        mix_list.append(mix)
        sources_list.append(sources)

    mix = torch.stack(mix_list, dim=0)          # [B, 1, T_max]
    sources = torch.stack(sources_list, dim=0)  # [B, S, 1, T_max]

    return mix, sources


# =========================
#   TIME-DOMAIN RWKV WRAPPER
# =========================

class TimeDomainRWKVSeparator(nn.Module):
    """
    Wraps RWKV-v7 separator to operate directly on time-domain waveforms.

    Input:
      mix: [B, 1, T]

    Output:
      est_sources: [B, S, T]
    """

    def __init__(self, n_embd: int, n_layer: int, num_sources: int = 2, head_mode: str = "mask"):
        super().__init__()
        self.n_embd = n_embd
        self.num_sources = num_sources

        # Project scalar sample → embedding
        self.in_proj = nn.Linear(1, n_embd)

        # RWKV core: expects [B, T, C] and outputs [B, T, S, C]
        self.rwkv = build_rwkv7_separator(
            n_embd=n_embd,
            n_layer=n_layer,
            num_sources=num_sources,
            head_mode=head_mode,
            enforce_bf16=False,
        )

        # Project embedding back to scalar sample
        self.out_proj = nn.Linear(n_embd, 1)

    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        """
        mix: [B, 1, T]
        returns est_sources: [B, S, T]
        """
        B, C, T = mix.shape
        assert C == 1, "Expected mono input [B,1,T]"

        # [B,1,T] -> [B,T,1]
        x = mix.transpose(1, 2)  # [B, T, 1]

        # Input projection
        x = self.in_proj(x)      # [B, T, n_embd]

        # RWKV separation: [B, T, n_embd] -> [B, T, S, n_embd]
        sep_lat = self.rwkv(x)   # [B, T, S, n_embd]

        # Output projection per source
        # [B, T, S, n_embd] -> [B, T, S, 1]
        out = self.out_proj(sep_lat)  # broadcast over last dim

        # [B, T, S, 1] -> [B, S, T]
        est_sources = out.squeeze(-1).permute(0, 2, 1).contiguous()  # [B,S,T]
        return est_sources


# =========================
#   SI-SDR + PIT
# =========================

def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-invariant SDR (SI-SDR) in dB.

    est, ref: [B, T]
    Returns: [B] (per-example SI-SDR)
    """
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)

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
    sdr11 = si_sdr(est1, s1)
    sdr22 = si_sdr(est2, s2)
    loss_perm1 = -(sdr11 + sdr22)  # [B]

    # permutation 2: (est1→s2, est2→s1)
    sdr12 = si_sdr(est1, s2)
    sdr21 = si_sdr(est2, s1)
    loss_perm2 = -(sdr12 + sdr21)  # [B]

    loss = torch.minimum(loss_perm1, loss_perm2)
    return loss.mean()


# =========================
#   TRAINING LOOP
# =========================

def train_one_epoch(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 5.0,
    writer: SummaryWriter = None,
) -> float:
    model.train()

    total_loss = 0.0
    num_batches_done = 0

    num_batches = len(loader)
    log_interval = max(1, num_batches // 4)  # 4 logs per epoch
    pbar = tqdm(loader, desc=f"Train epoch {epoch:03d}", leave=False)

    for batch_idx, (mix, sources) in enumerate(pbar, start=1):
        mix = mix.to(device)         # [B, 1, T]
        sources = sources.to(device) # [B, S, 1, T]
        B, S, _, T = sources.shape
        assert S == 2, "This script assumes 2 speakers."

        optimizer.zero_grad()

        # True sources [B, S, T]
        true_sources = sources.squeeze(2)  # [B, S, T]

        # Time-domain RWKV separation
        est_sources = model(mix)  # [B, S, T]

        loss = pit_si_sdr_loss(est_sources, true_sources)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_loss = float(loss.item())
        total_loss += batch_loss
        num_batches_done += 1
        avg_loss = total_loss / num_batches_done

        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=avg_loss, lr=current_lr)

        if writer is not None and (batch_idx % log_interval == 0 or batch_idx == num_batches):
            global_step = (epoch - 1) * num_batches + batch_idx
            writer.add_scalar("loss/train_step", batch_loss, global_step)
            writer.add_scalar("loss/train_step_avg", avg_loss, global_step)
            writer.add_scalar("lr_step", current_lr, global_step)

    return total_loss / max(1, num_batches_done)


@torch.no_grad()
def validate(
    epoch: int,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    num_batches_done = 0

    num_batches = len(loader)
    pbar = tqdm(loader, desc=f"Val   epoch {epoch:03d}", leave=False)

    for mix, sources in pbar:
        mix = mix.to(device)         # [B, 1, T]
        sources = sources.to(device) # [B, S, 1, T]
        B, S, _, T = sources.shape
        assert S == 2

        true_sources = sources.squeeze(2)  # [B, S, T]
        est_sources = model(mix)           # [B, S, T]

        loss = pit_si_sdr_loss(est_sources, true_sources)

        batch_loss = float(loss.item())
        total_loss += batch_loss
        num_batches_done += 1
        avg_loss = total_loss / num_batches_done
        pbar.set_postfix(val_loss=avg_loss)

    return total_loss / max(1, num_batches_done)


# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", type=str, required=True,
                    help="CSV with mix_path,s1_path,s2_path for training set.")
    ap.add_argument("--valid_csv", type=str, required=True,
                    help="CSV with mix_path,s1_path,s2_path for validation set.")
    ap.add_argument("--sample_rate", type=int, default=16000,
                    help="Input sample rate.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    # Kept for interface compatibility; unused in time-domain mode:
    ap.add_argument("--dac_model_type", type=str, default="16khz",
                    help="(Unused here) Kept for compatibility.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_dir", type=str, default="/workspace/checkpoints_rwkv_time")
    ap.add_argument("--n_layer", type=int, default=4,
                    help="Number of RWKV-v7 layers.")
    ap.add_argument("--head_mode", type=str, default="mask",
                    choices=["residual", "mask"],
                    help="RWKV separation head mode.")
    ap.add_argument("--n_embd", type=int, default=256,
                    help="Time-domain embedding dimension for RWKV.")

    # TensorBoard
    ap.add_argument("--log_dir", type=str, default="/workspace/checkpoints_rwkv_time/tb",
                    help="TensorBoard log directory")
    ap.add_argument("--log_hparams", action="store_true",
                    help="Log hyperparameters once at the beginning.")

    # LR scheduler
    ap.add_argument("--lr_scheduler", action="store_true",
                    help="Enable ReduceLROnPlateau LR scheduler on val_loss.")
    ap.add_argument("--lr_scheduler_start_epoch", type=int, default=1,
                    help="Epoch to start applying LR scheduler (inclusive).")
    ap.add_argument("--lr_scheduler_patience", type=int, default=5,
                    help="Scheduler patience (epochs without val improvement before LR is reduced).")
    ap.add_argument("--lr_scheduler_factor", type=float, default=0.5,
                    help="Factor to multiply LR by when scheduler triggers.")
    ap.add_argument("--lr_scheduler_min_lr", type=float, default=1e-6,
                    help="Minimum learning rate for scheduler.")

    # Early stopping
    ap.add_argument("--early_stop", action="store_true",
                    help="Enable early stopping based on val_loss.")
    ap.add_argument("--early_stop_patience", type=int, default=20,
                    help="Number of epochs with no val improvement before early stopping.")

    # Resume
    ap.add_argument("--resume_checkpoint", type=str, default=None,
                    help="Optional path to a previous checkpoint")

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    tb_log_dir = args.log_dir or os.path.join(args.save_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Datasets and loaders
    train_ds = Wsj02MixDataset(args.train_csv, sample_rate=args.sample_rate, segment_seconds=3.0)
    valid_ds = Wsj02MixDataset(args.valid_csv, sample_rate=args.sample_rate, segment_seconds=3.0)

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

    # RWKV time-domain separator
    C_lat = args.n_embd  # for naming consistency
    model = TimeDomainRWKVSeparator(
        n_embd=C_lat,
        n_layer=args.n_layer,
        num_sources=2,
        head_mode=args.head_mode,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Time-domain RWKV separator parameters: total={num_params:,}, trainable={num_trainable:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Optional LR scheduler
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_scheduler_factor,
            patience=args.lr_scheduler_patience,
            min_lr=args.lr_scheduler_min_lr,
        )
        print(
            f"[INFO] LR scheduler enabled: "
            f"start_epoch={args.lr_scheduler_start_epoch}, "
            f"patience={args.lr_scheduler_patience}, "
            f"factor={args.lr_scheduler_factor}, "
            f"min_lr={args.lr_scheduler_min_lr}"
        )
    else:
        scheduler = None

    # Resume from checkpoint (if provided)
    start_epoch = 1
    best_val = float("inf")

    if args.resume_checkpoint is not None:
        if not os.path.isfile(args.resume_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_checkpoint}")

        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        ckpt_epoch = ckpt.get("epoch", 0)
        ckpt_val_loss = ckpt.get("val_loss", None)
        if isinstance(ckpt_val_loss, (float, int)):
            best_val = float(ckpt_val_loss)

        start_epoch = ckpt_epoch + 1
        print(
            f"[INFO] Resuming from checkpoint: {args.resume_checkpoint} "
            f"(epoch {ckpt_epoch}, best_val={ckpt_val_loss})"
        )

    # Log hyperparameters once
    if args.log_hparams:
        hparams = {
            "sample_rate": args.sample_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "grad_clip": args.grad_clip,
            "n_layer": args.n_layer,
            "head_mode": args.head_mode,
            "n_embd": args.n_embd,
            "lr_scheduler": args.lr_scheduler,
            "lr_scheduler_start_epoch": args.lr_scheduler_start_epoch,
            "lr_scheduler_patience": args.lr_scheduler_patience,
            "lr_scheduler_factor": args.lr_scheduler_factor,
            "lr_scheduler_min_lr": args.lr_scheduler_min_lr,
            "early_stop": args.early_stop,
            "early_stop_patience": args.early_stop_patience,
            "resume_checkpoint": args.resume_checkpoint,
        }
        writer.add_hparams(hparams, {"hparam/placeholder": 0.0})

    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=args.grad_clip,
            writer=writer,
        )

        val_loss = validate(
            epoch, model, valid_loader, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[EPOCH {epoch:03d}] train_loss={train_loss:.4f} "
            f" val_loss={val_loss:.4f} (PIT SI-SDR)  lr={current_lr:.3e}"
        )

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", current_lr, epoch)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            epochs_no_improve = 0
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
        else:
            epochs_no_improve += 1

        if scheduler is not None and epoch >= args.lr_scheduler_start_epoch:
            scheduler.step(val_loss)

        if args.early_stop and epochs_no_improve >= args.early_stop_patience:
            print(
                f"[EARLY STOP] No validation improvement for "
                f"{epochs_no_improve} epochs. Stopping at epoch {epoch}."
            )
            break

    print(f"[DONE] Best validation loss: {best_val:.4f}")
    writer.close()


if __name__ == "__main__":
    main()
