#!/usr/bin/env python3
"""
RWKV-v7 + DAC inference script (CodecFormer-style) on WSJ0-2mix.

- Loads a trained RWKV separator checkpoint (from train_rwkv_cfstyle_withLRsched.py).
- Reads a CSV with columns: mix_path,s1_path,s2_path
- Runs separation on each mixture.
- Computes:
    * PIT SI-SDR vs clean sources (classic SI-SDR)
    * PIT Codec SI-SDR vs codec-distorted sources
    * Optional perceptual metrics:
        - PESQ (full-reference quality)
        - STOI (full-reference intelligibility)
        - DNSMOS (non-intrusive MOS prediction)
- Optionally saves separated audio to disk.

Usage example:

    python infer_rwkv_cfstyle.py \
        --csv /path/to/test.csv \
        --checkpoint /workspace/checkpoints_rwkv_dac/best_epoch050_loss-2.2799.pt \
        --sample_rate 8000 \
        --batch_size 2 \
        --enable_pesq \
        --enable_stoi \
        --enable_dnsmos \
        --save_audio_dir /workspace/separated_wavs
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
import numpy as np

# --- Optional metric imports (graceful fallbacks) ---
try:
    from pesq import pesq as pesq_fn
    _HAS_PESQ = True
except ImportError:
    pesq_fn = None
    _HAS_PESQ = False

try:
    from pystoi import stoi as stoi_fn
    _HAS_STOI = True
except ImportError:
    stoi_fn = None
    _HAS_STOI = False

try:
    from torchmetrics.functional.audio.dnsmos import (
        deep_noise_suppression_mean_opinion_score as dnsmos_fn,
    )
    _HAS_DNSMOS = True
except ImportError:
    dnsmos_fn = None
    _HAS_DNSMOS = False

# --- RWKV v7 CUDA settings (same as training) ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
# ------------------------------------------------

from rwkv_separator_Claudemod import build_rwkv7_separator
from codecformer3 import DACWrapper


# =========================
#   DATASET (EVAL)
# =========================

class Wsj02MixEvalDataset(Dataset):
    """
    WSJ0-2mix dataset loader for evaluation.

    CSV format (header row required):
        mix_path,s1_path,s2_path

    Differences from training dataset:
      - NO random cropping.
      - We load each utterance to full length; collate_fn will pad within a batch.
    """

    def __init__(self, csv_path: str, sample_rate: int = 16000):
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
        data, sr = sf.read(path, dtype="float32")  # data: [T] or [T, C]
        if data.ndim == 1:
            wav = torch.from_numpy(data).unsqueeze(0)  # [1, T]
        else:
            wav = torch.from_numpy(data.T)             # [C, T]
            wav = wav.mean(dim=0, keepdim=True)        # [1, T]

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        return wav  # [1, T]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        mix = self._load_mono(row["mix_path"])   # [1, Tm]
        s1  = self._load_mono(row["s1_path"])    # [1, T1]
        s2  = self._load_mono(row["s2_path"])    # [1, T2]

        # Align lengths by truncating to min length (like training)
        T = min(mix.size(-1), s1.size(-1), s2.size(-1))
        mix = mix[..., :T]
        s1  = s1[..., :T]
        s2  = s2[..., :T]

        sources = torch.stack([s1, s2], dim=0)  # [2, 1, T]

        return {
            "mix": mix,         # [1, T]
            "sources": sources, # [2, 1, T]
            "row": row,         # keep paths for naming output
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, str]]]:
    """
    Pad all examples in the batch to the max time length.

    Returns:
      mix:     [B, 1, T_max]
      sources: [B, S, 1, T_max]
      rows:    list of dicts (CSV rows) for file naming
    """
    lengths = [b["mix"].shape[-1] for b in batch]
    T_max = max(lengths)

    mix_list = []
    sources_list = []
    rows = []

    for b in batch:
        mix = b["mix"]         # [1, T]
        sources = b["sources"] # [S, 1, T]
        T = mix.shape[-1]
        pad_T = T_max - T

        if pad_T > 0:
            mix = F.pad(mix, (0, pad_T))                 # [1, T_max]
            sources = F.pad(sources, (0, pad_T))         # [S, 1, T_max]

        mix_list.append(mix)
        sources_list.append(sources)
        rows.append(b["row"])

    mix = torch.stack(mix_list, dim=0)          # [B, 1, T_max]
    sources = torch.stack(sources_list, dim=0)  # [B, S, 1, T_max]

    return mix, sources, rows


# =========================
#   SI-SDR + PIT (metrics)
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
    e_noise_energy  = (e_noise  ** 2).sum(dim=-1) + eps

    ratio = s_target_energy / e_noise_energy
    return 10 * torch.log10(ratio + eps)


def pit_si_sdr_per_utt(est_sources: torch.Tensor,
                       true_sources: torch.Tensor) -> torch.Tensor:
    """
    2-speaker PIT SI-SDR metric (per utterance).

    est_sources:  [B, S, T]
    true_sources: [B, S, T]
    Returns: [B] SI-SDR per utterance (avg over the 2 speakers, best permutation).
    """
    assert est_sources.ndim == 3 and true_sources.ndim == 3
    B, S, T = est_sources.shape
    assert S == 2, "This simple PIT implementation assumes 2 speakers."

    est1 = est_sources[:, 0, :]  # [B, T]
    est2 = est_sources[:, 1, :]
    s1   = true_sources[:, 0, :]
    s2   = true_sources[:, 1, :]

    # Permutation 1: (est1→s1, est2→s2)
    sdr11 = si_sdr(est1, s1)  # [B]
    sdr22 = si_sdr(est2, s2)  # [B]
    sum_sdr_perm1 = sdr11 + sdr22  # [B]

    # Permutation 2: (est1→s2, est2→s1)
    sdr12 = si_sdr(est1, s2)
    sdr21 = si_sdr(est2, s1)
    sum_sdr_perm2 = sdr12 + sdr21  # [B]

    best_sum = torch.maximum(sum_sdr_perm1, sum_sdr_perm2)  # [B]
    # average over speakers
    return best_sum / 2.0  # [B]


# =========================
#   EVAL LOOP
# =========================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dac: DACWrapper,
    loader: DataLoader,
    device: torch.device,
    input_sample_rate: int,
    save_audio_dir: str = None,
    enable_pesq: bool = False,
    enable_stoi: bool = False,
    enable_dnsmos: bool = False,
) -> None:
    model.eval()
    dac.model.eval()

    all_si_sdr_clean = []
    all_si_sdr_codec = []

    all_pesq = []
    all_stoi = []
    all_dnsmos = []

    if save_audio_dir is not None:
        os.makedirs(save_audio_dir, exist_ok=True)

    # Warn once if flags are set but packages are missing
    if enable_pesq and not _HAS_PESQ:
        print("[WARN] --enable_pesq set but 'pesq' package not found. PESQ will be skipped.")
        enable_pesq = False
    if enable_stoi and not _HAS_STOI:
        print("[WARN] --enable_stoi set but 'pystoi' package not found. STOI will be skipped.")
        enable_stoi = False
    if enable_dnsmos and not _HAS_DNSMOS:
        print("[WARN] --enable_dnsmos set but 'torchmetrics[audio]' (DNSMOS) not available. DNSMOS will be skipped.")
        enable_dnsmos = False

    pesq_mode = "nb" if input_sample_rate <= 8000 else "wb"

    for batch_idx, (mix, sources, rows) in enumerate(loader):
        mix = mix.to(device)         # [B, 1, T]
        sources = sources.to(device) # [B, S, 1, T]
        B, S, _, T = sources.shape
        assert S == 2, "Script assumes 2 speakers."

        # ---------- DAC encode mixture ----------
        mix_enc, orig_len = dac.get_encoded_features(mix)  # [B, C_lat, T_lat], [B]
        mix_q, _, _, _, _ = dac.get_quantized_features(mix_enc)  # [B, C_lat, T_lat]
        z_mix = mix_q.permute(0, 2, 1).contiguous()               # [B, T_lat, C_lat]

        # ---------- RWKV-v7 separation ----------
        sep_lat = model(z_mix)                                    # [B, T_lat, S, C_lat]
        sep_lat = sep_lat.permute(0, 2, 3, 1).contiguous()        # [B, S, C_lat, T_lat]

        # ---------- DAC decode each speaker ----------
        est_wavs = []
        for s_idx in range(S):
            z_s = sep_lat[:, s_idx, :, :]                         # [B, C_lat, T_lat]
            wav_hat = dac.get_decoded_signal(z_s, orig_len)       # [B, 1, T']
            est_wavs.append(wav_hat)

        est_sources = torch.stack(est_wavs, dim=1).squeeze(2)     # [B, S, T']
        # Make sure target has same time dim as est; chop/pad if needed
        T_est = est_sources.shape[-1]
        if T_est != T:
            if T_est < T:
                pad = T - T_est
                est_sources = F.pad(est_sources, (0, pad))
                T_est = T
            else:
                est_sources = est_sources[..., :T]
                T_est = T

        # Clean reference: [B, S, T_est]
        true_sources_clean = sources.squeeze(2)[..., :T_est]

        # Codec reference: t = Codec(s)
        B2, S2, T2 = true_sources_clean.shape
        assert B2 == B and S2 == S
        sources_flat = true_sources_clean.view(B * S, 1, T_est)   # [B*S,1,T]

        src_enc, src_orig_len = dac.get_encoded_features(sources_flat)    # [B*S,C_lat,T_lat],[B*S]
        src_q, _, _, _, _ = dac.get_quantized_features(src_enc)          # [B*S,C_lat,T_lat]
        codec_src_flat = dac.get_decoded_signal(src_q, src_orig_len)     # [B*S,1,T_est]
        codec_sources = codec_src_flat.view(B, S, T_est)                 # [B,S,T_est]

        # ---------- SI-SDR metrics ----------
        si_sdr_clean = pit_si_sdr_per_utt(est_sources, true_sources_clean)  # [B]
        si_sdr_codec = pit_si_sdr_per_utt(est_sources, codec_sources)       # [B]

        all_si_sdr_clean.append(si_sdr_clean.cpu())
        all_si_sdr_codec.append(si_sdr_codec.cpu())

        # ---------- Perceptual metrics ----------
        # Work on CPU numpy arrays for PESQ/STOI; DNSMOS can use torch.
        if enable_pesq or enable_stoi or enable_dnsmos:
            est_np   = est_sources.cpu().numpy()          # [B, S, T_est]
            clean_np = true_sources_clean.cpu().numpy()   # [B, S, T_est]

        batch_pesq = []
        batch_stoi = []
        batch_dnsmos = []

        for i in range(B):
            spk_pesq = []
            spk_stoi = []
            spk_dnsmos = []

            for s_idx in range(S):
                ref = clean_np[i, s_idx, :]
                deg = est_np[i, s_idx, :]

                # PESQ
                if enable_pesq:
                    try:
                        score_pesq = float(pesq_fn(input_sample_rate, ref, deg, pesq_mode))
                        spk_pesq.append(score_pesq)
                    except Exception as e:
                        # PESQ can throw for too-short signals or invalid lengths
                        pass

                # STOI
                if enable_stoi:
                    try:
                        score_stoi = float(stoi_fn(ref, deg, input_sample_rate, extended=False))
                        spk_stoi.append(score_stoi)
                    except Exception as e:
                        pass

                # DNSMOS (non-intrusive, no reference)
                if enable_dnsmos:
                    try:
                        # TorchMetrics DNSMOS expects Tensor [..., time]
                        wav_t = torch.from_numpy(deg).to(device)
                        scores = dnsmos_fn(wav_t, fs=input_sample_rate, personalized=False)
                        # scores: [4] = [p808_mos, mos_sig, mos_bak, mos_ovr]
                        mos_ovr = float(scores[-1].item())
                        spk_dnsmos.append(mos_ovr)
                    except Exception as e:
                        pass

            # Average over speakers if we have any valid scores
            if spk_pesq:
                batch_pesq.append(np.mean(spk_pesq))
            if spk_stoi:
                batch_stoi.append(np.mean(spk_stoi))
            if spk_dnsmos:
                batch_dnsmos.append(np.mean(spk_dnsmos))

        if batch_pesq:
            all_pesq.extend(batch_pesq)
        if batch_stoi:
            all_stoi.extend(batch_stoi)
        if batch_dnsmos:
            all_dnsmos.extend(batch_dnsmos)

        # ---------- Optional: save separated audio ----------
        if save_audio_dir is not None:
            est_np = est_sources.cpu().numpy()  # [B, S, T_est]
            for i in range(B):
                row = rows[i]
                base_name = os.path.splitext(os.path.basename(row["mix_path"]))[0]
                for s_idx in range(S):
                    out_path = os.path.join(
                        save_audio_dir,
                        f"{base_name}_spk{s_idx+1}.wav",
                    )
                    sf.write(out_path, est_np[i, s_idx], samplerate=input_sample_rate)

        # ---------- Per-batch summary ----------
        msg = (f"[BATCH {batch_idx:04d}] "
               f"SI-SDR clean={si_sdr_clean.mean().item():.2f} dB, "
               f"Codec SI-SDR={si_sdr_codec.mean().item():.2f} dB")
        if batch_pesq:
            msg += f", PESQ={np.mean(batch_pesq):.3f}"
        if batch_stoi:
            msg += f", STOI={np.mean(batch_stoi):.3f}"
        if batch_dnsmos:
            msg += f", DNSMOS_ovr={np.mean(batch_dnsmos):.3f}"
        print(msg)

    # ---- Aggregate ----
    all_si_sdr_clean = torch.cat(all_si_sdr_clean, dim=0)
    all_si_sdr_codec = torch.cat(all_si_sdr_codec, dim=0)

    print("\n========== EVALUATION SUMMARY ==========")
    print(f"Num utterances: {all_si_sdr_clean.numel()}")
    print(f"Mean SI-SDR vs clean: {all_si_sdr_clean.mean().item():.3f} dB")
    print(f"Mean Codec SI-SDR   : {all_si_sdr_codec.mean().item():.3f} dB")

    if all_pesq:
        print(f"Mean PESQ (avg over speakers): {np.mean(all_pesq):.3f}")
    if all_stoi:
        print(f"Mean STOI (avg over speakers): {np.mean(all_stoi):.3f}")
    if all_dnsmos:
        print(f"Mean DNSMOS overall MOS      : {np.mean(all_dnsmos):.3f}")
    print("========================================\n")


# =========================
#   MAIN
# =========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True,
                    help="CSV with mix_path,s1_path,s2_path for evaluation set.")
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to trained checkpoint (.pt) from train_rwkv_cfstyle_withLRsched.py.")
    ap.add_argument("--sample_rate", type=int, default=8000,
                    help="Input sample rate (will be resampled to DAC SR internally).")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save_audio_dir", type=str, default=None,
                    help="If set, save separated WAVs into this directory.")
    ap.add_argument("--enable_pesq", action="store_true",
                    help="Compute PESQ scores (requires 'pesq' package).")
    ap.add_argument("--enable_stoi", action="store_true",
                    help="Compute STOI scores (requires 'pystoi' package).")
    ap.add_argument("--enable_dnsmos", action="store_true",
                    help="Compute DNSMOS MOS scores (requires torchmetrics['audio']).")
    args = ap.parse_args()

    # Device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # ----- Load checkpoint -----
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_config = ckpt.get("config", None)
    latent_channels = ckpt.get("latent_channels", None)

    if ckpt_config is not None:
        print("[INFO] Loaded config from checkpoint:")
        for k, v in ckpt_config.items():
            print(f"  {k}: {v}")
    if latent_channels is None:
        raise RuntimeError("Checkpoint does not contain 'latent_channels' key.")

    # ----- DAC wrapper (same as training) -----
    dac = DACWrapper(
        input_sample_rate=args.sample_rate,
        DAC_model_path=None,
        DAC_sample_rate=16000,
        Freeze=True,
    )
    dac.model.to(device)
    dac.dac_sampler.to(device)
    dac.org_sampler.to(device)

    # ----- Build RWKV separator -----
    n_layer = ckpt_config.get("n_layer", 4) if ckpt_config is not None else 4
    head_mode = ckpt_config.get("head_mode", "mask") if ckpt_config is not None else "mask"

    model = build_rwkv7_separator(
        n_embd=latent_channels,
        n_layer=n_layer,
        num_sources=2,
        head_mode=head_mode,
        enforce_bf16=False,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Loaded model with {num_params:,} parameters.")

    # ----- Dataset & DataLoader -----
    eval_ds = Wsj02MixEvalDataset(args.csv, sample_rate=args.sample_rate)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # ----- Run evaluation -----
    evaluate(
        model=model,
        dac=dac,
        loader=eval_loader,
        device=device,
        input_sample_rate=args.sample_rate,
        save_audio_dir=args.save_audio_dir,
        enable_pesq=args.enable_pesq,
        enable_stoi=args.enable_stoi,
        enable_dnsmos=args.enable_dnsmos,
    )


if __name__ == "__main__":
    main()
