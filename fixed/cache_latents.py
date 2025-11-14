#!/usr/bin/env python
"""
FIXED VERSION: cache_latents.py
Encode WSJ0 source WAVs with the Descript Audio Codec
and save latent tensors (.pt) for later training.

Key fixes:
- Fixed model loading order (after CLI parsing)
- Dynamic FPS calculation based on actual DAC hop length
- Better path validation and error handling
- Improved metadata storage
"""

import argparse
import csv
import math
import os
import pathlib
import sys
import multiprocessing as mp
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import torch
import tqdm

import dac


# ---------------------------- Audio discovery (mirror mode) ----------------------------

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def find_audio_paths(root: pathlib.Path) -> List[pathlib.Path]:
    if root.is_file():
        return [root]
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            files.append(p)
    return files


# ---------------------------- FPS Calculation ----------------------------

def get_dac_fps(model, sr: int) -> float:
    """
    Calculate actual FPS based on DAC model's hop length.
    
    DAC typically uses:
    - 16kHz model: hop_length=320 → 50 fps
    - 24kHz model: hop_length=320 → 75 fps  
    - 44.1kHz model: hop_length=512 → ~86.13 fps
    """
    if hasattr(model, 'hop_length'):
        hop = model.hop_length
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'hop_length'):
        hop = model.encoder.hop_length
    else:
        # Fallback based on common DAC configurations
        if sr == 44100:
            hop = 512
        else:
            hop = 320
        print(f"[WARNING] Could not detect hop_length from model, using default {hop} for sr={sr}")
    
    fps = sr / hop
    return fps


# ---------------------------- Windowed encoding utility ----------------------------

def chunk_indices(num_samples: int, sr: int, win_seconds: float) -> List[Tuple[int, int]]:
    if win_seconds <= 0:
        return [(0, num_samples)]
    hop = int(sr * win_seconds)
    if hop <= 0:
        return [(0, num_samples)]
    idx = []
    start = 0
    while start < num_samples:
        end = min(num_samples, start + hop)
        idx.append((start, end))
        start = end
    return idx


# ---------------------------- Latent Statistics ----------------------------

def compute_latent_stats(z: torch.Tensor) -> Dict:
    """Compute statistics of latent tensor for normalization."""
    z_flat = z.reshape(-1)
    return {
        "mean": float(z_flat.mean().item()),
        "std": float(z_flat.std().item()),
        "min": float(z_flat.min().item()),
        "max": float(z_flat.max().item()),
    }


# ---------------------------- Encoding core ----------------------------

def _encode_tensor(
    wav_f32: np.ndarray,
    sr: int,
    device: str,
    win_seconds: float,
    codec_kwargs: Dict
) -> torch.Tensor:
    """Return latent tensor z: [1, C, T_latent]."""
    assert sr in (16000, 24000, 44100), f"Unsupported sample rate {sr}; choose model_type to match."
    wav = torch.from_numpy(wav_f32).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,N]
    with torch.no_grad():
        if win_seconds <= 0:
            z, *_ = _CODEC.encode(wav, **codec_kwargs)
            return z
        # windowed: concatenate latents along time dimension
        N = wav.shape[-1]
        pieces = []
        for s, e in chunk_indices(N, sr, win_seconds):
            z_part, *_ = _CODEC.encode(wav[..., s:e], **codec_kwargs)
            pieces.append(z_part)
        z = torch.cat(pieces, dim=-1)
        return z


def _write_pt(save_path: pathlib.Path, z: torch.Tensor, 
              sr: int, model_type: str, bitrate: Optional[str] = None,
              n_quantizers: Optional[int] = None):
    """
    Save latents as [T, C] float32 with comprehensive metadata.
    
    FIXED: Now calculates actual FPS and stores proper metadata.
    """
    # z is [1, C, T] from _CODEC.encode
    z = z.detach().cpu()
    assert z.ndim == 3 and z.shape[0] == 1, f"unexpected latent shape {tuple(z.shape)}"
    z = z.squeeze(0)             # [C, T]
    C, T = int(z.shape[0]), int(z.shape[1])
    
    # Calculate actual FPS
    fps = get_dac_fps(_CODEC, sr)
    
    # Transpose to standard [T, C] format
    z_tc = z.transpose(0, 1).contiguous()  # → [T, C]
    
    # Compute statistics for potential normalization
    stats = compute_latent_stats(z_tc)
    
    meta = {
        "z": z_tc,                      # [T, C] float32
        "C": C,                         # Number of channels
        "T": T,                         # Number of time frames
        "shape_order": "TC",            # Explicit shape order indicator
        "sr": sr,                       # Sample rate of original audio
        "fps": fps,                     # Actual frames per second
        "model_type": model_type,       # DAC model type
        "bitrate": bitrate,             # Bitrate if used
        "n_quantizers": n_quantizers,   # Number of quantizers if specified
        "stats": stats,                 # Statistics for normalization
        "dac_version": getattr(dac, '__version__', 'unknown'),
        "hop_length": getattr(_CODEC, 'hop_length', 
                            getattr(_CODEC.encoder, 'hop_length', None) if hasattr(_CODEC, 'encoder') else None),
    }
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(meta, save_path)
    
    # Log first save for debugging
    #if not hasattr(_write_pt, '_logged'):
    #    print(f"[DEBUG] First latent saved with shape [T={T}, C={C}], fps={fps:.2f}, stats={stats}")
    #    _write_pt._logged = True


# ---------------------------- Worker wrapper ----------------------------

def _worker_encode(args):
    """Worker-safe wrapper: (wav_path, save_path, device, expect_sr, win_seconds, codec_kwargs, model_type, bitrate, n_quantizers)."""
    wav_path, save_path, device, expect_sr, win_seconds, codec_kwargs, model_type, bitrate, n_quantizers = args
    if save_path.exists():
        return  # idempotent
    
    # Validate input file exists
    if not wav_path.exists():
        print(f"[ERROR] Input file not found: {wav_path}")
        return
        
    try:
        # read audio file
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        if wav.ndim == 2:  # mixdown if stereo
            wav = wav.mean(axis=1)
        if expect_sr is not None:
            assert sr == expect_sr, f"Expected {expect_sr}Hz, got {sr} @ {wav_path}"
        # encode
        z = _encode_tensor(wav, sr, device, win_seconds, codec_kwargs)
        _write_pt(save_path, z, sr, model_type, bitrate, n_quantizers)
    except Exception as e:
        print(f"[ERROR] Failed to process {wav_path}: {e}")
        return


# ---------------------------- Job builders ----------------------------

def build_jobs_from_csv(
    csv_path: pathlib.Path,
    out_root: pathlib.Path,
    mirror_from: Optional[pathlib.Path],
    expect_sr: Optional[int],
    device: str,
    win_seconds: float,
    codec_kwargs: Dict,
    model_type: str,
    bitrate: Optional[str],
    n_quantizers: Optional[int]
) -> List[Tuple]:
    rows = list(csv.DictReader(open(csv_path)))
    # column → subdir mapping
    col2sub = {"mix_path": "mix_clean", "s1_path": "s1", "s2_path": "s2"}
    jobs = []
    
    print(f"[DEBUG] Processing CSV: {len(rows)} rows")
    valid_files = 0
    missing_files = 0
    
    for i, row in enumerate(rows):
        for col, sub in col2sub.items():
            p = row.get(col, "")
            if not p:
                continue
            wav_path = pathlib.Path(p)
            
            # Validate path exists
            if not wav_path.exists():
                print(f"[WARNING] File not found (row {i}, {col}): {wav_path}")
                missing_files += 1
                continue
                
            valid_files += 1
            
            # mirrored relative path (if requested)
            if mirror_from is not None:
                try:
                    rel = pathlib.Path(os.path.relpath(str(wav_path), str(mirror_from)))
                except ValueError:
                    # Fallback: use filename only if mirror_from doesn't match
                    print(f"[WARNING] Cannot compute relative path for {wav_path} from {mirror_from}")
                    rel = pathlib.Path(wav_path.name)
                save_path = out_root / sub / rel.with_suffix(".pt")
            else:
                save_path = out_root / sub / (wav_path.stem + ".pt")
                
            jobs.append((wav_path, save_path, device, expect_sr, win_seconds, 
                        codec_kwargs, model_type, bitrate, n_quantizers))
    
    print(f"[INFO] CSV processing: {valid_files} valid files, {missing_files} missing files")
    if missing_files > 0:
        print(f"[WARNING] {missing_files} files from CSV were not found!")
        
    return jobs


def build_jobs_from_mirror(
    in_path: pathlib.Path,
    out_root: pathlib.Path,
    expect_sr: Optional[int],
    device: str,
    win_seconds: float,
    codec_kwargs: Dict,
    model_type: str,
    bitrate: Optional[str],
    n_quantizers: Optional[int]
) -> List[Tuple]:
    in_path = in_path.resolve()
    files = find_audio_paths(in_path)
    jobs = []
    for f in files:
        rel = f.relative_to(in_path) if in_path.is_dir() else pathlib.Path(f.name)
        save_path = (out_root / rel).with_suffix(".pt")
        jobs.append((f, save_path, device, expect_sr, win_seconds, 
                    codec_kwargs, model_type, bitrate, n_quantizers))
    return jobs


# ---------------------------- Codec Loading ----------------------------

def _load_codec(model_type: str, device: str, weights_path: Optional[str], model_tag: Optional[str]):
    """
    Load DAC model. We prefer built-in downloads; fall back to tag/weights if provided.
    """
    # Default path: official downloads for 16/24/44 kHz
    try:
        print(f"[INFO] Loading DAC model: {model_type}")
        model = dac.DAC.load(dac.utils.download(model_type))
    except Exception as e:
        print(f"[WARNING] Failed to load default model {model_type}: {e}")
        # Optional alternate pathways
        if weights_path is not None:
            print(f"[INFO] Trying custom weights: {weights_path}")
            model = dac.DAC.load(weights_path)
        elif model_tag is not None:
            print(f"[INFO] Trying model tag: {model_tag}")
            model = dac.DAC.load(dac.utils.download(model_tag))
        else:
            raise RuntimeError(f"Could not load DAC model {model_type} and no fallbacks provided")
    
    # Log model info for debugging
    print(f"[INFO] Successfully loaded DAC model: {model_type}")
    if hasattr(model, 'sample_rate'):
        print(f"[INFO] Model sample_rate: {model.sample_rate}")
    if hasattr(model, 'encoder'):
        print(f"[INFO] Encoder channels: {model.encoder.channels if hasattr(model.encoder, 'channels') else 'unknown'}")
    if hasattr(model, 'hop_length'):
        print(f"[INFO] Model hop_length: {model.hop_length}")
    
    return model.eval().to(device)


# ---------------------------- CLI and main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    # Input selection
    g_in = ap.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--csv", type=str, help="CSV with mix_path,s1_path,s2_path")
    g_in.add_argument("--in_path", type=str, help="File or directory (mirror mode)")
    ap.add_argument("--out_dir", required=True, type=str, help="Root folder to write latents")

    # Mirroring options (CSV mode only)
    ap.add_argument("--mirror_from", type=str, default=None,
                    help="Root folder to preserve substructure relative to this path (CSV mode)")

    # Performance mode
    ap.add_argument("--fast_mp", action="store_true", help="Use multiprocessing (faster, higher VRAM)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))

    # Codec/model options (like encode.py)
    ap.add_argument("--model_type", type=str, default="16khz", choices=["16khz", "24khz", "44khz"])
    ap.add_argument("--bitrate", type=str, default=None, choices=[None, "8kbps", "16kbps"])
    ap.add_argument("--n_quantizers", type=int, default=None, help="Override number of residual quantizers if supported")
    ap.add_argument("--weights_path", type=str, default=None, help="Path to custom model weights")
    ap.add_argument("--model_tag", type=str, default=None, help="Alternative tag identifying model weights")

    # Device & memory
    ap.add_argument("--device", type=str, default=None, help="'cuda' or 'cpu'. Default: auto")
    ap.add_argument("--win_seconds", type=float, default=0.0,
                    help="Encode in windows of N seconds (0 = full file). Helps on low VRAM.")

    args = ap.parse_args()

    out_root = pathlib.Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Resolve device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Expected SR per model_type
    expect_sr = {"16khz": 16000, "24khz": 24000, "44khz": 44100}[args.model_type]

    # Build codec kwargs for encode()
    codec_kwargs: Dict = {}
    if args.n_quantizers is not None:
        codec_kwargs["n_quantizers"] = int(args.n_quantizers)
    if args.bitrate is not None:
        codec_kwargs["bitrate"] = args.bitrate

    # Load model AFTER parsing arguments (FIXED)
    global _CODEC
    _CODEC = _load_codec(args.model_type, device, args.weights_path, args.model_tag)
    torch.set_grad_enabled(False)

    # Jobs
    if args.csv:
        mirror_from = pathlib.Path(args.mirror_from).resolve() if args.mirror_from else None
        jobs = build_jobs_from_csv(
            pathlib.Path(args.csv),
            out_root,
            mirror_from,
            expect_sr,
            device,
            args.win_seconds,
            codec_kwargs,
            args.model_type,
            args.bitrate,
            args.n_quantizers,
        )
    else:
        jobs = build_jobs_from_mirror(
            pathlib.Path(args.in_path),
            out_root,
            expect_sr,
            device,
            args.win_seconds,
            codec_kwargs,
            args.model_type,
            args.bitrate,
            args.n_quantizers,
        )

    print(f"[cache_latents] model_type={args.model_type} device={device} "
          f"win_seconds={args.win_seconds} fast_mp={args.fast_mp}")
    print(f"[cache_latents] jobs: {len(jobs)}  →  out_dir={out_root}")

    if not jobs:
        print("[WARNING] No valid jobs to process!")
        return

    # Run with progress bar
    print(f"[INFO] Starting encoding process...")
    if args.fast_mp:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers) as pool:
            # Use imap_unordered for better progress bar updates
            for _ in tqdm.tqdm(pool.imap_unordered(_worker_encode, jobs), 
                             total=len(jobs), 
                             desc="Encoding latents", 
                             unit="file",
                             ncols=80):
                pass
    else:
        for job in tqdm.tqdm(jobs, 
                           total=len(jobs), 
                           desc="Encoding latents", 
                           unit="file",
                           ncols=80):
            _worker_encode(job)
    
    print(f"[SUCCESS] Completed processing {len(jobs)} files")


# Global codec variable (will be initialized in main())
_CODEC = None

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()