
#!/usr/bin/env python
"""
cache_latents_rwkv_v7_fixed.py

Purpose
-------
Pre-compute codec embeddings (latents) from WAV files and save them as PyTorch .pt
payloads with **layout = [T, C]** which is the preferred on-disk convention for
feeding **RWKV-v7** (training will stack a batch dim to get [B, T, C]).

This script:
- Accepts either a CSV list (mixture, s1, s2) or an input directory to scan.
- Encodes audio using a Neural Audio Codec (e.g., Descript Audio Codec / EnCodec).
- Saves **.pt** with z.shape == [T, C] plus rich metadata (fps, codec tag, sr, bitrate, n_quantizers, etc.).
- Mirrors input directory structure under --out_dir, ensuring files land in their respective folders.

Why [T, C] on disk?
-------------------
RWKV-v7 blocks typically operate on tensors laid out as [B, T, C] during training/inference.
Storing per-file latents as [T, C] keeps them batch-agnostic; the dataloader adds the B dimension.

Usage
-----
1) CSV mode:
   python cache_latents_rwkv_v7_fixed.py \
       --csv data/libri2mix_train.csv \
       --out_dir data/latents/train \
       --model_type dac_16khz --device cuda

   CSV must have headers: mix_path,s1_path,s2_path
   (You can also pass --save_targets 0 to only save mixture embeddings.)

2) Directory mode (single-stream, e.g., just mixtures):
   python cache_latents_rwkv_v7_fixed.py \
       --in_dir data/mixtures \
       --out_dir data/latents/mixtures \
       --model_type dac_16khz --device cuda --mirror_dirs 1

Notes
-----
- This file does not ship the codec. It expects a codec "encode" function returning latents as [1, C, T] or [C, T].
- This script will normalize shape to [T, C].
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import torchaudio

# --------------------------------- Codec loader ----------------------------------

def load_codec(model_type: str, device: str, bitrate: Optional[float], n_quantizers: Optional[int]):
    """
    Replace this stub with your actual codec loader.
    It must return an object with a .encode(waveform, sr) -> latent Tensor
    where latent is shaped either [1, C, T] or [C, T] (will be normalized).
    """
    try:
        # Example for DAC if you have a wrapper:
        # from dac_wrapper import load_dac
        # return load_dac(model_type=model_type, device=device, bitrate=bitrate, n_quantizers=n_quantizers)
        pass
    except Exception as e:
        raise RuntimeError(f"Failed to load codec '{model_type}': {e}")
    return DummyCodec(model_type, device, bitrate, n_quantizers)


class DummyCodec:
    """Fallback stub that simply downsamples audio to 100 fps frames of 64-D as a placeholder."""
    def __init__(self, model_type, device, bitrate, n_quantizers):
        self.model_type = model_type
        self.device = device
        self.bitrate = bitrate
        self.n_quantizers = n_quantizers
        self.fps = 320  # typical latent rate used by many 16 kHz codecs; replace if your codec exposes its fps

    def encode(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # wav: [1, T]
        # Produce a fake latent: frame @ 10 ms = ~100 fps, but we'll use self.fps
        hop = int(sr / self.fps)
        if hop <= 0: hop = max(1, sr // 320)
        T = wav.shape[-1] // hop
        C = 64
        z = torch.randn(1, C, T, dtype=torch.float32, device=wav.device)
        return z

# -------------------------------- Utility helpers --------------------------------

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}

def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def guess_fps_from_codec(codec_obj, sr: int) -> int:
    # Try to read from codec, else fallback by sample rate
    fps = getattr(codec_obj, "fps", None)
    if isinstance(fps, int) and fps > 0:
        return fps
    # Fallbacks (adjust per your codec):
    # Many 16 kHz codecs use ~320 fps; 24 kHz ~ 480; 44.1 kHz ~ 882
    if sr == 16000:
        return 320
    if sr == 24000:
        return 480
    if sr in (44100, 48000):
        return 882 if sr == 44100 else 960
    return 320

def normalize_to_TxC(latent: torch.Tensor) -> torch.Tensor:
    """
    Normalize latent to shape [T, C].
    Accepts [1, C, T], [C, T], or [T, C].
    """
    if latent.dim() == 3 and latent.shape[0] == 1:
        # [1, C, T] -> [T, C]
        latent = latent.squeeze(0).permute(1, 0).contiguous()
    elif latent.dim() == 2:
        # [C, T] or [T, C] -> disambiguate by assuming more time than channels
        if latent.shape[0] < latent.shape[1]:
            # likely [C, T]
            latent = latent.permute(1, 0).contiguous()
        # else assume already [T, C]
    else:
        raise ValueError(f"Unsupported latent shape: {tuple(latent.shape)}")
    return latent

def save_pt(save_path: Path, z_TxC: torch.Tensor, meta: Dict):
    ensure_parent(save_path)
    payload = {
        "z": z_TxC.cpu().float(),   # [T, C]
        "layout": "TC",
        **meta
    }
    torch.save(payload, save_path)

def mirror_rel_path(src: Path, in_root: Path) -> Path:
    """Return the path of src relative to in_root, preserving subdirectories."""
    try:
        return src.relative_to(in_root)
    except Exception:
        # Fallback: use just the file name
        return Path(src.name)

# ------------------------------ Core processing ----------------------------------

def process_file(wav_path: Path, out_dir: Path, codec, device: str,
                 bitrate: Optional[float], n_quantizers: Optional[int],
                 expect_sr: Optional[int], mirror_dirs: bool,
                 in_root: Optional[Path]) -> Path:
    wav_path = wav_path.resolve()
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)

    wav, sr = torchaudio.load(str(wav_path))
    if wav.dim() == 2:
        wav = wav[:1]  # make mono [1, T]
    if expect_sr and sr != expect_sr:
        # Resample to expected sr (safer than assert)
        wav = torchaudio.functional.resample(wav, sr, expect_sr)
        sr = expect_sr

    wav = wav.to(device)
    z = codec.encode(wav, sr)  # expected [1, C, T] or [C, T]; we normalize next
    z_TxC = normalize_to_TxC(z)

    # Destination path
    if in_root and mirror_dirs:
        rel = mirror_rel_path(wav_path, in_root).with_suffix(".pt")
        dst = out_dir / rel
    else:
        dst = out_dir / (wav_path.stem + ".pt")

    meta = {
        "fps": guess_fps_from_codec(codec, sr),
        "sr": sr,
        "codec": getattr(codec, "model_type", "unknown"),
        "bitrate": bitrate,
        "n_quantizers": n_quantizers,
        "source_path": str(wav_path),
        "schema_version": 2,
        "dtype": "float32",
        "shape": tuple(z_TxC.shape)
    }
    save_pt(dst, z_TxC, meta)
    return dst

def scan_audio_files(in_dir: Path) -> List[Path]:
    files = []
    for p in in_dir.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS and p.is_file():
            files.append(p)
    files.sort()
    return files

# ------------------------------------- CLI ---------------------------------------

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--csv", type=str, help="CSV with headers: mix_path,s1_path,s2_path")
    g.add_argument("--in_dir", type=str, help="Directory containing WAVs to encode")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model_type", type=str, default="dac_16khz")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bitrate", type=float, default=None)
    ap.add_argument("--n_quantizers", type=int, default=None)
    ap.add_argument("--expect_sr", type=int, default=16000, help="Resample to this SR if different")
    ap.add_argument("--mirror_dirs", type=int, default=1, help="Mirror input subdirs under out_dir (1/0)")
    ap.add_argument("--save_targets", type=int, default=1, help="CSV mode: also save s1/s2 (1/0)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    codec = load_codec(args.model_type, args.device, args.bitrate, args.n_quantizers)

    if args.csv:
        csv_path = Path(args.csv)
        assert csv_path.exists(), f"CSV not found: {csv_path}"
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            required = {"mix_path", "s1_path", "s2_path"}
            assert required.issubset(reader.fieldnames), f"CSV must have headers: {required}"
            for row in reader:
                mix = Path(row["mix_path"])
                s1 = Path(row["s1_path"])
                s2 = Path(row["s2_path"])

                # mixture
                process_file(
                    mix, out_dir / "mix", codec, args.device, args.bitrate, args.n_quantizers,
                    args.expect_sr, bool(args.mirror_dirs), mix.parent
                )

                if args.save_targets:
                    # targets
                    process_file(
                        s1, out_dir / "s1", codec, args.device, args.bitrate, args.n_quantizers,
                        args.expect_sr, bool(args.mirror_dirs), s1.parent
                    )
                    process_file(
                        s2, out_dir / "s2", codec, args.device, args.bitrate, args.n_quantizers,
                        args.expect_sr, bool(args.mirror_dirs), s2.parent
                    )
    else:
        in_dir = Path(args.in_dir)
        assert in_dir.exists(), f"Input dir not found: {in_dir}"
        wavs = scan_audio_files(in_dir)
        for wav in wavs:
            process_file(
                wav, Path(args.out_dir), codec, args.device, args.bitrate, args.n_quantizers,
                args.expect_sr, bool(args.mirror_dirs), in_dir
            )

if __name__ == "__main__":
    main()
