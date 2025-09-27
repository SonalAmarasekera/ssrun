#!/usr/bin/env python
"""
Encode Libri2Mix WAVs (mix_clean, s1, s2) with the 16-kHz Descript Audio Codec
and save latents (.pt) into separate subfolders.

Usage:
    # For train-100
    python cache_latents.py --csv data/train-100.csv --out_dir data/latents/train-100

    # For dev
    python cache_latents.py --csv data/dev.csv --out_dir data/latents/dev

    # For test
    python cache_latents.py --csv data/test.csv --out_dir data/latents/test
"""

import argparse
import csv
import pathlib
import multiprocessing as mp
import soundfile as sf
import torch
import dac
import tqdm


# ----------------------------------------------------------------------
# helper: encode one file → save .pt
# args: (wav_path, save_path)
# ----------------------------------------------------------------------
def encode_path(args):
    wav_path, save_path = args
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000, f"Expected 16 kHz audio, got {sr} @ {wav_path}"
    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        z, *_ = _CODEC.encode(wav)  # shape: [1, 128, T_latent]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z.cpu()}, save_path)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: mix_path,s1_path,s2_path")
    ap.add_argument("--out_dir", required=True, help="Root folder for latents of this split (e.g., latents/train-100)")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    out_root = pathlib.Path(args.out_dir)

    # Where each CSV column should be stored
    column_to_subdir = {
        "mix_path": "mix_clean",
        "s1_path": "s1",
        "s2_path": "s2",
    }

    # Build (wav_path, save_path) jobs
    jobs = []
    for row in rows:
        for col, subdir in column_to_subdir.items():
            if col not in row or not row[col]:
                continue  # tolerate CSVs that might omit a column
            wav_path = pathlib.Path(row[col])
            save_path = out_root / subdir / (wav_path.stem + ".pt")
            if not save_path.exists():  # skip already encoded
                jobs.append((wav_path, save_path))

    print(f"[cache_latents] Split root: {out_root}")
    print(f"[cache_latents] To encode: {len(jobs)} files "
          f"→ subfolders {{'mix_clean','s1','s2'}}")

    if not jobs:
        return

    # Safer cross-platform start method
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        list(tqdm.tqdm(pool.imap(encode_path, jobs), total=len(jobs)))


# ----------------------------------------------------------------------
# Global codec (loaded once). Workers import the module and reuse this.
_CODEC = dac.DAC.load(dac.utils.download("16khz")).eval().cuda()
torch.set_grad_enabled(False)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
