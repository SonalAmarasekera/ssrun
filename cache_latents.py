#!/usr/bin/env python
"""
Encode Libri2Mix source WAVs with the 16-kHz Descript Audio Codec
and save latent tensors (.pt) for later training.

Usage:
    python cache_latents.py --csv data/train.csv --out_dir data/latents/train
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
# helper: encode one (s1 or s2) file → save .pt
# ----------------------------------------------------------------------
def encode_path(args):
    wav_path, save_path = args
    wav, sr = sf.read(wav_path, dtype="float32")
    assert sr == 16000, f"bad sr {sr} in {wav_path}"
    wav = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).cuda()
    with torch.no_grad():
        z, *_ = _CODEC.encode(wav)              # [1, 128, F]
    torch.save({"z": z.cpu()}, save_path)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--workers", type=int, default=mp.cpu_count() // 2)
    args = ap.parse_args()

    df = list(csv.DictReader(open(args.csv)))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build (wav_path, save_path) tuples
    jobs = []
    for row in df:
        for col in ("s1_path", "s2_path"):
            wav_path = pathlib.Path(row[col])
            save_path = out_dir / (wav_path.stem + ".pt")
            if not save_path.exists():          # skip already encoded
                jobs.append((wav_path, save_path))

    print(f"Total to encode: {len(jobs)}")

    ctx = mp.get_context("spawn")              # safer on all OSes
    with ctx.Pool(args.workers) as pool:
        list(tqdm.tqdm(pool.imap(encode_path, jobs), total=len(jobs)))


# ----------------------------------------------------------------------
# global codec (single GPU load, reused by forked processes)
# must be defined at module level so workers can import it
_CODEC = dac.DAC.load(dac.utils.download("16khz")).eval().cuda()
torch.set_grad_enabled(False)                  # save VRAM

# ----------------------------------------------------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)   # explicit & idempotent
    main()

