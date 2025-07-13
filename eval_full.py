"""
Evaluate a trained RWKV-DAC separator on a Libri2Mix CSV and
output objective metrics (SI-SDR / SI-SDR i, SDR / SDR i, STOI, PESQ-WB)
to JSON.

If a library is missing (pystoi, pypesq) the corresponding metric
is skipped silently.
"""
from __future__ import annotations
import argparse, csv, json, pathlib, warnings
from collections import defaultdict

import torch, soundfile as sf, dac
from separator_rwkv6 import RWKV6Separator

# ---------- optional metrics -----------------------------------------------
try:
    from pystoi import stoi                    # `pip install pystoi`
except ImportError:
    stoi = None

try:
    from pypesq import pesq as pesq_wb         # `pip install pypesq`
except ImportError:
    pesq_wb = None
# ---------------------------------------------------------------------------

ENC_HOP = 320      # 16-kHz DAC hop


# ---------- helpers --------------------------------------------------------
def crop_pair(a: torch.Tensor, b: torch.Tensor):
    """Crop both signals to the shorter length along the last axis."""
    L = min(a.size(-1), b.size(-1))
    return a[..., :L], b[..., :L]


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps=1e-8) -> float:
    est, ref = crop_pair(est, ref)
    ref, est = ref - ref.mean(), est - est.mean()
    s_target = (est * ref).sum() * ref / (ref.pow(2).sum() + eps)
    e_noise  = est - s_target
    return 10 * torch.log10((s_target.pow(2).sum() + eps) /
                            (e_noise .pow(2).sum() + eps))


def sdr(est: torch.Tensor, ref: torch.Tensor, eps=1e-8) -> float:
    est, ref = crop_pair(est, ref)
    err = ref - est
    return 10 * torch.log10((ref.pow(2).sum() + eps) /
                            (err.pow(2).sum() + eps))


def load_row(row: dict[str, str]):
    mix, s1, s2 = row["mix_path"], row["s1_path"], row["s2_path"]
    mix_w, _ = sf.read(mix, dtype="float32")
    s1_w,  _ = sf.read(s1,  dtype="float32")
    s2_w,  _ = sf.read(s2,  dtype="float32")
    return mix_w, s1_w, s2_w


# ---------------------------------------------------------------------------
def evaluate(csv_path: str,
             ckpt_path: str,
             out_json: str,
             batch_size: int = 1,
             device: str | None = None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- model & codec ---------------------------------------------
    codec = dac.DAC.load(dac.utils.download("16khz")).to(device).eval()
    model = RWKV6Separator(codec).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"],
                          strict=False)
    model.eval()

    rows = list(csv.DictReader(open(csv_path)))
    agg: defaultdict[str, list[float]] = defaultdict(list)

    # ---------- iterate CSV in mini-batches -------------------------------
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]

        mix_lst, ref_pairs = [], []
        for row in batch:
            mix_w, s1_w, s2_w = load_row(row)
            if len(mix_w) < ENC_HOP:
                warnings.warn(f"⤷ skipped < hop-size: {row['mix_path']}")
                continue
            mix_lst.append(torch.tensor(mix_w))
            ref_pairs.append((torch.tensor(s1_w), torch.tensor(s2_w)))

        if not mix_lst:
            continue

        mix_tensor = torch.stack(mix_lst).unsqueeze(1).to(device)

        with torch.no_grad():
            preds, _ = model(mix_tensor)                # list len = 2
            est_wavs = [codec.decode(p)[0].cpu()
                        for p in preds]                 # [2][B,1,T]

        for mix_w, (ref1, ref2), est1, est2 in zip(
                mix_lst, ref_pairs, est_wavs[0], est_wavs[1]):

            mix_t = torch.as_tensor(mix_w)

            for est, ref in ((est1.squeeze(), ref1), (est2.squeeze(), ref2)):
                # --- SI-SDR & improvement --------------------------------
                si_est = si_sdr(est, ref)
                si_mix = si_sdr(mix_t, ref)
                agg["sisdr"].append(float(si_est))
                agg["sisdr_i"].append(float(si_est - si_mix))

                # --- SDR & improvement -----------------------------------
                sd_est = sdr(est, ref)
                sd_mix = sdr(mix_t, ref)
                agg["sdr"].append(float(sd_est))
                agg["sdr_i"].append(float(sd_est - sd_mix))

                # ---------- STOI ----------------------------------------
                if stoi is not None:
                    x_c, y_c = crop_pair(est, ref)
                    agg["stoi"].append(
                        float(stoi(y_c.cpu().numpy(),
                                   x_c.cpu().numpy(), 16000,
                                   extended=False))
                    )

                # ---------- PESQ-WB -------------------------------------
                if pesq_wb is not None:
                    x_c, y_c = crop_pair(est, ref)
                    agg["pesq"].append(
                        float(pesq_wb(16000,
                                      y_c.cpu().numpy(),
                                      x_c.cpu().numpy()))
                    )

    if not agg:
        raise RuntimeError("No valid mixtures processed")

    results = {k: sum(v) / len(v) for k, v in agg.items()}
    pathlib.Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as fo:
        json.dump(results, fo, indent=2)
    print("Saved", out_json)
    for k, v in results.items():
        print(f"{k}: {v:.3f}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  required=True, help="Libri2Mix-style CSV")
    ap.add_argument("--ckpt", required=True, help="Separator checkpoint (.pt)")
    ap.add_argument("--out",  required=True, help="Output JSON path")
    ap.add_argument("--batch_size", type=int, default=1)
    args = ap.parse_args()

    evaluate(args.csv, args.ckpt, args.out, args.batch_size)
