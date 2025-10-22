# train_separator.py
from __future__ import annotations
from pathlib import Path
import os, math, json, random
from dataclasses import dataclass, asdict
from typing import Optional, Callable

# ---- RWKV v7 CUDA JIT: must be set BEFORE importing rwkv_orig_model ----
os.environ.setdefault("RWKV_JIT_ON", "1")         # build the CUDA kernel at import
os.environ.setdefault("RWKV_CUDA_ON", "1")        # ensure CUDA path
os.environ.setdefault("RWKV_MY_TESTING", "x070")  # select v7 TimeMix/ChannelMix
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")  # v7 prefers bf16 activations
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")   # keep in sync with your config (head_size_a)
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial

from dataset_latents import RWKVLatentDataset
from collate_latents import collate_rwkv_latents
from rwkv_separator_v7_bi import RWKVv7Separator, SeparatorV7Config
from pit_losses import total_separator_loss

# RWKV-v7 fused kernel prefers chunked T that is a multiple of this.
CHUNK_LEN = 16

# ----------------- hyperparameters -----------------
@dataclass
class HParams:
    # data
    train_root: str = "/content/latents/min/train"
    val_root:   str = "/content/latents/min/test"
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True

    # segmenting (crop to fixed seconds during train for speed)
    seg_seconds: float = 6.0
    latent_fps: float = 50.0  # set to your DAC frame-rate (e.g., 50 or 75)

    # model
    layers: int = 16
    head_size_a: int = 64
    hidden_dim: int | None = None        # auto (C//2 rounded to multiple of head_size_a) when None
    dir_drop_p: float = 0.5              # train bi with direction dropout
    use_mask: bool = True
    enforce_bf16: bool = True

    # optimization
    epochs: int = 10
    lr: float = 8e-4
    weight_decay: float = 1e-2
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 5.0
    warmup_steps: int = 2000
    cosine_min_lr_ratio: float = 0.1

    # regularizers
    lambda_residual_l2: float = 1e-4
    lambda_mask_entropy: float = 0.0

    # ---- Embedding-Loss (CodecFormer-EL style) ----
    el_mode: str = "none"                # options: 'none' | 'latent' | 'decoder'
    lambda_el: float = 0.0               # scaling factor for the embedding loss
    el_cosine: bool = True               # cosine or L2 similarity

    # misc
    seed: int = 123
    log_every: int = 100
    val_every_steps: int = 1000
    ckpt_dir: str = "checkpoints"
    ema_decay: float = 0.999

hp = HParams()

# ----------------- helpers -----------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    @torch.no_grad()
    def store(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                v.copy_(self.shadow[k])

def cosine_lr(step: int, total: int, base_lr: float, min_ratio: float = 0.1) -> float:
    if step < hp.warmup_steps:
        return base_lr * (step + 1) / max(1, hp.warmup_steps)
    t = (step - hp.warmup_steps) / max(1, total - hp.warmup_steps)
    return base_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * (1 - t))))

def crop_batch_to_seconds(batch: dict, seg_seconds: float, fps: float, chunk_len: int) -> dict:
    """
    Crop each sample to ~seg_seconds using known latent fps; keeps [B,T,C] and mask shape.
    Frames are rounded up to a multiple of chunk_len to satisfy the kernel.
    """
    z_mix, z_s1, z_s2, mask = batch["z_mix"], batch.get("z_s1"), batch.get("z_s2"), batch["mask"]
    B, T, C = z_mix.shape
    frames = int(seg_seconds * float(fps))
    if chunk_len and frames % chunk_len != 0:
        frames = ((frames + chunk_len - 1) // chunk_len) * chunk_len
    if T <= frames:
        return batch  # no crop

    # choose per-sample random start within valid range
    starts = torch.randint(low=0, high=T - frames + 1, size=(B,), device=z_mix.device)
    idx = torch.arange(frames, device=z_mix.device)[None, :] + starts[:, None]  # [B,frames]
    idx = idx.clamp(0, T - 1)
    batch["z_mix"] = torch.gather(z_mix, 1, idx[..., None].expand(B, frames, C))
    if z_s1 is not None:
        batch["z_s1"] = torch.gather(z_s1, 1, idx[..., None].expand(B, frames, C))
    if z_s2 is not None:
        batch["z_s2"] = torch.gather(z_s2, 1, idx[..., None].expand(B, frames, C))
    batch["mask"] = torch.ones(B, frames, dtype=mask.dtype, device=mask.device)
    return batch

def autodetect_C(root: str, default_C: int = 1024) -> int:
    """
    Scan the latent cache to read 'C' from any .pt. Falls back to default_C if not found.
    """
    root = Path(root)
    any_pt: Optional[Path] = None
    # look first in a 'mixture' subdir (typical layout), else anywhere
    candidates = []
    if (root / "mixture").exists():
        candidates = list((root / "mixture").rglob("*.pt"))
    if not candidates:
        candidates = list(root.rglob("*.pt"))
    if candidates:
        any_pt = candidates[0]
        try:
            obj = torch.load(any_pt, map_location="cpu")
            return int(obj.get("C", default_C))
        except Exception:
            pass
    return default_C

# ---------- TEACHER HELPERS (Embedding-Loss) ----------
class SimpleLatentTeacher(nn.Module):
    """
    Lightweight teacher that maps DAC latents [B,T,C] -> [B,T,E] (or [B,E] after pooling).
    Keep it FROZEN (eval mode, no optimizer) so it acts as a teacher.
    """
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B,T,C] -> [B,T,E]
        return self.net(z)

def setup_teachers(
    hp,
    model_cfg,            # SeparatorV7Config (we use model_cfg.in_dim for latent teacher input size)
    device: torch.device
):
    """
    Returns (latent_teacher_fn, decode_fn, embed_fn) according to hp.el_mode:
      - 'latent'  -> (teacher, None, None)
      - 'decoder' -> (None, decode_fn, embed_fn)
      - 'none'    -> (None, None, None)
    NOTE: You can replace the placeholder decoder/embed with your real DAC + SSL models.
    """
    latent_teacher_fn = None
    decode_fn = None
    embed_fn = None

    if hp.el_mode == "latent":
        # Instantiate & FREEZE the latent teacher
        teacher = SimpleLatentTeacher(in_dim=model_cfg.in_dim, emb_dim=getattr(hp, "el_emb_dim", 128)).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        latent_teacher_fn = teacher

    elif hp.el_mode == "decoder":
        # ---- Replace these placeholders with your actual DAC decoder + embedding model ----
        # Example sketch:
        #
        # from my_dac_pkg import DAC
        # from my_ssl_pkg import Wav2VecEmbedder
        # dac = DAC.load_from_checkpoint("dac_decoder.ckpt").to(device).eval()
        # ssl = Wav2VecEmbedder.from_pretrained("facebook/wav2vec2-base").to(device).eval()
        #
        # def _decode_fn(latents: torch.Tensor) -> torch.Tensor:
        #     with torch.no_grad():
        #         wav = dac.decode(latents)       # [B,L] or [B,1,L]
        #     return wav
        #
        # def _embed_fn(wav: torch.Tensor) -> torch.Tensor:
        #     with torch.no_grad():
        #         emb = ssl(wav)                  # [B,E] or [B,T',E]
        #     return emb

        # Placeholders that raise if used without user replacement:
        def _decode_fn(_):
            raise RuntimeError("Please provide a real DAC decode_fn for decoder EL mode.")
        def _embed_fn(_):
            raise RuntimeError("Please provide a real embedding embed_fn for decoder EL mode.")

        decode_fn = _decode_fn
        embed_fn  = _embed_fn

    # else: 'none' -> all None

    return latent_teacher_fn, decode_fn, embed_fn

# ----------------- MAIN TRAIN -----------------
def main():
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Determine latent channel dimension C ----
    C_meta = autodetect_C(hp.train_root, default_C=1024)

    # ---- Data ----
    train_ds = RWKVLatentDataset(hp.train_root, require_targets=True, expected_C=C_meta)
    val_ds   = RWKVLatentDataset(hp.val_root,   require_targets=True, expected_C=C_meta)

    collate_fn = partial(collate_rwkv_latents, chunk_len=CHUNK_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=hp.batch_size, shuffle=True,
        num_workers=hp.num_workers, pin_memory=hp.pin_memory,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=hp.batch_size, shuffle=False,
        num_workers=hp.num_workers, pin_memory=hp.pin_memory,
        collate_fn=collate_fn, drop_last=False
    )

    # ---- Model (RWKV-v7 Bi core) ----
    cfg = SeparatorV7Config(
        in_dim=C_meta, layers=hp.layers, head_size_a=hp.head_size_a,
        hidden_dim=hp.hidden_dim, dir_drop_p=hp.dir_drop_p,
        use_mask=hp.use_mask, enforce_bf16=hp.enforce_bf16
    )
    model = RWKVv7Separator(cfg).to(device)

    # ---- Optional teachers for Embedding-Loss (you can replace these with real callables) ----
    latent_teacher_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None  # [B,T,C] -> [B,E] or [B,T,E]
    decode_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None          # [B,T,C] -> [B,L] waveform
    embed_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None           # [B,L]   -> [B,E] or [B,T',E]

    # ---- I/O ----
    os.makedirs(hp.ckpt_dir, exist_ok=True)
    with open(Path(hp.ckpt_dir) / "hparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    # ---- Optim & EMA ----
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, betas=hp.betas, weight_decay=hp.weight_decay)
    ema = EMA(model, hp.ema_decay)

    # ---- Training ----
    global_step = 0
    total_steps = hp.epochs * len(train_loader)

    for epoch in range(hp.epochs):
        model.train()
        for batch in train_loader:
            # Move to device & crop
            for k in ("z_mix", "z_s1", "z_s2", "mask"):
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)
            batch = crop_batch_to_seconds(batch, hp.seg_seconds, hp.latent_fps, CHUNK_LEN)

            z_mix, z_s1, z_s2, mask = batch["z_mix"], batch["z_s1"], batch["z_s2"], batch["mask"]
            B, T, C = z_mix.shape
            # shape contracts
            assert z_s1.shape == (B, T, C) and z_s2.shape == (B, T, C), "Target shapes must match mix."
            assert mask.shape == (B, T), "Mask must be [B,T]."
            assert C == cfg.in_dim, \
                f"Batch C={C} disagrees with model C={cfg.in_dim}. Re-cache to a single DAC config."

            # forward (bf16 autocast if on cuda and configured)
            if device.type == "cuda" and hp.enforce_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(z_mix)
                    loss, logs, _ = total_separator_loss(
                        out, z_s1, z_s2, mask,
                        lambda_residual_l2=hp.lambda_residual_l2,
                        lambda_mask_entropy=hp.lambda_mask_entropy,
                        # ---- EL knobs ----
                        el_mode=hp.el_mode,
                        lambda_el=hp.lambda_el,
                        el_cosine=hp.el_cosine,
                        latent_teacher=latent_teacher_fn,
                        decode_fn=decode_fn,
                        embed_fn=embed_fn,
                    )
            else:
                out = model(z_mix)
                loss, logs, _ = total_separator_loss(
                    out, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
                    el_mode=hp.el_mode,
                    lambda_el=hp.lambda_el,
                    el_cosine=hp.el_cosine,
                    latent_teacher=latent_teacher_fn,
                    decode_fn=decode_fn,
                    embed_fn=embed_fn,
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step()
            ema.update(model)

            # LR schedule
            lr_now = cosine_lr(global_step, total_steps, hp.lr, hp.cosine_min_lr_ratio)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            if global_step % hp.log_every == 0:
                print(
                    f"step {global_step:6d} | lr {lr_now:.2e} | "
                    f"loss {logs['loss/total']:.4f} | pit {logs['loss/pit_latent']:.4f} "
                    f"| L12 {logs['L_12']:.4f} L21 {logs['L_21']:.4f} | EL {logs['loss/EL']:.4f}"
                )

            if global_step % hp.val_every_steps == 0 and global_step > 0:
                val_loss = evaluate(model, val_loader, device)
                # EMA eval
                ema.store(model)
                ema.copy_to(model)
                val_loss_ema = evaluate(model, val_loader, device)
                model.load_state_dict(ema.backup)
                # save
                torch.save(
                    {
                        "step": global_step,
                        "model": model.state_dict(),
                        "ema": ema.shadow,
                        "hp": asdict(hp),
                        "val": float(val_loss),
                        "val_ema": float(val_loss_ema),
                    },
                    Path(hp.ckpt_dir) / f"ckpt_step{global_step}_val{val_loss:.4f}_valEMA{val_loss_ema:.4f}.pt",
                )
                print(f"[ckpt] step {global_step} saved")
            global_step += 1

    # final save (EMA)
    ema.store(model)
    ema.copy_to(model)
    torch.save(
        {"step": global_step, "model": model.state_dict(), "ema": ema.shadow, "hp": asdict(hp)},
        Path(hp.ckpt_dir) / "final_ema.pt",
    )
    print("Training complete.")

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    use_bf16 = (device.type == "cuda")

    for batch in loader:
        z_mix = batch["z_mix"].to(device, non_blocking=True)
        z_s1  = batch["z_s1"].to(device, non_blocking=True)
        z_s2  = batch["z_s2"].to(device, non_blocking=True)
        mask  = batch["mask"].to(device, non_blocking=True)

        if use_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(z_mix)
                loss, logs, _ = total_separator_loss(
                    out, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
                    el_mode=hp.el_mode,
                    lambda_el=hp.lambda_el,
                    el_cosine=hp.el_cosine,
                )
        else:
            out = model(z_mix)
            loss, logs, _ = total_separator_loss(
                out, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy,
                el_mode=hp.el_mode,
                lambda_el=hp.lambda_el,
                el_cosine=hp.el_cosine,
            )

        losses.append(float(logs["loss/total"]))

    return sum(losses) / max(1, len(losses))

if __name__ == "__main__":
    # keep explicit for clarity (also set at top)
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    main()
