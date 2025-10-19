# train_separator.py
from pathlib import Path
import os, math, time, json, random
from dataclasses import dataclass, asdict

# ---- RWKV v7 CUDA JIT: must be set BEFORE importing rwkv_orig_model ----
import os
os.environ.setdefault("RWKV_JIT_ON", "1")         # build the CUDA kernel at import
os.environ.setdefault("RWKV_CUDA_ON", "1")        # ensure CUDA path
os.environ.setdefault("RWKV_MY_TESTING", "x070")  # select v7 TimeMix/ChannelMix
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")  # v7 prefers bf16 activations
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")   # keep in sync with your config (head_size_a)
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_latents import RWKVLatentDataset
from collate_latents import collate_rwkv_latents
from rwkv_separator_v7_bi import RWKVv7Separator, SeparatorV7Config
from pit_losses import total_separator_loss
from functools import partial

CHUNK_LEN = 16  # must match the kernel’s constant

# ----------------- hyperparameters -----------------
@dataclass
class HParams:
    # data
    train_root: str = "/content/latents/min/train"
    val_root:   str = "/content/latents/min/test"
    batch_size: int = 1
    num_workers: int = 2
    pin_memory: bool = True
    # segmenting (crop to fixed seconds during train for speed)
    seg_seconds: float = 6.0   # cropped length in seconds worth of latents (approx by fps)
    latent_fps: float = 50.0   # <-- SET THIS to your DAC latent frame rate (e.g., 50 or 75)
    # model
    layers: int = 16
    head_size_a: int = 64
    hidden_dim: int | None = None     # auto (C//2 rounded to multiple of head_size_a) when None
    dir_drop_p: float = 0.5           # train bi with direction dropout
    use_mask: bool = True
    enforce_bf16: bool = True
    # optimization
    epochs: int = 1
    lr: float = 8e-4
    weight_decay: float = 1e-2
    betas: tuple[float,float] = (0.9, 0.95)
    grad_clip: float = 5.0
    warmup_steps: int = 2000
    cosine_min_lr_ratio: float = 0.1
    # regularizers
    lambda_residual_l2: float = 1e-4
    lambda_mask_entropy: float = 0.0
    # misc
    seed: int = 123
    log_every: int = 100
    val_every_steps: int = 1000
    ckpt_dir: str = "checkpoints"
    ema_decay: float = 0.999

hp = HParams()

# ----------------- helpers -----------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.is_floating_point()}
    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.is_floating_point():
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def store(self, model): self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def copy_to(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow: v.copy_(self.shadow[k])

def cosine_lr(step, total, base_lr, min_ratio=0.1):
    if step < hp.warmup_steps:
        return base_lr * (step+1) / hp.warmup_steps
    t = (step - hp.warmup_steps) / max(1, total - hp.warmup_steps)
    return base_lr * (min_ratio + 0.5*(1-min_ratio)*(1 + math.cos(math.pi * (1-t))))

def crop_batch_to_seconds(batch, seg_seconds: float, fps: float, chunk_len: int):
    """Crop each sample to ~seg_seconds using a known latent fps; preserves [B,T,C] and mask.
       Frames are rounded up to a multiple of chunk_len to satisfy the kernel."""
    z_mix, z_s1, z_s2, mask = batch["z_mix"], batch.get("z_s1"), batch.get("z_s2"), batch["mask"] 
    B, T, C = z_mix.shape
    frames = int(seg_seconds * float(fps))
    if chunk_len and frames % chunk_len != 0:
        frames = ((frames + chunk_len - 1) // chunk_len) * chunk_len
    if T <= frames:  # no crop
        return batch
    # choose per-sample random start within valid range
    starts = torch.randint(low=0, high=T-frames+1, size=(B,), device=z_mix.device)
    idx = torch.arange(frames, device=z_mix.device)[None, :] + starts[:, None]  # [B,frames]
    idx = idx.clamp(0, T-1)
    batch["z_mix"] = torch.gather(z_mix, 1, idx[..., None].expand(B, frames, C))
    if z_s1 is not None: batch["z_s1"] = torch.gather(z_s1, 1, idx[..., None].expand(B, frames, C))
    if z_s2 is not None: batch["z_s2"] = torch.gather(z_s2, 1, idx[..., None].expand(B, frames, C))
    batch["mask"]  = torch.ones(B, frames, dtype=mask.dtype, device=mask.device)  # fully valid after crop
    return batch

def normalize_latent(z):
    mean = z.mean(dim=(1, 2), keepdim=True)
    std = z.std(dim=(1, 2), keepdim=True)
    return (z - mean) / (std + 1e-5)

# ----------------- main train -----------------
def main():
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    probe_ds = RWKVLatentDataset(hp.train_root, require_targets=True)
    probe = torch.load(next((hp for hp in (p for p in (probe_ds.mix_dir.rglob("*.pt")))), None))
    C_meta = 512
    #C_meta = int(probe.get("C", 1024))  # fall back if older cache

    # Data
    train_ds = RWKVLatentDataset(hp.train_root, require_targets=True, expected_C=512)
    val_ds   = RWKVLatentDataset(hp.val_root,   require_targets=True, expected_C=512)

    collate_fn = partial(collate_rwkv_latents, chunk_len=CHUNK_LEN)

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                              num_workers=hp.num_workers, pin_memory=hp.pin_memory,
                              collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False,
                              num_workers=hp.num_workers, pin_memory=hp.pin_memory,
                              collate_fn=collate_fn, drop_last=False)

    # Model (RWKV-v7 Bi core)
    # infer C from a sample
    sample = next(iter(train_loader))
    C = sample["z_mix"].shape[-1]
    cfg = SeparatorV7Config(
        in_dim=C, layers=hp.layers, head_size_a=hp.head_size_a,
        hidden_dim=hp.hidden_dim, dir_drop_p=hp.dir_drop_p,
        use_mask=hp.use_mask, enforce_bf16=hp.enforce_bf16
    )
    model = RWKVv7Separator(cfg).to(device)
    os.makedirs(hp.ckpt_dir, exist_ok=True)
    with open(Path(hp.ckpt_dir) / "hparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    # Optim
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, betas=hp.betas, weight_decay=hp.weight_decay)

    # EMA
    ema = EMA(model, hp.ema_decay)

    # Training
    global_step = 0
    total_steps = hp.epochs * (len(train_loader))
    scaler = None  # we run bf16 autocast (no scaler) by default

    for epoch in range(hp.epochs):
        model.train()
        for batch in train_loader:
            # Move to device & crop
            for k in ("z_mix","z_s1","z_s2","mask","sr"):
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)
            batch = crop_batch_to_seconds(batch, hp.seg_seconds, hp.latent_fps, CHUNK_LEN)

            z_mix, z_s1, z_s2, mask = batch["z_mix"], batch["z_s1"], batch["z_s2"], batch["mask"]
            # Assertions to keep contract honest
            B,T,C = z_mix.shape
            assert z_s1.shape == (B,T,C) and z_s2.shape == (B,T,C)
            assert mask.shape == (B,T)

            z_mix = normalize_latent(z_mix)
            z_s1 = normalize_latent(z_s1)
            z_s2 = normalize_latent(z_s2)

            # forward (bf16 autocast if on cuda and configured)
            if device.type == "cuda" and hp.enforce_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(z_mix)
                    loss, logs, perm = total_separator_loss(out, z_s1, z_s2, mask,
                                                            lambda_residual_l2=hp.lambda_residual_l2,
                                                            lambda_mask_entropy=hp.lambda_mask_entropy)
            else:
                out = model(z_mix)
                loss, logs, perm = total_separator_loss(out, z_s1, z_s2, mask,
                                                        lambda_residual_l2=hp.lambda_residual_l2,
                                                        lambda_mask_entropy=hp.lambda_mask_entropy)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step()
            ema.update(model)

            # LR schedule
            lr_now = cosine_lr(global_step, total_steps, hp.lr, hp.cosine_min_lr_ratio)
            for pg in opt.param_groups: pg["lr"] = lr_now

            if global_step % hp.log_every == 0:
                print(f"step {global_step:6d} | lr {lr_now:.2e} | "
                      f"loss {logs['loss/total']:.4f} | pit {logs['loss/pit_latent']:.4f} "
                      f"| L12 {logs['L_12']:.4f} L21 {logs['L_21']:.4f}")

            if global_step % hp.val_every_steps == 0 and global_step > 0:
                val_loss = evaluate(model, val_loader, device)
                ema.store(model); ema.copy_to(model)
                val_loss_ema = evaluate(model, val_loader, device)
                model.load_state_dict(ema.backup)
                # save
                torch.save({"step": global_step, "model": model.state_dict(),
                            "ema": ema.shadow, "hp": asdict(hp)},
                           Path(hp.ckpt_dir) / f"ckpt_step{global_step}_val{val_loss:.4f}_valEMA{val_loss_ema:.4f}.pt")
                print(f"[ckpt] step {global_step} saved")
            global_step += 1

    # final save
    ema.store(model); ema.copy_to(model)
    torch.save({"step": global_step, "model": model.state_dict(),
                "ema": ema.shadow, "hp": asdict(hp)},
               Path(hp.ckpt_dir) / "final_ema.pt")
    print("Training complete.")

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    use_bf16 = (device.type == "cuda")

    for batch in loader:
        # ✅ Unpack the batch (this line was missing)
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
                    lambda_mask_entropy=hp.lambda_mask_entropy
                )
        else:
            out = model(z_mix)
            loss, logs, _ = total_separator_loss(
                out, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy
            )

        losses.append(float(logs["loss/total"]))

    return sum(losses) / max(1, len(losses))


if __name__ == "__main__":
    # Make sure CUDA path uses x070 kernels inside the imported model
    os.environ.setdefault("RWKV_MY_TESTING", "x070")
    main()
