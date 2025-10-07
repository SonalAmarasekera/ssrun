import argparse
import datetime
import pathlib
from tqdm.auto import tqdm

import dac
import torch
from torch.utils.data import DataLoader

from libri2mix_ds import Libri2MixDataset
from pit_latent_mse import pit_mse
from separator_rwkv6 import RWKV6Separator
from utils.collate import collate_latent_batch
from utils.lr_sched import CosineWarmup
from utils.seed import seed_everything

ENC_HOP = 320  # Descript 16‑kHz codec hop

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def load_cfg(path: str):
    import yaml; return yaml.safe_load(open(path))


def slice_latent(z_file: str, frame0: int, n_frames: int, device):
    z = torch.load(z_file)["z"]        # [1,C,F] or [C,F]
    if z.dim() == 3: z = z.squeeze(0)
    end = frame0 + n_frames
    if end <= z.shape[-1]:
        out = z[:, frame0:end]
    else:                               # right‑pad
        pad = torch.zeros(z.shape[0], end - z.shape[-1], dtype=z.dtype, device=z.device)
        out = torch.cat([z[:, frame0:], pad], dim=-1)
    return out.to(device)


def evaluate(model, loader, device):
    model.eval(); acc = 0.0; n = 0
    with torch.no_grad():
        for mix, lat_paths, starts in loader:
            mix = mix.to(device)
            f_starts = (starts // ENC_HOP).tolist()
            pred,_ = model(mix); T = pred[0].shape[-1]
            tgt1 = torch.stack([slice_latent(p1, f0, T, device) for (p1,_),f0 in zip(lat_paths,f_starts)])
            tgt2 = torch.stack([slice_latent(p2, f0, T, device) for (_,p2),f0 in zip(lat_paths,f_starts)])
            loss = pit_mse(pred, tgt1, tgt2).clamp(min=1e-12)
            acc += (-10*torch.log10(loss)).item(); n+=1
    model.train(); return acc/n


def save_ckpt(path, model, step, epoch):
    torch.save({"model": model.state_dict(), "step": step, "epoch": epoch}, path)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main(cfg, resume_path=""):
    seed_everything(cfg.get("seed", 42))

    train_ds = Libri2MixDataset(cfg["train_csv"], segment=cfg["seg_sec"], cache_latents=True)
    dev_ds   = Libri2MixDataset(cfg["dev_csv"],   segment=cfg["seg_sec"], cache_latents=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True,
                              collate_fn=collate_latent_batch)
    dev_loader   = DataLoader(dev_ds, batch_size=1, collate_fn=collate_latent_batch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec  = dac.DAC.load(dac.utils.download("16khz")).to(device).eval()
    model  = RWKV6Separator(codec, depth=cfg["depth"]).to(device)

    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    total = cfg["epochs"]*len(train_loader)//cfg["accum_steps"]
    sched = CosineWarmup(opt, warmup=cfg["warm_steps"], max_steps=total, min_lr=cfg.get("min_lr",1e-5))
    scaler= torch.amp.GradScaler(device.type, enabled=cfg["amp"])

    run_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tb  = torch.utils.tensorboard.SummaryWriter(pathlib.Path("runs")/run_name)
    ckpt_dir = pathlib.Path("checkpoints")/run_name; ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -------- optional resume --------
    step = 0; start_epoch = 0; best = -1e9
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        step        = ckpt.get("step",0)
        start_epoch = ckpt.get("epoch",0)+1
        print(f"✓ Resumed from {resume_path} (epoch {start_epoch})")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train(); inner = tqdm(train_loader, desc=f"Epoch {epoch}", ncols=90)
        for mix, lat_paths, starts in inner:
            mix = mix.to(device)
            f_starts = (starts // ENC_HOP).tolist()
            with torch.amp.autocast(device.type, enabled=cfg["amp"]):
                pred_lat,_ = model(mix)
                T = pred_lat[0].shape[-1]
                pred_lat = [torch.nan_to_num(p, nan=0., posinf=1e4, neginf=-1e4) for p in pred_lat]
            tgt1 = torch.stack([slice_latent(p1, f0, T, device) for (p1,_),f0 in zip(lat_paths,f_starts)])
            tgt2 = torch.stack([slice_latent(p2, f0, T, device) for (_,p2),f0 in zip(lat_paths,f_starts)])

            with torch.amp.autocast(device.type, enabled=cfg["amp"]):
                loss = pit_mse(pred_lat, tgt1, tgt2) / cfg["accum_steps"]

            if not torch.isfinite(loss):
                inner.write(f"⚠️ step {step}: nan/inf loss skipped")
                opt.zero_grad(); model.zero_grad(); continue

            scaler.scale(loss).backward()
            if (step+1)%cfg["accum_steps"]==0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
                tb.add_scalar("train/loss", loss.item()*cfg["accum_steps"], step)

            if step%100==0: inner.set_postfix({"l":f"{loss.item():.3f}"})
            step+=1

        sisdr = evaluate(model, dev_loader, device)
        tb.add_scalar("dev/SI-SDR", sisdr, step)
        if sisdr>best:
            best=sisdr; save_ckpt(ckpt_dir/"best.pt", model, step, epoch)
        save_ckpt(ckpt_dir/f"epoch{epoch:02d}.pt", model, step, epoch)
        print(f"Epoch {epoch} | dev SI-SDR {sisdr:.2f} | best {best:.2f}")

# ------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--cfg", default="configs/train_thesis_model.yaml")
    pa.add_argument("--resume", default="", help="path to checkpoint to resume from")
    args = pa.parse_args(); cfg = load_cfg(args.cfg)
    main(cfg, args.resume)
