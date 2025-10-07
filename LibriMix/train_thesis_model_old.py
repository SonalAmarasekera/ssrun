#!/usr/bin/env python
import argparse, torch, yaml, pathlib, datetime, json
from torch.utils.data import DataLoader
from libri2mix_ds import Libri2MixDataset
from separator_rwkv6 import RWKV6Separator
from pit_latent_mse import pit_mse
from utils.seed import seed_everything
from utils.lr_sched import CosineWarmup  # tiny helper

def main(cfg):
    seed_everything(cfg["seed"])
    # ---------- data ----------
    train_ds = Libri2MixDataset(cfg["train_csv"], segment=cfg["seg_sec"],
                                cache_latents=True)
    dev_ds   = Libri2MixDataset(cfg["dev_csv"],   segment=cfg["seg_sec"],
                                cache_latents=True)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True)
    dev_loader   = DataLoader(dev_ds, batch_size=1, shuffle=False,
                              num_workers=2)
    # ---------- model ----------
    import dac
    codec = dac.DAC.load(dac.utils.download("16khz")).cuda().eval()
    model = RWKV6Separator(codec, depth=cfg["depth"]).cuda()
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = CosineWarmup(opt, warmup=cfg["warm_steps"],
                         max_steps=cfg["epochs"]*len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["amp"])
    # ---------- logging ----------
    run = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path("runs")/run; log_dir.mkdir(parents=True)
    tb = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_dir = pathlib.Path("checkpoints")/run; ckpt_dir.mkdir(parents=True)
    # ---------- train loop ----------
    step = 0; best_sisdr = -1e9
    for epoch in range(cfg["epochs"]):
        model.train()
        for b, (mix, srcs, lat_paths) in enumerate(train_loader):
            mix = mix.cuda().unsqueeze(1)         # [B,1,T]
            tgt_lat = [torch.load(p)["z"].cuda() for p in lat_paths[0]]
            with torch.cuda.amp.autocast(enabled=cfg["amp"]):
                pred_lat, _ = model(mix)
                loss = pit_mse(pred_lat, *tgt_lat)/cfg["accum_steps"]
            scaler.scale(loss).backward()
            if (b+1)%cfg["accum_steps"]==0:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(
                     model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); opt.zero_grad()
                sched.step(); step+=1
                tb.add_scalar("train/loss", loss.item()*cfg["accum_steps"], step)
        # ---------- dev eval ----------
        if epoch%cfg["eval_every"]==0:
            sisdr = eval_dev(model, dev_loader)   # small helper defined below
            tb.add_scalar("dev/SI-SDR", sisdr, step)
            if sisdr>best_sisdr:
                best_sisdr=sisdr
                torch.save({"step":step,"epoch":epoch,"model":model.state_dict()},
                           ckpt_dir/"best.pt")
        torch.save({"step":step,"epoch":epoch,"model":model.state_dict()},
                   ckpt_dir/f"epoch{epoch:02d}.pt")
        print(f"E{epoch} done  dev SI-SDR={sisdr:.2f}  best={best_sisdr:.2f}")
    print("Training finished")

def eval_dev(model, loader):
    model.eval(); tot=0; cnt=0
    with torch.no_grad():
        for mix, srcs, lat_paths in loader:
            mix=mix.cuda().unsqueeze(1)
            tgt_lat=[torch.load(p)["z"].cuda() for p in lat_paths[0]]
            pred,_=model(mix); loss=pit_mse(pred,*tgt_lat, reduce=False)
            sisdr=-10*torch.log10(loss.mean()).item()
            tot+=sisdr; cnt+=1
    model.train(); return tot/cnt

if __name__=="__main__":
    cfg = yaml.safe_load(open("train_thesis_model.yaml"))
    main(cfg)
