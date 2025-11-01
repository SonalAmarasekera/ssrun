# train_separator.py
from pathlib import Path
import os, math, json, random, time
from dataclasses import dataclass, asdict
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# --- RWKV v7 CUDA settings (must be set before importing RWKV model) ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")
# -----------------------------------------------------------------------

from dataset_latents import RWKVLatentDataset
from collate_latents import collate_rwkv_latents
from rwkv_separator_v7_bi_old import RWKVv7Separator, SeparatorV7Config
from pit_losses import total_separator_loss

# ----------------- ReduceLROnPlateau Scheduler (copied from schedulers.py) -----------------
class ReduceLROnPlateau:
    """Learning rate scheduler which decreases the learning rate if the loss
    function of interest gets stuck on a plateau, or starts to increase.
    The difference from NewBobLRScheduler is that, this one keeps a memory of
    the last step where do not observe improvement, and compares against that
    particular loss value as opposed to the most recent loss.

    Arguments
    ---------
    lr_min : float
        The minimum allowable learning rate.
    factor : float
        Factor with which to reduce the learning rate.
    patience : int
        How many epochs to wait before reducing the learning rate.
    dont_halve_until_epoch : int
        Number of epochs to wait until halving.

    Example
    -------
    >>> from torch.optim import Adam
    >>> from speechbrain.nnet.linear import Linear
    >>> inp_tensor = torch.rand([1, 660, 3])
    >>> model = Linear(n_neurons=10, input_size=3)
    >>> optim = Adam(lr=1.0, params=model.parameters())
    >>> output = model(inp_tensor)
    >>> scheduler = ReduceLROnPlateau(0.25, 0.5, 2, 1)
    >>> curr_lr, next_lr = scheduler(
    ...     [optim], current_epoch=1, current_loss=10.0
    ... )
    >>> curr_lr, next_lr = scheduler(
    ...     [optim], current_epoch=2, current_loss=11.0
    ... )
    >>> curr_lr, next_lr = scheduler(
    ...     [optim], current_epoch=3, current_loss=13.0
    ... )
    >>> curr_lr, next_lr = scheduler(
    ...     [optim], current_epoch=4, current_loss=14.0
    ... )
    >>> next_lr
    0.5
    """

    def __init__(
        self, lr_min=1e-8, factor=0.5, patience=2, dont_halve_until_epoch=5
    ):
        self.lr_min = lr_min
        self.factor = factor
        self.patience = patience
        self.patience_counter = 0
        self.losses = []
        self.dont_halve_until_epoch = dont_halve_until_epoch
        self.anchor = 99999

    def __call__(self, optim_list, current_epoch, current_loss):
        """
        Arguments
        ---------
        optim_list : list of optimizers
            The optimizers to update using this scheduler.
        current_epoch : int
            Number of times the dataset has been iterated.
        current_loss : int
            A number for determining whether to change the learning rate.

        Returns
        -------
        current_lr : float
            The learning rate before the update.
        next_lr : float
            The learning rate after the update.
        """
        for opt in optim_list:
            current_lr = opt.param_groups[0]["lr"]

            if current_epoch <= self.dont_halve_until_epoch:
                next_lr = current_lr
                self.anchor = current_loss
            else:
                if current_loss <= self.anchor:
                    self.patience_counter = 0
                    next_lr = current_lr
                    self.anchor = current_loss
                elif (
                    current_loss > self.anchor
                    and self.patience_counter < self.patience
                ):
                    self.patience_counter = self.patience_counter + 1
                    next_lr = current_lr
                else:
                    next_lr = current_lr * self.factor
                    self.patience_counter = 0

            # impose the lower bound
            next_lr = max(next_lr, self.lr_min)

        # Updating current loss
        self.losses.append(current_loss)

        return current_lr, next_lr

    def save(self, path):
        """Saves the current metrics on the specified path."""
        data = {
            "losses": self.losses,
            "anchor": self.anchor,
            "patience_counter": self.patience_counter,
        }
        torch.save(data, path)

    def load(self, path, end_of_epoch=False):
        """Loads the needed information."""
        del end_of_epoch  # Unused in this class
        data = torch.load(path)
        self.losses = data["losses"]
        self.anchor = data["anchor"]
        self.patience_counter = data["patience_counter"]

# ----------------- Debugging Functions -----------------
def debug_model_forward(model, sample_batch, device):
    """Debug the model's forward pass and outputs"""
    print("\n" + "="*60)
    print("MODEL FORWARD PASS DEBUG")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        # Test forward pass
        z_mix = sample_batch["z_mix"].to(device)
        z_s1 = sample_batch["z_s1"].to(device)
        z_s2 = sample_batch["z_s2"].to(device)
        
        print(f"Input z_mix shape: {z_mix.shape}")
        print(f"Input z_mix range: {z_mix.min():.4f} to {z_mix.max():.4f}")
        print(f"Input z_mix mean: {z_mix.mean():.4f}, std: {z_mix.std():.4f}")
        
        if device.type == "cuda" and hp.enforce_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(z_mix)
        else:
            output = model(z_mix)
        
        # Handle dictionary output - check what keys are available
        print(f"Model output type: {type(output)}")
        
        if isinstance(output, dict):
            print("Model output keys:", list(output.keys()))
            # Look for the main output tensor - common keys in separation models
            main_output = None
            for key in ['z_hat', 'output', 'separated', 'mask', 'embeddings']:
                if key in output:
                    main_output = output[key]
                    print(f"Using output key: '{key}' with shape: {main_output.shape}")
                    break
            
            if main_output is None and len(output) > 0:
                # Use the first value if no common key found
                first_key = list(output.keys())[0]
                main_output = output[first_key]
                print(f"Using first output key: '{first_key}' with shape: {main_output.shape}")
        else:
            main_output = output
            print(f"Model output shape: {main_output.shape}")
        
        if main_output is not None:
            print(f"Main output range: {main_output.min():.4f} to {main_output.max():.4f}")
            print(f"Main output mean: {main_output.mean():.4f}, std: {main_output.std():.4f}")
            
            # Check for NaNs/Infs
            if torch.isnan(main_output).any():
                print("🚨 MODEL OUTPUT CONTAINS NaNs!")
            if torch.isinf(main_output).any():
                print("🚨 MODEL OUTPUT CONTAINS Infs!")
                
            # Check if output is reasonable (not all zeros or constant)
            if main_output.std() < 1e-6:
                print("🚨 MODEL OUTPUT IS CONSTANT (near zero variance)!")
        else:
            print("❌ Could not find main output tensor in model results!")
            
        return output

def debug_loss_computation(model, sample_batch, device):
    """Debug the loss computation"""
    print("\n" + "="*60)
    print("LOSS COMPUTATION DEBUG")
    print("="*60)
    
    model.train()
    z_mix = sample_batch["z_mix"].to(device)
    z_s1 = sample_batch["z_s1"].to(device)
    z_s2 = sample_batch["z_s2"].to(device)
    mask = sample_batch["mask"].to(device)
    
    if device.type == "cuda" and hp.enforce_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(z_mix)
            loss, logs, perm = total_separator_loss(
                output, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy
            )
    else:
        output = model(z_mix)
        loss, logs, perm = total_separator_loss(
            output, z_s1, z_s2, mask,
            lambda_residual_l2=hp.lambda_residual_l2,
            lambda_mask_entropy=hp.lambda_mask_entropy
        )
    
    print(f"Total loss: {loss.item():.4f}")
    for key, value in logs.items():
        print(f"  {key}: {value:.4f}")
    
    if torch.isnan(loss):
        print("🚨 LOSS IS NaN!")
    if torch.isinf(loss):
        print("🚨 LOSS IS Inf!")
        
    return loss, logs

def debug_gradients(model, sample_batch, device):
    """Debug gradient flow"""
    print("\n" + "="*60)
    print("GRADIENT DEBUG")
    print("="*60)
    
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr)
    
    z_mix = sample_batch["z_mix"].to(device)
    z_s1 = sample_batch["z_s1"].to(device)
    z_s2 = sample_batch["z_s2"].to(device)
    mask = sample_batch["mask"].to(device)
    
    # Forward pass
    if device.type == "cuda" and hp.enforce_bf16:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(z_mix)
            loss, logs, perm = total_separator_loss(
                output, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy
            )
    else:
        output = model(z_mix)
        loss, logs, perm = total_separator_loss(
            output, z_s1, z_s2, mask,
            lambda_residual_l2=hp.lambda_residual_l2,
            lambda_mask_entropy=hp.lambda_mask_entropy
        )
    
    # Backward pass
    opt.zero_grad()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    zero_grad_params = 0
    nan_grad_params = 0
    inf_grad_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            
            if grad_norm == 0:
                zero_grad_params += 1
            if torch.isnan(param.grad).any():
                nan_grad_params += 1
            if torch.isinf(param.grad).any():
                inf_grad_params += 1
        else:
            zero_grad_params += 1
    
    print(f"Total gradient norm: {total_grad_norm:.4f}")
    print(f"Parameters with zero gradients: {zero_grad_params}")
    print(f"Parameters with NaN gradients: {nan_grad_params}")
    print(f"Parameters with Inf gradients: {inf_grad_params}")
    
    if zero_grad_params == len(list(model.parameters())):
        print("🚨 ALL GRADIENTS ARE ZERO - MODEL NOT LEARNING!")
    if nan_grad_params > 0:
        print("🚨 SOME GRADIENTS CONTAIN NaNs!")
    if inf_grad_params > 0:
        print("🚨 SOME GRADIENTS CONTAIN Infs!")
    
    return total_grad_norm

def debug_overfit_small_batch(model, device, num_iterations=10):
    """Test if model can overfit a very small batch"""
    print("\n" + "="*60)
    print("OVERFITTING TEST (Small Batch)")
    print("="*60)
    
    # Create a tiny fixed batch
    B, T, C = 2, 100, model.cfg.in_dim
    z_mix = torch.randn(B, T, C, device=device) * 0.1
    z_s1 = torch.randn(B, T, C, device=device) * 0.1
    z_s2 = torch.randn(B, T, C, device=device) * 0.1
    mask = torch.ones(B, T, device=device)
    
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    initial_loss = None
    for i in range(num_iterations):
        opt.zero_grad()
        
        if device.type == "cuda" and hp.enforce_bf16:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(z_mix)
                loss, logs, perm = total_separator_loss(
                    output, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy
                )
        else:
            output = model(z_mix)
            loss, logs, perm = total_separator_loss(
                output, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy
            )
        
        loss.backward()
        opt.step()
        
        if i == 0:
            initial_loss = loss.item()
        
        print(f"Overfit iter {i+1}: loss = {loss.item():.4f}")
    
    final_loss = loss.item()
    improvement = initial_loss - final_loss if initial_loss else 0
    
    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    print(f"Improvement: {improvement:.4f}")
    
    if improvement > 0.1:  # Should see at least 0.1 improvement
        print("✅ Model can learn - overfitting test PASSED")
    else:
        print("❌ Model cannot learn - overfitting test FAILED")
    
    return improvement

# ----------------- Hyperparameters -----------------
@dataclass
class HParams:
    # data
    train_root: str = "/workspace/latents/min/train"
    val_root:   str = "/workspace/latents/min/dev"
    batch_size: int = 20
    num_workers: int = 5
    pin_memory: bool = True
    # segmenting
    seg_seconds: float = 6.0
    latent_fps: float = 50.0
    # model
    layers: int = 16
    head_size_a: int = 64
    hidden_dim: int = 256
    dir_drop_p: float = 0.3
    use_mask: bool = True
    enforce_bf16: bool = True
    # optimization
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    # Remove warmup and cosine parameters since we're using ReduceLROnPlateau
    # warmup_steps: int = 3000
    # cosine_min_lr_ratio: float = 0.1
    # regularizers
    lambda_residual_l2: float = 1e-5
    lambda_mask_entropy: float = 5e-4
    # embedding-loss (EL) knobs
    #el_mode: str = "none"        # options: "none", "latent", or "decoder"
    #lambda_el: float = 0.0       # "0.1-0.3" for <latent and decoder>, "0.0" for <none>
    #el_cosine: bool = True       # cosine vs L2
    #el_emb_dim: int = 128        
    # misc
    seed: int = 123
    #log_every: int = 100
    #val_every_steps: int = 1000
    ckpt_dir: str = "checkpoints"
    ema_decay: float = 0.999
    # LR scheduler parameters
    lr_min: float = 1e-6          # Minimum learning rate
    lr_factor: float = 0.5        # Factor to reduce LR by
    lr_patience: int = 2          # Epochs to wait before reducing LR
    lr_dont_halve_until_epoch: int = 5  # Don't reduce LR until this epoch
    # Debugging
    debug_mode: bool = False       # Enable debugging steps

hp = HParams()

# ----------------- Utility helpers -----------------
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
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    @torch.no_grad()
    def store(self, model): self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def copy_to(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow: v.copy_(self.shadow[k])

def crop_batch_to_seconds(batch, seg_seconds, fps, chunk_len=None):
    """Crop batch to fixed seconds using latent fps; preserves shapes and mask."""
    z_mix, z_s1, z_s2, mask = batch["z_mix"], batch.get("z_s1"), batch.get("z_s2"), batch["mask"]
    B, T, C = z_mix.shape
    frames = int(seg_seconds * float(fps))
    if chunk_len and frames % chunk_len != 0:
        frames = ((frames + chunk_len - 1) // chunk_len) * chunk_len
    if T <= frames:
        return batch
    starts = torch.randint(low=0, high=T - frames + 1, size=(B,), device=z_mix.device)
    idx = torch.arange(frames, device=z_mix.device)[None, :] + starts[:, None]
    idx = idx.clamp(0, T - 1)
    for k in ("z_mix", "z_s1", "z_s2"):
        if k in batch and batch[k] is not None:
            batch[k] = torch.gather(batch[k], 1, idx[..., None].expand(B, frames, C))
    batch["mask"] = torch.ones(B, frames, dtype=mask.dtype, device=mask.device)
    return batch

# ----------------- Teacher setup for EL -----------------
class SimpleLatentTeacher(nn.Module):
    """Light MLP teacher for latent-teacher mode."""
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, z):  # z: [B,T,C]
        return self.net(z)

def setup_teachers(hp, cfg, device):
    """Return teacher handles depending on el_mode."""
    latent_teacher = None
    decode_fn = None
    embed_fn = None

    if hp.el_mode == "latent":
        latent_teacher = SimpleLatentTeacher(cfg.in_dim, hp.el_emb_dim).to(device)
        latent_teacher.eval()
        for p in latent_teacher.parameters():
            p.requires_grad = False

    elif hp.el_mode == "decoder":
        # These should point to your actual DAC decode/embed functions
        from descript_audio_codec import DAC  # example placeholder
        dac = DAC.load("descript-audio-codec-16khz").to(device)
        decode_fn = lambda z: dac.decode(z)
        embed_fn = lambda wav: dac.encode(wav)["z"]
        # freeze DAC
        for p in dac.parameters():
            p.requires_grad = False

    return latent_teacher, decode_fn, embed_fn

# ----------------- Main training -----------------
def main():
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # probe channel dimension
    probe_ds = RWKVLatentDataset(hp.train_root, require_targets=True)
    probe = torch.load(next(p for p in probe_ds.mix_dir.rglob("*.pt")))
    C_meta = int(probe.get("C", 1024))

    # datasets
    train_ds = RWKVLatentDataset(hp.train_root, require_targets=True, expected_C=C_meta)
    val_ds = RWKVLatentDataset(hp.val_root, require_targets=True, expected_C=C_meta)
    collate_fn = partial(collate_rwkv_latents, chunk_len=None)

    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True,
                              num_workers=hp.num_workers, pin_memory=hp.pin_memory,
                              collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False,
                            num_workers=hp.num_workers, pin_memory=hp.pin_memory,
                            collate_fn=collate_fn, drop_last=False)

    # model
    sample = next(iter(train_loader))
    C = sample["z_mix"].shape[-1]
    cfg = SeparatorV7Config(
        in_dim=C,
        layers=hp.layers,
        head_size_a=hp.head_size_a,
        hidden_dim=hp.hidden_dim,
        dir_drop_p=hp.dir_drop_p,
        use_mask=hp.use_mask,
        enforce_bf16=hp.enforce_bf16
    )
    model = RWKVv7Separator(cfg).to(device)
    
    # Store cfg in model for debugging
    model.cfg = cfg

    print(f"Model initialized with:")
    print(f"  - Input dimension: {cfg.in_dim}")
    print(f"  - Layers: {cfg.layers}")
    print(f"  - Hidden dimension: {cfg.hidden_dim}")
    print(f"  - Head size: {cfg.head_size_a}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    os.makedirs(hp.ckpt_dir, exist_ok=True)
    with open(Path(hp.ckpt_dir) / "hparams.json", "w") as f:
        json.dump(asdict(hp), f, indent=2)

    # ================= DEBUGGING STEPS =================
    if hp.debug_mode:
        print("\n" + "="*60)
        print("STARTING DEBUGGING PHASE")
        print("="*60)
        
        # Get a sample batch for debugging
        sample_batch = next(iter(train_loader))
        for k in sample_batch:
            if torch.is_tensor(sample_batch[k]):
                sample_batch[k] = sample_batch[k].to(device)
        
        # Run all debugging steps
        debug_model_forward(model, sample_batch, device)
        debug_loss_computation(model, sample_batch, device)
        debug_gradients(model, sample_batch, device)
        improvement = debug_overfit_small_batch(model, device)
        
        print("\n" + "="*60)
        print("DEBUGGING COMPLETE")
        print("="*60)
        
        # Ask user if they want to continue
        if improvement <= 0.1:
            response = input("Model failed overfitting test. Continue training? (y/n): ")
            if response.lower() != 'y':
                print("Exiting training.")
                return

    # ================= MAIN TRAINING =================
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, betas=hp.betas, weight_decay=hp.weight_decay)
    ema = EMA(model, hp.ema_decay)
    
    # Initialize ReduceLROnPlateau scheduler
    lr_scheduler = ReduceLROnPlateau(
        lr_min=hp.lr_min,
        factor=hp.lr_factor, 
        patience=hp.lr_patience,
        dont_halve_until_epoch=hp.lr_dont_halve_until_epoch
    )

    # teacher setup
    #latent_teacher, decode_fn, embed_fn = setup_teachers(hp, cfg, device)

    # Training
    global_step = 0
    steps_per_epoch = len(train_loader)
    total_steps = hp.epochs * steps_per_epoch
    scaler = None  # bf16 autocast path uses no scaler

    for epoch in range(hp.epochs):
        model.train()
        # running averages for the epoch
        run_loss = 0.0
        run_pit  = 0.0
        prog = tqdm(train_loader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{hp.epochs}", leave=True)

        for step_in_epoch, batch in enumerate(prog):
            # Move to device & crop
            for k in ("z_mix","z_s1","z_s2","mask","sr"):
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)

            # NOTE: if you're padding to CHUNK_LEN inside the model, keep crop chunk_len=None here
            batch = crop_batch_to_seconds(batch, hp.seg_seconds, hp.latent_fps, chunk_len=None)

            z_mix, z_s1, z_s2, mask = batch["z_mix"], batch["z_s1"], batch["z_s2"], batch["mask"]
            B,T,C = z_mix.shape

            # forward
            if device.type == "cuda" and hp.enforce_bf16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(z_mix)
                    loss, logs, perm = total_separator_loss(
                        out, z_s1, z_s2, mask,
                        lambda_residual_l2=hp.lambda_residual_l2,
                        lambda_mask_entropy=hp.lambda_mask_entropy
                    )
            else:
                out = model(z_mix)
                loss, logs, perm = total_separator_loss(
                    out, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy
                )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
            opt.step()
            ema.update(model)

            # REMOVED: Step-wise LR scheduling with cosine_lr
            # We now use epoch-based ReduceLROnPlateau at the end of each epoch

            # update running meters & progress bar every step
            run_loss += float(logs["loss/total"])
            run_pit  += float(logs["loss/pit_latent"])
            avg_loss = run_loss / (step_in_epoch + 1)
            avg_pit  = run_pit  / (step_in_epoch + 1)
            
            # Show current LR in progress bar
            current_lr = opt.param_groups[0]["lr"]
            prog.set_postfix(lr=f"{current_lr:.2e}", loss=f"{avg_loss:.4f}", pit=f"{avg_pit:.4f}")

            global_step += 1

        # ----- end of epoch: print summary, run full validation, update LR scheduler -----
        epoch_train_loss = run_loss / max(1, steps_per_epoch)
        epoch_train_pit  = run_pit  / max(1, steps_per_epoch)
        print(f"[epoch {epoch+1}] train_loss {epoch_train_loss:.4f} | train_pit {epoch_train_pit:.4f}")

        # Standard validation
        val_loss = evaluate(model, val_loader, device)

        # Update learning rate based on validation performance
        old_lr, new_lr = lr_scheduler([opt], epoch + 1, val_loss)
        # ✅ Actually apply the new learning rate
        for param_group in opt.param_groups:
            param_group["lr"] = new_lr
        if old_lr != new_lr:
            print(f"LR changed from {old_lr:.2e} to {new_lr:.2e} based on val_loss {val_loss:.4f}")

        # EMA validation
        ema.store(model)
        ema.copy_to(model)
        val_loss_ema = evaluate(model, val_loader, device)
        model.load_state_dict(ema.backup)

        # Save epoch checkpoint
        ckpt_path = Path(hp.ckpt_dir) / f"epoch{epoch+1:03d}_val{val_loss:.4f}_valEMA{val_loss_ema:.4f}.pt"
        torch.save(
            {
                "epoch": epoch + 1, 
                "step": global_step, 
                "model": model.state_dict(),
                "ema": ema.shadow, 
                "hp": asdict(hp),
                "lr_scheduler": {
                    "losses": lr_scheduler.losses,
                    "anchor": lr_scheduler.anchor,
                    "patience_counter": lr_scheduler.patience_counter
                }
            },
            ckpt_path
        )
        print(f"[ckpt] saved {ckpt_path}")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    for batch in loader:
        z_mix = batch["z_mix"].to(device)
        z_s1  = batch["z_s1"].to(device)
        z_s2  = batch["z_s2"].to(device)
        mask  = batch["mask"].to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device.type=="cuda")):
            out = model(z_mix)
            loss, logs, _ = total_separator_loss(
                out, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy,
                #el_mode=hp.el_mode,
                #lambda_el=hp.lambda_el,
                #el_cosine=hp.el_cosine,
                #latent_teacher=latent_teacher,
                #decode_fn=decode_fn,
                #embed_fn=embed_fn,
            )
        losses.append(float(logs["loss/total"]))
    return sum(losses) / max(1, len(losses))

if __name__ == "__main__":
    main()
