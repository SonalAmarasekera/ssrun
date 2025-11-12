# train_separator_fixed_withLR.py
"""
FIXED training script for RWKV-based speech separator.
Major fixes:
1. Proper model output format handling
2. Dataset normalization
3. Correct config parameters
4. Better gradient monitoring
5. Improved learning rate scheduling
6. **NEW: Embedding Loss (CodecFormer-EL style) integration**
"""

from pathlib import Path
import os, math, json, random, time
from dataclasses import dataclass, asdict
from functools import partial
from typing import Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm.auto import tqdm

# --- RWKV v7 CUDA settings (must be set before importing RWKV model) ---
os.environ.setdefault("RWKV_JIT_ON", "1")
os.environ.setdefault("RWKV_CUDA_ON", "1")
os.environ.setdefault("RWKV_MY_TESTING", "x070")
os.environ.setdefault("RWKV_FLOAT_MODE", "bf16")
os.environ.setdefault("RWKV_HEAD_SIZE_A", "64")

# Import fixed dataset with normalization
from dataset_latents_fixed import RWKVLatentDataset
from collate_latents import collate_rwkv_latents

# Import model and wrapper
from rwkv_separator_DSmod import RWKVv7Separator, SeparatorV7Config, build_rwkv7_separator
from rwkv_separator_wrapper import RWKVSeparatorWrapper

# Import loss functions
from pit_losses import total_separator_loss


@dataclass
class FixedHParams:
    """Fixed hyperparameters with better defaults + Embedding Loss support."""
    
    # Data paths
    train_data: str = "/workspace/latents/min/train/"
    val_data: str = "/workspace/latents/min/dev/"
    ckpt_dir: str = "/workspace/checkpoints"
    
    # Model architecture
    n_layer: int = 8  # Changed from 'layers'
    n_embd: Optional[int] = None  # Auto-detect from data
    head_size_a: int = 64
    head_hidden: int = 256  # Changed from 'hidden_dim'
    num_sources: int = 2
    head_mode: str = "residual"  # or "mask"
    
    # Training parameters
    batch_size: int = 4  # Start small
    epochs: int = 100
    lr: float = 5e-5  # Lower initial LR
    lr_warmup_steps: int = 1000
    lr_min: float = 1e-7
    weight_decay: float = 0.01
    grad_clip: float = 0.5  # More aggressive clipping
    
    # Loss configuration
    lambda_residual_l2: float = 1e-4
    lambda_mask_entropy: float = 0.0
    mse_scale: float = 0.1  # Scale MSE for stability
    
    # ============================================
    # NEW: Embedding Loss (CodecFormer-EL) Parameters
    # ============================================
    use_embedding_loss: bool = False  # Enable/disable embedding loss
    el_mode: str = "latent"  # 'none' | 'latent' | 'decoder'
    lambda_el: float = 0.1  # Weight for embedding loss (0.0 to disable)
    el_cosine: bool = True  # Use cosine similarity (True) or MSE (False)
    # Note: latent_teacher, decode_fn, embed_fn are set to None by default
    # The 'latent' mode with latent_teacher=None uses identity projection (fallback)
    # ============================================
    
    # Data processing
    seg_seconds: Optional[float] = 3.0  # Process 3-second segments
    latent_fps: float = 50.0  # For 16kHz DAC
    chunk_len: Optional[int] = None
    
    # Normalization (CRITICAL)
    normalize_latents: bool = True
    force_recompute_stats: bool = False
    
    # Training options
    enforce_bf16: bool = False  # Start with FP32 for debugging
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Logging
    log_every_n_steps: int = 10
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    tensorboard_dir: str = "runs/rwkv_separator"
    save_all_checkpoints: bool = True  # Save checkpoint after every epoch
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint to resume from
    
    # Learning rate scheduler
    use_lr_scheduler: bool = True  # Enable LR scheduling
    lr_scheduler_type: str = "cosine"  # "cosine", "step", or "plateau"
    
    # Debug mode
    debug_mode: bool = True
    debug_overfit_batches: int = 10
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class GradientMonitor:
    """Monitor gradient statistics during training."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.grad_norms = []
        self.param_norms = []
        self.update_ratios = []
    
    def log_gradients(self, model, lr: float):
        """Log gradient statistics."""
        total_norm = 0
        param_norm = 0
        update_norm = 0
        num_params = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm += p.data.norm(2).item() ** 2
                grad_norm = p.grad.data.norm(2).item() ** 2
                total_norm += grad_norm
                
                # Approximate update magnitude
                update = lr * p.grad.data
                update_norm += update.norm(2).item() ** 2
                num_params += 1
        
        if num_params > 0:
            total_norm = total_norm ** 0.5
            param_norm = param_norm ** 0.5
            update_norm = update_norm ** 0.5
            
            self.grad_norms.append(total_norm)
            self.param_norms.append(param_norm)
            
            # Update ratio: how much parameters change relative to their magnitude
            if param_norm > 0:
                ratio = update_norm / param_norm
                self.update_ratios.append(ratio)
                return total_norm, ratio
        
        return 0.0, 0.0
    
    def get_stats(self):
        """Get current statistics."""
        if not self.grad_norms:
            return {"grad_norm": 0, "update_ratio": 0}
        
        return {
            "grad_norm": self.grad_norms[-1],
            "grad_norm_avg": sum(self.grad_norms) / len(self.grad_norms),
            "update_ratio": self.update_ratios[-1] if self.update_ratios else 0,
        }


def create_datasets(hp: FixedHParams):
    """Create datasets with normalization."""
    
    print("\n" + "="*60)
    print("Creating datasets with normalization...")
    print("="*60)
    
    # Training dataset with normalization
    train_dataset = RWKVLatentDataset(
        root=hp.train_data,
        require_targets=True,
        expected_C=hp.n_embd,  # Will auto-detect if None
        normalize_latents=hp.normalize_latents,
        norm_stats_path=f"{hp.train_data}/latent_stats.json",
        force_recompute_stats=hp.force_recompute_stats,
    )
    
    # Get actual channel count
    if hp.n_embd is None:
        hp.n_embd = train_dataset.expected_C
        print(f"Auto-detected latent channels: C={hp.n_embd}")
    
    # Validation dataset (uses same normalization stats!)
    val_dataset = RWKVLatentDataset(
        root=hp.val_data,
        require_targets=True,
        expected_C=hp.n_embd,
        normalize_latents=hp.normalize_latents,
        norm_stats_path=f"{hp.train_data}/latent_stats.json",  # Use train stats
        force_recompute_stats=False,  # Never recompute for val
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Normalization: {'ENABLED' if hp.normalize_latents else 'DISABLED'}")
    
    if hp.normalize_latents:
        norm_params = train_dataset.get_normalization_params()
        print(f"Norm stats - mean: {norm_params['mean']:.4f}, std: {norm_params['std']:.4f}")
    
    return train_dataset, val_dataset


def create_model(hp: FixedHParams):
    """Create model with proper configuration."""
    
    print("\n" + "="*60)
    print("Creating RWKV separator model...")
    print("="*60)
    
    # Create config with FIXED parameter names
    cfg = SeparatorV7Config(
        n_embd=hp.n_embd,  # Fixed from 'in_dim'
        n_layer=hp.n_layer,  # Fixed from 'layers'
        head_size_a=hp.head_size_a,
        enforce_bf16=hp.enforce_bf16,
        num_sources=hp.num_sources,
        head_hidden=hp.head_hidden,
        head_mode=hp.head_mode,
    )
    
    # Build model
    model = RWKVv7Separator(cfg)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Head mode: {hp.head_mode}")
    
    return model


def test_model_output(model, hp: FixedHParams):
    """Test model output format."""
    
    print("\n" + "="*60)
    print("Testing model output format...")
    print("="*60)
    
    # Create dummy input [B,T,C]
    B, T, C = 2, 160, hp.n_embd  # T must be multiple of 16
    x = torch.randn(B, T, C).to(hp.device)
    
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # Check if output is a dictionary with expected keys
    if isinstance(output, dict):
        print(f"Output keys: {list(output.keys())}")
        required_keys = [f'pred{i+1}' for i in range(hp.num_sources)]
        
        for key in required_keys:
            if key in output:
                print(f"  ✓ {key}: shape={output[key].shape}, dtype={output[key].dtype}")
            else:
                print(f"  ✗ {key}: MISSING!")
                return False
        
        # Test with loss function
        targets = torch.randn_like(output['pred1'])
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool).to(hp.device)
        
        try:
            from pit_losses import pit_latent_mse_2spk
            loss, perm, extras = pit_latent_mse_2spk(
                output, targets, targets, mask
            )
            print(f"\nLoss test: SUCCESS (loss={loss.item():.4f})")
            return True
        except Exception as e:
            print(f"\nLoss test: FAILED - {e}")
            return False
    else:
        print(f"ERROR: Output is not a dictionary! Got {type(output)}")
        return False


def train_one_epoch(
    model, 
    train_loader, 
    optimizer, 
    hp: FixedHParams,
    epoch: int,
    grad_monitor: GradientMonitor,
    writer: Optional[SummaryWriter] = None,
):
    """Train for one epoch with proper monitoring."""
    
    model.train()
    device = hp.device
    
    running_loss = 0.0
    running_pit = 0.0
    running_el = 0.0  # NEW: Track embedding loss
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{hp.epochs}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        z_mix = batch["z_mix"].to(device, non_blocking=True)
        z_s1 = batch["z_s1"].to(device, non_blocking=True)
        z_s2 = batch["z_s2"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        
        B, T, C = z_mix.shape
        
        # Check input statistics (for debugging)
        if batch_idx == 0 and epoch == 0:
            print(f"\nFirst batch stats:")
            print(f"  z_mix: mean={z_mix.mean():.4f}, std={z_mix.std():.4f}")
            print(f"  range: [{z_mix.min():.2f}, {z_mix.max():.2f}]")
            
            # NEW: Print embedding loss settings
            if hp.use_embedding_loss:
                print(f"\n🔥 Embedding Loss ENABLED:")
                print(f"  el_mode: {hp.el_mode}")
                print(f"  lambda_el: {hp.lambda_el}")
                print(f"  el_cosine: {hp.el_cosine}")
            else:
                print(f"\n⚠️  Embedding Loss DISABLED")
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        
        if hp.enforce_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(z_mix)
                
                # ============================================
                # NEW: Pass embedding loss parameters
                # ============================================
                loss, logs, perm = total_separator_loss(
                    output, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
                    # Embedding loss parameters
                    el_mode=hp.el_mode if hp.use_embedding_loss else "none",
                    lambda_el=hp.lambda_el if hp.use_embedding_loss else 0.0,
                    el_cosine=hp.el_cosine,
                    latent_teacher=None,  # Use fallback (identity projection)
                    decode_fn=None,
                    embed_fn=None,
                )
        else:
            output = model(z_mix)
            
            # Scale loss for numerical stability
            if hp.mse_scale != 1.0:
                # Scale targets for MSE computation
                z_s1_scaled = z_s1 * hp.mse_scale
                z_s2_scaled = z_s2 * hp.mse_scale
                output_scaled = {k: v * hp.mse_scale if 'pred' in k else v 
                               for k, v in output.items()}
                
                # ============================================
                # NEW: Pass embedding loss parameters
                # ============================================
                loss, logs, perm = total_separator_loss(
                    output_scaled, z_s1_scaled, z_s2_scaled, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
                    # Embedding loss parameters
                    el_mode=hp.el_mode if hp.use_embedding_loss else "none",
                    lambda_el=hp.lambda_el if hp.use_embedding_loss else 0.0,
                    el_cosine=hp.el_cosine,
                    latent_teacher=None,
                    decode_fn=None,
                    embed_fn=None,
                )
            else:
                # ============================================
                # NEW: Pass embedding loss parameters
                # ============================================
                loss, logs, perm = total_separator_loss(
                    output, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
                    # Embedding loss parameters
                    el_mode=hp.el_mode if hp.use_embedding_loss else "none",
                    lambda_el=hp.lambda_el if hp.use_embedding_loss else 0.0,
                    el_cosine=hp.el_cosine,
                    latent_teacher=None,
                    decode_fn=None,
                    embed_fn=None,
                )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if hp.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
        
        # Log gradients
        grad_norm, update_ratio = grad_monitor.log_gradients(model, hp.lr)
        
        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"\nWARNING: Non-finite loss at batch {batch_idx}!")
            print(f"  Loss: {loss.item()}")
            print(f"  z_mix: min={z_mix.min():.2f}, max={z_mix.max():.2f}")
            continue
        
        # Optimizer step
        optimizer.step()
        
        # Update running averages
        running_loss += loss.item()
        running_pit += logs["loss/pit_latent"].item()
        running_el += logs["loss/EL"].item()  # NEW: Track EL
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        avg_pit = running_pit / (batch_idx + 1)
        avg_el = running_el / (batch_idx + 1)  # NEW
        
        # NEW: Updated progress bar with EL
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'pit': f'{avg_pit:.4f}',
            'EL': f'{avg_el:.4f}',  # NEW
            'grad': f'{grad_norm:.2f}'
        })
        
        # Detailed logging
        if (batch_idx + 1) % hp.log_every_n_steps == 0 and writer is not None:
            global_step = epoch * num_batches + batch_idx
            
            # Loss components
            writer.add_scalar('Train/loss_total', loss.item(), global_step)
            writer.add_scalar('Train/loss_pit', logs["loss/pit_latent"].item(), global_step)
            writer.add_scalar('Train/loss_reg', logs["loss/reg_residual"].item(), global_step)
            writer.add_scalar('Train/loss_EL', logs["loss/EL"].item(), global_step)  # NEW
            
            # Gradient statistics
            writer.add_scalar('Train/grad_norm', grad_norm, global_step)
            writer.add_scalar('Train/update_ratio', update_ratio, global_step)
            
            # Learning rate
            writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'], global_step)
    
    return running_loss / num_batches


def evaluate(
    model,
    val_loader,
    hp: FixedHParams,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
):
    """Evaluate on validation set."""
    
    model.eval()
    device = hp.device
    
    total_loss = 0.0
    total_pit = 0.0
    total_el = 0.0  # NEW
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            z_mix = batch["z_mix"].to(device, non_blocking=True)
            z_s1 = batch["z_s1"].to(device, non_blocking=True)
            z_s2 = batch["z_s2"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            
            # Forward pass
            output = model(z_mix)
            
            # ============================================
            # NEW: Pass embedding loss parameters
            # ============================================
            loss, logs, perm = total_separator_loss(
                output, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy,
                # Embedding loss parameters
                el_mode=hp.el_mode if hp.use_embedding_loss else "none",
                lambda_el=hp.lambda_el if hp.use_embedding_loss else 0.0,
                el_cosine=hp.el_cosine,
                latent_teacher=None,
                decode_fn=None,
                embed_fn=None,
            )
            
            total_loss += loss.item()
            total_pit += logs["loss/pit_latent"].item()
            total_el += logs["loss/EL"].item()  # NEW
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_pit = total_pit / num_batches
    avg_el = total_el / num_batches  # NEW
    
    # Log to TensorBoard
    if writer is not None:
        writer.add_scalar('Val/loss_total', avg_loss, epoch)
        writer.add_scalar('Val/loss_pit', avg_pit, epoch)
        writer.add_scalar('Val/loss_EL', avg_el, epoch)  # NEW
    
    return avg_loss


def main():
    # Create hyperparameters
    hp = FixedHParams()
        
    print("\n" + "="*60)
    print("RWKV Separator Training with Embedding Loss Support")
    print("="*60)
    print(f"Device: {hp.device}")
    print(f"Batch size: {hp.batch_size}")
    print(f"Epochs: {hp.epochs}")
    print(f"Learning rate: {hp.lr}")
    
    # NEW: Print embedding loss settings
    print("\n" + "="*60)
    print("Embedding Loss Configuration:")
    print("="*60)
    if hp.use_embedding_loss:
        print(f"✅ ENABLED")
        print(f"  Mode: {hp.el_mode}")
        print(f"  Weight (lambda_el): {hp.lambda_el}")
        print(f"  Cosine similarity: {hp.el_cosine}")
    else:
        print(f"❌ DISABLED (set use_embedding_loss=True to enable)")
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(hp)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_rwkv_latents,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_rwkv_latents,
    )
    
    # Debug mode: overfit on small subset
    if hp.debug_mode:
        print(f"\n⚠️  DEBUG MODE: Overfitting on {hp.debug_overfit_batches} batches")
        train_subset = torch.utils.data.Subset(
            train_dataset, 
            list(range(min(hp.debug_overfit_batches * hp.batch_size, len(train_dataset))))
        )
        train_loader = DataLoader(
            train_subset,
            batch_size=hp.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_rwkv_latents,
        )
    
    # Create model
    model = create_model(hp)
    model = model.to(hp.device)
    
    # Wrap model
    model = RWKVSeparatorWrapper(model)
    
    # Test model output
    if not test_model_output(model, hp):
        print("\n❌ Model output test failed! Check model architecture.")
        return
    
    print("\n✅ Model output test passed!")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    
    if hp.resume_from_checkpoint and os.path.exists(hp.resume_from_checkpoint):
        print("\n" + "=" * 60)
        print(f"Resuming from checkpoint: {hp.resume_from_checkpoint}")
        print("=" * 60)
        
        checkpoint = torch.load(hp.resume_from_checkpoint, map_location=hp.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', 0)
        
        # Check if embedding loss settings changed
        saved_hp = checkpoint.get('hp', {})
        if saved_hp.get('use_embedding_loss') != hp.use_embedding_loss:
            print(f"\n⚠️  WARNING: Embedding loss setting changed!")
            print(f"   Saved: {saved_hp.get('use_embedding_loss')}")
            print(f"   Current: {hp.use_embedding_loss}")
        
        if saved_hp:
            print(f"\nCheckpoint info:")
            print(f"   Epoch: {start_epoch}")
            print(f"   Train loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
            print(f"   Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
            print(f"   Best val loss: {best_val_loss:.4f}")
            print(f"   Continuing from epoch {start_epoch}...")
    
    # Create learning rate scheduler
    scheduler = None
    if hp.use_lr_scheduler:
        print("\n" + "=" * 60)
        print("Creating learning rate scheduler...")
        print("=" * 60)
        
        remaining_epochs = hp.epochs - start_epoch
        
        if hp.lr_scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=remaining_epochs,
                eta_min=hp.lr_min
            )
            print(f"Scheduler: Cosine Annealing")
            print(f"  Start LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Min LR: {hp.lr_min:.2e}")
            print(f"  T_max: {remaining_epochs} epochs")
            
        elif hp.lr_scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=20,  # Decay every 20 epochs
                gamma=0.5      # Multiply by 0.5
            )
            print(f"Scheduler: Step Decay")
            print(f"  Start LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Step size: 20 epochs")
            print(f"  Decay factor: 0.5")
            
        elif hp.lr_scheduler_type == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                min_lr=hp.lr_min
            )
            print(f"Scheduler: Reduce on Plateau")
            print(f"  Start LR: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Min LR: {hp.lr_min:.2e}")
            print(f"  Patience: 5 epochs")
            print(f"  Factor: 0.5")
        else:
            print(f"⚠️  Unknown scheduler type: {hp.lr_scheduler_type}")
            print("   Training without LR scheduler")
            scheduler = None
    else:
        print("\n⚠️  LR scheduler disabled (use_lr_scheduler=False)")
        print(f"   Using constant LR: {hp.lr:.2e}")
    
    # Create gradient monitor
    grad_monitor = GradientMonitor()
    
    # Initialize TensorBoard writer
    print("\n" + "=" * 60)
    print("Initializing TensorBoard...")
    print("=" * 60)
    writer = SummaryWriter(hp.tensorboard_dir)
    print(f"TensorBoard logs: {hp.tensorboard_dir}")
    print(f"Run: tensorboard --logdir={hp.tensorboard_dir}")
    
    # Log hyperparameters to TensorBoard
    hparams_dict = {k: v for k, v in asdict(hp).items() if isinstance(v, (int, float, str, bool))}
    writer.add_text('Hyperparameters', json.dumps(hparams_dict, indent=2), 0)
    
    # Log model architecture
    writer.add_text('Model/Architecture', f"""
    n_layer: {hp.n_layer}
    n_embd: {hp.n_embd}
    num_sources: {hp.num_sources}
    head_mode: {hp.head_mode}
    total_params: {sum(p.numel() for p in model.parameters()):,}
    trainable_params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}
    
    Embedding Loss:
    use_embedding_loss: {hp.use_embedding_loss}
    el_mode: {hp.el_mode}
    lambda_el: {hp.lambda_el}
    el_cosine: {hp.el_cosine}
    """, 0)
    
    # Training loop setup
    os.makedirs(hp.ckpt_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    if start_epoch > 0:
        print(f"Resuming training from epoch {start_epoch}...")
    else:
        print("Starting training loop...")
    print("=" * 60)
    print(f"Training epochs: {start_epoch} → {hp.epochs}")
    print(f"Best val loss so far: {best_val_loss:.4f}")
    print("=" * 60)
    
    for epoch in range(start_epoch, hp.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, hp, epoch, grad_monitor, writer
        )
        
        if torch.isnan(torch.tensor(train_loss)):
            print("Training stopped due to NaN loss!")
            writer.close()
            break
        
        # Log epoch train loss to TensorBoard
        writer.add_scalar('Epoch/train_loss', train_loss, epoch)
        
        # Validate
        if (epoch + 1) % hp.val_every_n_epochs == 0:
            val_loss = evaluate(model, val_loader, hp, writer, epoch)
            
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, time={epoch_time:.1f}s")
            
            # Log to TensorBoard
            writer.add_scalar('Epoch/val_loss', val_loss, epoch)
            writer.add_scalar('Epoch/time_seconds', epoch_time, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hp': asdict(hp),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'best_val_loss': best_val_loss,
                }
                save_path = Path(hp.ckpt_dir) / "best_model.pt"
                torch.save(checkpoint, save_path)
                print(f"  ✅ Saved best model (val_loss={val_loss:.4f})")
                
                # Log best model update to TensorBoard
                writer.add_scalar('Best/val_loss', best_val_loss, epoch)
                writer.add_scalar('Best/epoch', best_epoch, epoch)
        
        # Save checkpoint after every epoch if enabled
        if hp.save_all_checkpoints:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hp': asdict(hp),
                'train_loss': train_loss,
                'val_loss': val_loss if (epoch + 1) % hp.val_every_n_epochs == 0 else None,
                'best_val_loss': best_val_loss,
            }
            save_path = Path(hp.ckpt_dir) / f"checkpoint_epoch{epoch+1:03d}.pt"
            torch.save(checkpoint, save_path)
            
            # Only print every N epochs to avoid clutter
            if (epoch + 1) % hp.save_every_n_epochs == 0:
                print(f"  💾 Saved checkpoint: {save_path.name}")
        else:
            # Original behavior: save periodic checkpoint
            if (epoch + 1) % hp.save_every_n_epochs == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hp': asdict(hp),
                    'train_loss': train_loss,
                    'val_loss': val_loss if (epoch + 1) % hp.val_every_n_epochs == 0 else None,
                }
                save_path = Path(hp.ckpt_dir) / f"checkpoint_epoch{epoch+1:03d}.pt"
                torch.save(checkpoint, save_path)
                print(f"  💾 Saved checkpoint: {save_path.name}")
        
        # Learning rate scheduler step
        current_lr = optimizer.param_groups[0]["lr"]
        
        if scheduler is not None:
            if hp.lr_scheduler_type == "plateau":
                # ReduceLROnPlateau needs validation loss
                if (epoch + 1) % hp.val_every_n_epochs == 0:
                    scheduler.step(val_loss)
            else:
                # Cosine and Step schedulers just need step()
                scheduler.step()
            
            # Log learning rate to TensorBoard
            new_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar('Epoch/learning_rate', new_lr, epoch)
            
            # Print LR change if significant
            if abs(new_lr - current_lr) / current_lr > 0.01:  # >1% change
                print(f"  📉 Learning rate: {current_lr:.2e} → {new_lr:.2e}")
        else:
            # Log constant LR
            writer.add_scalar('Epoch/learning_rate', current_lr, epoch)
    
    # Training complete - log final summary
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}")
    print(f"Best model saved at: {Path(hp.ckpt_dir) / 'best_model.pt'}")
    print(f"TensorBoard logs: {hp.tensorboard_dir}")
    print("=" * 60)
    
    # Add final summary to TensorBoard
    writer.add_text('Training/Summary', f"""
    Training Complete!
    
    Best Epoch: {best_epoch + 1}
    Best Validation Loss: {best_val_loss:.4f}
    Total Epochs: {hp.epochs}
    Final Train Loss: {train_loss:.4f}
    
    Embedding Loss Used: {hp.use_embedding_loss}
    """, hp.epochs)
    
    writer.close()


if __name__ == "__main__":
    main()
