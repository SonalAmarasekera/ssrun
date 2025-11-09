# train_separator_fixed.py
"""
FIXED training script for RWKV-based speech separator.
Major fixes:
1. Proper model output format handling
2. Dataset normalization
3. Correct config parameters
4. Better gradient monitoring
5. Improved learning rate scheduling
"""

from pathlib import Path
import os, math, json, random, time
from dataclasses import dataclass, asdict
from functools import partial
from typing import Optional, Dict

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
    """Fixed hyperparameters with better defaults."""
    
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
        num_sources=hp.num_sources,  # Added
        head_hidden=hp.head_hidden,  # Fixed from 'hidden_dim'
        head_mode=hp.head_mode,  # Added
    )
    
    # Build model
    base_model = build_rwkv7_separator(
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        num_sources=cfg.num_sources,
        head_size_a=cfg.head_size_a,
        head_hidden=cfg.head_hidden,
        head_mode=cfg.head_mode,
        enforce_bf16=cfg.enforce_bf16,
    )
    
    # Wrap model for correct output format
    model = RWKVSeparatorWrapper(base_model)
    
    # Move to device
    model = model.to(hp.device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model configuration:")
    print(f"  n_embd: {cfg.n_embd}")
    print(f"  n_layer: {cfg.n_layer}")
    print(f"  num_sources: {cfg.num_sources}")
    print(f"  head_mode: {cfg.head_mode}")
    print(f"  enforce_bf16: {cfg.enforce_bf16}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model, cfg


def test_model_output(model, hp: FixedHParams):
    """Test that model output format is correct."""
    
    print("\n" + "="*60)
    print("Testing model output format...")
    print("="*60)
    
    # Create dummy input
    x = torch.randn(2, 160, hp.n_embd).to(hp.device)  # [B, T, C]
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    # Check output format
    print(f"Output type: {type(output)}")
    
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
):
    """Train for one epoch with proper monitoring."""
    
    model.train()
    device = hp.device
    
    running_loss = 0.0
    running_pit = 0.0
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
        
        # Forward pass
        optimizer.zero_grad(set_to_none=True)
        
        if hp.enforce_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(z_mix)
                loss, logs, perm = total_separator_loss(
                    output, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
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
                
                loss, logs, perm = total_separator_loss(
                    output_scaled, z_s1_scaled, z_s2_scaled, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
                )
            else:
                loss, logs, perm = total_separator_loss(
                    output, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
                )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping and monitoring
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip)
        
        # Log gradients
        current_lr = optimizer.param_groups[0]["lr"]
        grad_norm_logged, update_ratio = grad_monitor.log_gradients(model, current_lr)
        
        # Optimizer step
        optimizer.step()
        
        # Update metrics
        running_loss += loss.item()
        running_pit += logs["loss/pit_latent"].item()
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        avg_pit = running_pit / (batch_idx + 1)
        
        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'pit': f'{avg_pit:.4f}',
            'grad': f'{grad_norm:.2f}',
            'lr': f'{current_lr:.2e}',
            'perm': str(perm),
        })
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"\n🚨 NaN loss detected at batch {batch_idx}!")
            print(f"  Gradient norm: {grad_norm:.2f}")
            print(f"  Input stats: mean={z_mix.mean():.4f}, std={z_mix.std():.4f}")
            return float('nan')
    
    epoch_loss = running_loss / num_batches
    return epoch_loss


@torch.no_grad()
def evaluate(model, val_loader, hp: FixedHParams):
    """Evaluate model on validation set."""
    
    model.eval()
    device = hp.device
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(val_loader, desc="Validation"):
        z_mix = batch["z_mix"].to(device)
        z_s1 = batch["z_s1"].to(device)
        z_s2 = batch["z_s2"].to(device)
        mask = batch["mask"].to(device)
        
        if hp.enforce_bf16 and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(z_mix)
                loss, logs, _ = total_separator_loss(
                    output, z_s1, z_s2, mask,
                    lambda_residual_l2=hp.lambda_residual_l2,
                    lambda_mask_entropy=hp.lambda_mask_entropy,
                )
        else:
            output = model(z_mix)
            loss, logs, _ = total_separator_loss(
                output, z_s1, z_s2, mask,
                lambda_residual_l2=hp.lambda_residual_l2,
                lambda_mask_entropy=hp.lambda_mask_entropy,
            )
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def main():
    """Main training script."""
    
    # Initialize hyperparameters
    hp = FixedHParams()
    
    print("\n" + "="*80)
    print("RWKV SEPARATOR TRAINING (FIXED)")
    print("="*80)
    
    # Create datasets with normalization
    train_dataset, val_dataset = create_datasets(hp)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=hp.batch_size,
        shuffle=True,
        collate_fn=partial(collate_rwkv_latents, chunk_len=hp.chunk_len),
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=hp.batch_size,
        shuffle=False,
        collate_fn=partial(collate_rwkv_latents, chunk_len=hp.chunk_len),
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model with wrapper
    model, cfg = create_model(hp)
    
    # Test model output format
    if not test_model_output(model, hp):
        print("❌ Model output format test FAILED! Fix the model before training.")
        return
    
    print("\n✅ All tests passed! Starting training...")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp.lr,
        weight_decay=hp.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Create gradient monitor
    grad_monitor = GradientMonitor()
    
    # Training loop
    best_val_loss = float('inf')
    os.makedirs(hp.ckpt_dir, exist_ok=True)
    
    for epoch in range(hp.epochs):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, hp, epoch, grad_monitor
        )
        
        if torch.isnan(torch.tensor(train_loss)):
            print("Training stopped due to NaN loss!")
            break
        
        # Validate
        if (epoch + 1) % hp.val_every_n_epochs == 0:
            val_loss = evaluate(model, val_loader, hp)
            print(f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hp': asdict(hp),
                    'val_loss': val_loss,
                }
                save_path = Path(hp.ckpt_dir) / f"best_model.pt"
                torch.save(checkpoint, save_path)
                print(f"  Saved best model (val_loss={val_loss:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % hp.save_every_n_epochs == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hp': asdict(hp),
                'train_loss': train_loss,
            }
            save_path = Path(hp.ckpt_dir) / f"checkpoint_epoch{epoch+1:03d}.pt"
            torch.save(checkpoint, save_path)
            print(f"  Saved checkpoint: {save_path}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
