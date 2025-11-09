# dataset_latents_fixed.py
"""
FIXED VERSION: dataset_latents.py
Improved dataset for loading DAC latents with:
- Automatic normalization with cached statistics
- Better tensor orientation detection using metadata
- Validation of channel consistency
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset
import json

HEAD_SIZE_A = 64  # Expected to be divisible by this for RWKV v7


def _to_tc_robust(z, expected_C: int | None = None) -> torch.Tensor:
    """
    Convert latent to [T, C] format with robust handling.
    
    Priority order:
    1. Use metadata if available (most reliable)
    2. Use expected_C if provided
    3. Use heuristics as fallback
    """
    # Extract tensor and metadata
    if isinstance(z, dict):
        t = torch.as_tensor(z["z"])
        meta_C = z.get("C")
        meta_T = z.get("T")
        shape_order = z.get("shape_order", None)
    else:
        t = torch.as_tensor(z)
        meta_C = meta_T = shape_order = None
    
    # Ensure 2D
    if t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim != 2:
        raise ValueError(f"Latent must be 2D [T,C] or squeezable to 2D, got {tuple(t.shape)}")
    
    # Method 1: Use explicit shape_order metadata (most reliable)
    if shape_order == "TC":
        # Already in [T, C] format
        if expected_C is not None and t.shape[1] != expected_C:
            print(f"[WARNING] Expected C={expected_C} but got C={t.shape[1]} from metadata")
        return t.contiguous().to(torch.float32)
    elif shape_order == "CT":
        # Need to transpose from [C, T] to [T, C]
        t = t.transpose(0, 1)
        if expected_C is not None and t.shape[1] != expected_C:
            print(f"[WARNING] Expected C={expected_C} but got C={t.shape[1]} from metadata")
        return t.contiguous().to(torch.float32)
    
    # Method 2: Use C and T metadata if available
    if meta_C is not None and meta_T is not None:
        if t.shape == (meta_C, meta_T):
            t = t.transpose(0, 1)  # [C, T] -> [T, C]
        elif t.shape == (meta_T, meta_C):
            pass  # Already [T, C]
        else:
            print(f"[WARNING] Shape {tuple(t.shape)} doesn't match metadata C={meta_C}, T={meta_T}")
        
        if expected_C is not None and t.shape[1] != expected_C:
            print(f"[WARNING] After metadata-based transpose: expected C={expected_C}, got C={t.shape[1]}")
        return t.contiguous().to(torch.float32)
    
    # Method 3: Use expected_C if provided
    d0, d1 = int(t.shape[0]), int(t.shape[1])
    if expected_C is not None:
        if d1 == expected_C:
            # Already [T, C]
            pass
        elif d0 == expected_C:
            # [C, T] -> [T, C]
            t = t.transpose(0, 1)
        else:
            # Neither dimension matches expected_C
            print(f"[WARNING] Neither dimension ({d0}, {d1}) matches expected_C={expected_C}")
            # Use heuristic: smaller dimension is likely channels
            if d0 < d1:
                t = t.transpose(0, 1)
    else:
        # Method 4: Heuristics as fallback
        # Assume channels are:
        # - Divisible by 64 (common for neural codecs)
        # - Smaller dimension (channels < time frames usually)
        # - Less than 2048 (reasonable upper bound for channels)
        
        if d1 % HEAD_SIZE_A == 0 and d1 <= 2048:
            # Likely already [T, C]
            pass
        elif d0 % HEAD_SIZE_A == 0 and d0 <= 2048:
            # Likely [C, T] -> transpose
            t = t.transpose(0, 1)
        else:
            # Both or neither divisible by 64, use size heuristic
            if d0 < d1:
                t = t.transpose(0, 1)
            # else keep as is
    
    return t.contiguous().to(torch.float32)


class RWKVLatentDataset(Dataset):
    def __init__(
        self,
        root,
        require_targets=True,
        extensions=(".pt",),
        expected_C: int | None = None,
        normalize_latents=True,
        norm_stats_path: str | None = None,
        force_recompute_stats=False,
    ):
        self.root = Path(root)
        self.mix_dir = self.root / "mix_clean"
        self.s1_dir = self.root / "s1"
        self.s2_dir = self.root / "s2"
        self.require_targets = require_targets
        self.extensions = extensions
        self.normalize_latents = normalize_latents
        
        # Validate directories
        assert self.mix_dir.is_dir(), f"Missing {self.mix_dir}"
        if require_targets:
            assert self.s1_dir.is_dir() and self.s2_dir.is_dir(), "Missing s1/ or s2/"
        
        # Build list of utterances
        mix_files = [p for p in self.mix_dir.rglob("*") if p.suffix in extensions]
        if require_targets:
            s1_set = {p.relative_to(self.s1_dir).with_suffix("").as_posix() 
                     for p in self.s1_dir.rglob("*") if p.suffix in extensions}
            s2_set = {p.relative_to(self.s2_dir).with_suffix("").as_posix() 
                     for p in self.s2_dir.rglob("*") if p.suffix in extensions}
        
        items = []
        for m in mix_files:
            rel = m.relative_to(self.mix_dir).with_suffix("")
            if require_targets:
                key = rel.as_posix()
                if key not in s1_set or key not in s2_set:
                    continue
            items.append(rel)
        
        assert len(items) > 0, f"No items found under {self.root}"
        self.items = sorted(items)
        
        # Auto-detect or validate expected_C
        if expected_C is None:
            # Auto-detect from first file
            first_mix = self.mix_dir / (str(self.items[0]) + ".pt")
            obj = torch.load(first_mix, map_location="cpu", weights_only=False)
            z = _to_tc_robust(obj)
            expected_C = z.shape[1]
            print(f"[INFO] Auto-detected latent channels: C={expected_C}")
        
        self.expected_C = expected_C
        
        # Filter items by channel count
        if self.expected_C is not None:
            kept = []
            dropped = 0
            for rel in self.items:
                mix_p = self.mix_dir / (str(rel) + ".pt")
                obj = torch.load(mix_p, map_location="cpu", weights_only=False)
                z = _to_tc_robust(obj, expected_C=self.expected_C)
                C_here = z.shape[1]
                if C_here == self.expected_C:
                    kept.append(rel)
                else:
                    dropped += 1
            
            if dropped > 0:
                print(f"[INFO] Filtered out {dropped} files with C != {self.expected_C}")
            self.items = kept
        
        assert len(self.items) > 0, f"No items with C={self.expected_C} under {self.root}"
        print(f"[INFO] Dataset initialized with {len(self.items)} items, C={self.expected_C}")
        
        # Initialize normalization statistics
        self.mean = 0.0
        self.std = 1.0
        if normalize_latents:
            self._init_normalization(norm_stats_path, force_recompute_stats)
    
    def _init_normalization(self, norm_stats_path: str | None, force_recompute: bool):
        """Initialize normalization statistics."""
        stats_file = Path(norm_stats_path) if norm_stats_path else self.root / "latent_stats.json"
        
        if not force_recompute and stats_file.exists():
            # Load existing stats
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            self.mean = stats['mean']
            self.std = stats['std']
            print(f"[INFO] Loaded normalization stats from {stats_file}")
            print(f"       mean={self.mean:.4f}, std={self.std:.4f}")
        else:
            # Compute stats from dataset
            print("[INFO] Computing normalization statistics...")
            self._compute_norm_stats()
            
            # Save stats for future use
            stats = {'mean': self.mean, 'std': self.std}
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_file, 'w') as f:
                json.dump(stats, f)
            print(f"[INFO] Saved normalization stats to {stats_file}")
    
    def _compute_norm_stats(self):
        """Compute mean and std from a sample of the dataset."""
        sample_size = min(100, len(self.items))
        indices = torch.randperm(len(self.items))[:sample_size]
        
        # First pass: compute mean
        n_total = 0
        sum_total = 0.0
        
        for idx in indices:
            rel = self.items[idx]
            
            # Load all sources for comprehensive stats
            mix_p = self.mix_dir / (str(rel) + ".pt")
            obj = torch.load(mix_p, map_location="cpu", weights_only=False)
            z_mix = _to_tc_robust(obj, expected_C=self.expected_C)
            
            sum_total += z_mix.sum().item()
            n_total += z_mix.numel()
            
            if self.require_targets:
                s1_p = self.s1_dir / (str(rel) + ".pt")
                s2_p = self.s2_dir / (str(rel) + ".pt")
                
                obj_s1 = torch.load(s1_p, map_location="cpu", weights_only=False)
                obj_s2 = torch.load(s2_p, map_location="cpu", weights_only=False)
                
                z_s1 = _to_tc_robust(obj_s1, expected_C=self.expected_C)
                z_s2 = _to_tc_robust(obj_s2, expected_C=self.expected_C)
                
                sum_total += z_s1.sum().item() + z_s2.sum().item()
                n_total += z_s1.numel() + z_s2.numel()
        
        self.mean = sum_total / n_total
        
        # Second pass: compute std
        sum_sq = 0.0
        
        for idx in indices:
            rel = self.items[idx]
            
            mix_p = self.mix_dir / (str(rel) + ".pt")
            obj = torch.load(mix_p, map_location="cpu", weights_only=False)
            z_mix = _to_tc_robust(obj, expected_C=self.expected_C)
            
            sum_sq += ((z_mix - self.mean) ** 2).sum().item()
            
            if self.require_targets:
                s1_p = self.s1_dir / (str(rel) + ".pt")
                s2_p = self.s2_dir / (str(rel) + ".pt")
                
                obj_s1 = torch.load(s1_p, map_location="cpu", weights_only=False)
                obj_s2 = torch.load(s2_p, map_location="cpu", weights_only=False)
                
                z_s1 = _to_tc_robust(obj_s1, expected_C=self.expected_C)
                z_s2 = _to_tc_robust(obj_s2, expected_C=self.expected_C)
                
                sum_sq += ((z_s1 - self.mean) ** 2).sum().item()
                sum_sq += ((z_s2 - self.mean) ** 2).sum().item()
        
        self.std = (sum_sq / n_total) ** 0.5
        
        # Ensure std is not too small
        self.std = max(self.std, 0.01)
        
        print(f"[INFO] Computed norm stats from {sample_size} samples:")
        print(f"       mean={self.mean:.4f}, std={self.std:.4f}")
    
    def __len__(self) -> int:
        return len(self.items)
    
    def _load_payload(self, p: Path) -> Dict:
        """Load and validate payload."""
        obj = torch.load(p, map_location="cpu", weights_only=False)
        assert "z" in obj, f"Missing 'z' in {p}"
        
        # Log statistics from first file if available
        if not hasattr(self, '_logged_stats') and 'stats' in obj:
            stats = obj['stats']
            print(f"[DEBUG] File stats - mean: {stats.get('mean', 'N/A'):.4f}, "
                  f"std: {stats.get('std', 'N/A'):.4f}, "
                  f"range: [{stats.get('min', 'N/A'):.2f}, {stats.get('max', 'N/A'):.2f}]")
            self._logged_stats = True
        
        return obj
    
    def __getitem__(self, idx: int) -> Dict:
        """Get item with optional normalization."""
        rel = self.items[idx]
        mix_p = self.mix_dir / (str(rel) + ".pt")
        mix_obj = self._load_payload(mix_p)
        
        # Convert to [T, C]
        zmix = _to_tc_robust(mix_obj, expected_C=self.expected_C)
        
        # Apply normalization
        if self.normalize_latents:
            zmix = (zmix - self.mean) / (self.std + 1e-8)
        
        ex: Dict = {
            "utt_id": str(rel),
            "z_mix": zmix,  # [T, C]
            "fps": float(mix_obj.get("fps", 50)),  # Use actual FPS from metadata
            "sr": int(mix_obj.get("sr", 16000)),
        }
        
        if self.require_targets:
            s1_p = self.s1_dir / (str(rel) + ".pt")
            s2_p = self.s2_dir / (str(rel) + ".pt")
            s1_obj = self._load_payload(s1_p)
            s2_obj = self._load_payload(s2_p)
            
            z_s1 = _to_tc_robust(s1_obj, expected_C=self.expected_C)
            z_s2 = _to_tc_robust(s2_obj, expected_C=self.expected_C)
            
            # Apply normalization to targets
            if self.normalize_latents:
                z_s1 = (z_s1 - self.mean) / (self.std + 1e-8)
                z_s2 = (z_s2 - self.mean) / (self.std + 1e-8)
            
            ex["z_s1"] = z_s1
            ex["z_s2"] = z_s2
        
        return ex
    
    def get_normalization_params(self) -> Dict[str, float]:
        """Return normalization parameters for use in inference."""
        return {
            "mean": self.mean,
            "std": self.std,
            "normalized": self.normalize_latents,
        }
