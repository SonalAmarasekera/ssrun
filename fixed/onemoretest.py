#!/usr/bin/env python3
import torch
from pathlib import Path

def comprehensive_check(file_path):
    s1_data = torch.load(Path(file_path).parent.parent / "s1" / Path(file_path).name, weights_only=False)
    s2_data = torch.load(Path(file_path).parent.parent / "s2" / Path(file_path).name, weights_only=False)
    
    z1 = s1_data['z']
    z2 = s2_data['z']
    
    # All metrics
    cosine = torch.nn.functional.cosine_similarity(z1.flatten(), z2.flatten(), dim=0).item()
    diff = (z1 - z2).abs().mean().item()
    #ratio = diff / z1.norm().item()
    rel_diff = diff / ((z1.std() + z2.std()) / 2).item()
    
    print(f"File: {Path(file_path).name}")
    print(f"  Cosine similarity: {cosine:.6f} ({' ❌  ' if cosine > 0.95 else ' ✅  '})")
    #print(f"  Ratio: {ratio:.6f} ({' ❌  ' if ratio < 0.05 else ' ✅  '})")
    print(f"  Relative diff: {rel_diff:.6f} ({' ❌  ' if rel_diff < 0.1 else ' ✅  '})")
    
    # Verdict
    if cosine < 0.95 and rel_diff > 0.1:
        print("  ✅ GOOD: Should be separable")
    elif cosine < 0.95 and ratio > 0.01:
        print("  ⚠️  MARGINAL: Might be difficult")
    else:
        print("  ❌ BAD: Too similar to separate")

# Test on your files
comprehensive_check("/workspace/latents/min/train/mix_clean/40no0311_1.3078_40mo0314_-1.3078.pt")