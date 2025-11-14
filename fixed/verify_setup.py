#!/usr/bin/env python3
"""
Quick Verification Checklist
Run this before starting training to ensure everything is fixed.
"""

import sys
from pathlib import Path
import torch


def check_validation_data():
    """Check if validation data has been fixed."""
    print("\n" + "="*60)
    print("CHECK 1: Validation Data")
    print("="*60)
    
    val_root = Path("/workspace/latents/min/dev")
    
    if not val_root.exists():
        print("❌ Validation directory doesn't exist!")
        print(f"   Expected: {val_root}")
        return False
    
    s1_files = list((val_root / "s1").glob("*.pt"))
    s2_files = list((val_root / "s2").glob("*.pt"))
    
    if not s1_files or not s2_files:
        print("❌ No validation files found!")
        return False
    
    print(f"✓ Found {len(s1_files)} s1 files, {len(s2_files)} s2 files")
    
    # Load first file pair
    s1_data = torch.load(s1_files[0], weights_only=False)
    s2_data = torch.load(s2_files[0], weights_only=False)
    
    z1 = s1_data['z']
    z2 = s2_data['z']
    
    diff = (z1 - z2).abs().mean().item()
    ratio = diff / (z1.norm().item() + 1e-8)
    
    print(f"\nSample file: {s1_files[0].name}")
    print(f"  s1 norm: {z1.norm().item():.2f}")
    print(f"  s2 norm: {z2.norm().item():.2f}")
    print(f"  Difference: {diff:.4f}")
    print(f"  Ratio: {ratio:.6f}")
    
    if ratio < 0.01:
        print("\n🚨 VALIDATION DATA STILL BROKEN!")
        print("   s1 and s2 are nearly identical")
        print("   Run: python cache_latents_fixed.py --csv val.csv ...")
        return False
    elif ratio < 0.05:
        print("\n⚠️  Validation data questionable (ratio < 0.05)")
        return False
    else:
        print("\n✅ VALIDATION DATA IS CORRECT!")
        return True


def check_train_data():
    """Check if training data is correct."""
    print("\n" + "="*60)
    print("CHECK 2: Training Data")
    print("="*60)
    
    train_root = Path("/workspace/latents/min/train")
    
    if not train_root.exists():
        print("❌ Training directory doesn't exist!")
        return False
    
    s1_files = list((train_root / "s1").glob("*.pt"))
    s2_files = list((train_root / "s2").glob("*.pt"))
    
    if not s1_files or not s2_files:
        print("❌ No training files found!")
        return False
    
    print(f"✓ Found {len(s1_files)} s1 files, {len(s2_files)} s2 files")
    
    # Check a few files
    num_check = min(3, len(s1_files))
    all_good = True
    
    for i in range(num_check):
        s1_data = torch.load(s1_files[i], weights_only=False)
        s2_data = torch.load(s2_files[i], weights_only=False)
        
        z1 = s1_data['z']
        z2 = s2_data['z']
        
        diff = (z1 - z2).abs().mean().item()
        ratio = diff / (z1.norm().item() + 1e-8)
        
        status = "✅" if ratio > 0.05 else "⚠️" if ratio > 0.01 else "🚨"
        print(f"\nFile {i+1}: {s1_files[i].name}")
        print(f"  Ratio: {ratio:.6f} {status}")
        
        if ratio < 0.05:
            all_good = False
    
    if all_good:
        print("\n✅ TRAINING DATA IS CORRECT!")
        return True
    else:
        print("\n🚨 TRAINING DATA HAS ISSUES!")
        return False


def check_model_architecture():
    """Check if model is using fixed separation head."""
    print("\n" + "="*60)
    print("CHECK 3: Model Architecture")
    print("="*60)
    
    try:
        # Try to import and check
        from rwkv_separator_Claudemod import RWKVv7Separator, SeparatorV7Config
        
        # Create small test model
        cfg = SeparatorV7Config(
            n_embd=256,
            n_layer=2,
            num_sources=2,
            head_hidden=128,
        )
        
        model = RWKVv7Separator(cfg)
        
        # Check head type
        head_class = type(model.head).__name__
        print(f"Separation head type: {head_class}")
        
        if "Fixed" in head_class or "Direct" in head_class:
            print("✅ Using fixed separation head!")
            return True
        elif "Gated" in head_class:
            print("🚨 Still using SeparationHeadGated!")
            print("   Replace with SeparationHeadFixed")
            return False
        else:
            print(f"⚠️  Unknown head type: {head_class}")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def check_training_script():
    """Check if training script has diversity loss."""
    print("\n" + "="*60)
    print("CHECK 4: Training Script")
    print("="*60)
    
    try:
        with open("train_separator_EL-LR.py", "r") as f:
            content = f.read()
        
        has_diversity = "diversity" in content.lower()
        has_diagnostic = "diagnostic" in content.lower()
        
        if has_diversity:
            print("✅ Training script has diversity loss")
        else:
            print("⚠️  Training script missing diversity loss")
            print("   Consider adding it (see training_modifications.py)")
        
        if has_diagnostic:
            print("✅ Training script has diagnostics")
        else:
            print("⚠️  Training script missing diagnostics")
        
        return has_diversity
        
    except FileNotFoundError:
        print("⚠️  Could not find train_separator_EL-LR.py")
        return False


def main():
    """Run all checks."""
    print("="*60)
    print("VERIFICATION CHECKLIST")
    print("="*60)
    
    results = {}
    
    # Run checks
    results['validation_data'] = check_validation_data()
    results['training_data'] = check_train_data()
    results['model'] = check_model_architecture()
    results['training_script'] = check_training_script()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    all_pass = all(results.values())
    
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check.replace('_', ' ').title()}")
    
    print("\n" + "="*60)
    
    if all_pass:
        print("🎉 ALL CHECKS PASSED!")
        print("\nYou're ready to train:")
        print("  python train.py --debug_mode=True --debug_overfit_batches=1 --epochs=30")
        print("\nExpected results:")
        print("  - Loss < 0.01 by epoch 20")
        print("  - Output ratio > 0.05")
        print("  - Gradients > 0.001")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED!")
        print("\nFix the issues above before training.")
        print("\nMost critical:")
        if not results['validation_data']:
            print("  1. Regenerate validation data")
        if not results['model']:
            print("  2. Replace separation head with SeparationHeadFixed")
        return 1


if __name__ == "__main__":
    sys.exit(main())