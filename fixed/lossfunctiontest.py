#!/usr/bin/env python3
"""
Diagnose if loss function has a floor bug
"""
import torch
import torch.nn.functional as F
from pit_losses import total_separator_loss

def test_loss_function():
    """Test if loss reaches zero for perfect predictions."""
    
    print("="*70)
    print("LOSS FUNCTION DIAGNOSTIC")
    print("="*70)
    
    # Create test data
    B, T, C = 1, 100, 1024
    
    # Ground truth sources
    z_s1 = torch.randn(B, T, C)
    z_s2 = torch.randn(B, T, C)
    
    print(f"\n📊 Test Setup:")
    print(f"  Batch: {B}, Time: {T}, Channels: {C}")
    print(f"  z_s1 norm: {z_s1.norm():.2f}")
    print(f"  z_s2 norm: {z_s2.norm():.2f}")
    print(f"  Difference: {(z_s1 - z_s2).abs().mean():.4f}")
    
    # Test 1: Perfect predictions
    print(f"\n{'='*70}")
    print("TEST 1: Perfect Predictions (should be ~0)")
    print('='*70)
    
    preds_perfect = {
        'pred1': z_s1.clone(),
        'pred2': z_s2.clone(),
    }
    
    loss_perfect, logs_perfect, perm = total_separator_loss(
        preds_perfect, z_s1, z_s2, pad_mask=None,
        lambda_residual_l2=0.0,
        el_mode="none",
        lambda_el=0.0,
    )
    
    print(f"Loss (perfect): {loss_perfect.item():.10f}")
    print(f"  Expected: ~0.0 (< 1e-6)")
    
    if loss_perfect.item() < 1e-6:
        print(f"  ✅ PASS: Loss correctly reaches zero")
    elif loss_perfect.item() < 1e-3:
        print(f"  ⚠️  WARNING: Small non-zero floor ({loss_perfect.item():.6f})")
    else:
        print(f"  ❌ FAIL: Large loss floor ({loss_perfect.item():.6f})")
        print(f"  → Loss function has a bug!")
    
    # Test 2: Swapped predictions
    print(f"\n{'='*70}")
    print("TEST 2: Swapped Predictions (should be ~0)")
    print('='*70)
    
    preds_swapped = {
        'pred1': z_s2.clone(),
        'pred2': z_s1.clone(),
    }
    
    loss_swapped, logs_swapped, perm = total_separator_loss(
        preds_swapped, z_s1, z_s2, pad_mask=None,
        lambda_residual_l2=0.0,
        el_mode="none",
        lambda_el=0.0,
    )
    
    print(f"Loss (swapped): {loss_swapped.item():.10f}")
    print(f"  PIT should handle permutation")
    
    if loss_swapped.item() < 1e-6:
        print(f"  ✅ PASS: PIT works correctly")
    else:
        print(f"  ❌ FAIL: PIT not handling permutation")
    
    # Test 3: Random predictions
    print(f"\n{'='*70}")
    print("TEST 3: Random Predictions (should be large)")
    print('='*70)
    
    preds_random = {
        'pred1': torch.randn(B, T, C),
        'pred2': torch.randn(B, T, C),
    }
    
    loss_random, logs_random, perm = total_separator_loss(
        preds_random, z_s1, z_s2, pad_mask=None,
        lambda_residual_l2=0.0,
        el_mode="none",
        lambda_el=0.0,
    )
    
    print(f"Loss (random): {loss_random.item():.6f}")
    print(f"  Expected: > 1.0")
    
    if loss_random.item() > 1.0:
        print(f"  ✅ PASS: Random predictions have high loss")
    else:
        print(f"  ❌ FAIL: Random loss too low")
    
    # Test 4: Check MSE directly
    print(f"\n{'='*70}")
    print("TEST 4: Direct MSE Check")
    print('='*70)
    
    mse_perfect = F.mse_loss(z_s1, z_s1)
    mse_random = F.mse_loss(z_s1, torch.randn(B, T, C))
    
    print(f"MSE(z_s1, z_s1): {mse_perfect.item():.10f}")
    print(f"MSE(z_s1, random): {mse_random.item():.6f}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print('='*70)
    
    if loss_perfect.item() < 1e-6 and loss_random.item() > 1.0:
        print("✅ Loss function is working correctly")
        print("\n💡 If training plateaus at 0.17:")
        print("  → Issue is in model architecture or training")
        print("  → NOT in loss function")
        return True
    else:
        print("❌ Loss function has issues")
        print("\n🔧 Possible problems:")
        if loss_perfect.item() > 1e-6:
            print("  - Loss doesn't reach zero for perfect predictions")
            print("  - Check _masked_mse implementation")
            print("  - Check for extra regularization terms")
        if loss_random.item() < 1.0:
            print("  - Loss too small for random predictions")
            print("  - Possible normalization issue")
        return False


if __name__ == "__main__":
    test_loss_function()