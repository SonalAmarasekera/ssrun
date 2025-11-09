# rwkv_separator_wrapper.py
"""
Wrapper for RWKV Separator to ensure correct output format for loss function.
This wrapper converts the model's tensor output [B,T,S,C] to the dictionary format
expected by the PIT loss function.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class RWKVSeparatorWrapper(nn.Module):
    """
    Wrapper that converts RWKV separator output to dictionary format for PIT loss.
    
    The underlying model outputs shape [B,T,num_sources,C]
    This wrapper converts it to {'pred1': [B,T,C], 'pred2': [B,T,C], ...}
    """
    
    def __init__(self, base_model):
        """
        Args:
            base_model: The RWKV separator model instance
        """
        super().__init__()
        self.model = base_model
        self.cfg = base_model.cfg
        self.num_sources = getattr(base_model.cfg, 'num_sources', 2)
        
    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with output format conversion.
        
        Args:
            x: Input tensor [B, T, C]
            pad_mask: Optional padding mask [B, T] (not used by base model but kept for compatibility)
            
        Returns:
            Dictionary with keys 'pred1', 'pred2', etc. containing separated sources
        """
        # Get model output [B, T, num_sources, C]
        y = self.model(x)
        
        # Check output shape
        if y.dim() == 4:
            B, T, S, C = y.shape
            assert S == self.num_sources, f"Expected {self.num_sources} sources, got {S}"
        else:
            raise ValueError(f"Expected 4D output [B,T,S,C], got shape {y.shape}")
        
        # Convert to dictionary format
        output_dict = {}
        
        # Add separated sources
        for i in range(self.num_sources):
            output_dict[f'pred{i+1}'] = y[:, :, i, :]  # [B, T, C]
        
        # Optionally add mask outputs if in mask mode
        if hasattr(self.cfg, 'head_mode') and self.cfg.head_mode == "mask":
            # In mask mode, the outputs are actually masks that should be sigmoid-activated
            for i in range(self.num_sources):
                output_dict[f'mask{i+1}'] = torch.sigmoid(y[:, :, i, :])
        
        # Optionally add residual outputs if model produces them
        # (Your current model doesn't, but this is for future compatibility)
        if hasattr(self, 'compute_residuals') and self.compute_residuals:
            # Compute residuals as difference from input
            for i in range(self.num_sources):
                output_dict[f'resid{i+1}'] = output_dict[f'pred{i+1}'] - x
        
        return output_dict
    
    @property
    def device(self):
        """Get device of the model."""
        return next(self.model.parameters()).device
    
    def train(self, mode=True):
        """Set training mode."""
        super().train(mode)
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set eval mode."""
        super().eval()
        self.model.eval()
        return self


def create_wrapped_separator(cfg_or_model, **kwargs):
    """
    Factory function to create a wrapped separator.
    
    Args:
        cfg_or_model: Either a SeparatorV7Config or an already instantiated model
        **kwargs: Additional arguments if creating from config
        
    Returns:
        Wrapped model ready for training
    """
    if hasattr(cfg_or_model, 'forward'):
        # Already a model instance
        return RWKVSeparatorWrapper(cfg_or_model)
    else:
        # It's a config, need to create model first
        from rwkv_separator_DSmod import RWKVv7Separator
        model = RWKVv7Separator(cfg_or_model, **kwargs)
        return RWKVSeparatorWrapper(model)


# Alternative: Monkey-patch the existing model class
def patch_separator_forward(model_class):
    """
    Monkey-patch an existing separator class to output the correct format.
    This modifies the class in-place.
    
    Usage:
        from rwkv_separator_DSmod import RWKVv7Separator
        patch_separator_forward(RWKVv7Separator)
    """
    original_forward = model_class.forward
    
    def wrapped_forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Call original forward
        y = original_forward(x)
        
        # Convert output
        if not isinstance(y, dict):
            if y.dim() == 4:
                B, T, S, C = y.shape
                output_dict = {}
                for i in range(S):
                    output_dict[f'pred{i+1}'] = y[:, :, i, :]
                return output_dict
            else:
                # Assume it's already [B,T,C] and there's only one source?
                return {'pred1': y, 'pred2': y}  # Duplicate for compatibility
        return y
    
    model_class.forward = wrapped_forward
    return model_class


# Test function
def test_wrapper():
    """Test the wrapper with a mock model."""
    
    class MockSeparator(nn.Module):
        """Mock separator for testing."""
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.linear = nn.Linear(cfg.n_embd, cfg.n_embd * cfg.num_sources)
            
        def forward(self, x):
            B, T, C = x.shape
            y = self.linear(x)  # [B, T, C*num_sources]
            y = y.view(B, T, self.cfg.num_sources, C)  # [B, T, S, C]
            return y
    
    # Create mock config
    class MockConfig:
        n_embd = 128
        num_sources = 2
        head_mode = "residual"
    
    # Test wrapper
    cfg = MockConfig()
    base_model = MockSeparator(cfg)
    wrapped_model = RWKVSeparatorWrapper(base_model)
    
    # Test forward pass
    x = torch.randn(2, 100, 128)  # [B, T, C]
    output = wrapped_model(x)
    
    print("Wrapper test results:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output type: {type(output)}")
    print(f"  Output keys: {list(output.keys())}")
    for k, v in output.items():
        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
    
    # Test with loss function
    from pit_losses import pit_latent_mse_2spk
    
    targets = torch.randn(2, 100, 128)
    mask = torch.ones(2, 100, dtype=torch.bool)
    
    loss, perm, extras = pit_latent_mse_2spk(
        output, targets, targets, mask
    )
    print(f"\n  Loss computation: {loss.item():.4f}")
    print(f"  Permutation: {perm}")
    
    return wrapped_model


if __name__ == "__main__":
    print("Testing RWKV Separator Wrapper...")
    test_wrapper()
    print("\nWrapper test completed successfully!")
