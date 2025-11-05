import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import math

class RWKVv7AudioSeparator(nn.Module):
    """
    Enhanced speech separator using proven RWKV-v7 architecture for DAC codec embeddings
    Input: [B, T, C] where C is DAC latent dimension (e.g., 1024)
    Output: Separated sources with same dimensions
    """
    
    def __init__(self, 
                 in_dim: int = 1024,           # DAC latent dimension
                 hidden_dim: int = 1024,       # Model dimension
                 n_layers: int = 12,           # Number of RWKV layers
                 n_head: int = 16,             # Number of attention heads
                 head_size: int = 64,          # Head size (must divide hidden_dim)
                 dropout: float = 0.1,
                 num_sources: int = 2):        # Number of sources to separate
        
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_sources = num_sources
        
        # Ensure hidden_dim is compatible with head_size
        assert hidden_dim % head_size == 0, "hidden_dim must be divisible by head_size"
        
        # Input projection to match hidden dimension
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # Create RWKV blocks with proper configuration
        self.blocks = nn.ModuleList([
            RWKVAudioBlock(hidden_dim, n_head, head_size, dropout, layer_id=i)
            for i in range(n_layers)
        ])
        
        # Layer norm after blocks
        self.ln_out = nn.LayerNorm(hidden_dim)
        
        # Enhanced separation heads with frequency-aware processing
        self.separation_heads = nn.ModuleList([
            SeparationHead(hidden_dim, in_dim, n_head)
            for _ in range(num_sources)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization from proven RWKV-v7"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, T, C] DAC codec embeddings
        Returns:
            Dict with separated sources and optional masks
        """
        B, T, C = x.shape
        
        # Input projection
        h = self.input_proj(x)  # [B, T, hidden_dim]
        
        # Store v_first for RWKV blocks
        v_first = None
        
        # Process through RWKV blocks
        for block in self.blocks:
            h, v_first = block(h, v_first)
        
        # Final layer norm
        h = self.ln_out(h)
        
        # Generate separated sources
        outputs = {}
        for i, head in enumerate(self.separation_heads):
            source_out = head(h, x)  # Each head produces one source
            outputs[f'source_{i+1}'] = source_out
        
        return outputs


class RWKVAudioBlock(nn.Module):
    """
    Adapted RWKV block for audio processing with proper bidirectional handling
    """
    
    def __init__(self, dim: int, n_head: int, head_size: int, dropout: float, layer_id: int):
        super().__init__()
        
        self.dim = dim
        self.n_head = n_head
        self.layer_id = layer_id
        
        # Layer norms
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
        # Time mixing (attention) with bidirectional capability
        self.tmix = TimeMixBiDirectional(dim, n_head, head_size, layer_id)
        
        # Channel mixing (FFN)
        self.cmix = ChannelMix(dim, layer_id)
        
        # Drop path for regularization
        self.drop_path = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor, v_first: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Time mixing with residual
        attn_out, v_first = self.tmix(self.ln1(x), v_first)
        x = x + self.drop_path(attn_out)
        
        # Channel mixing with residual
        ffn_out = self.cmix(self.ln2(x))
        x = x + self.drop_path(ffn_out)
        
        return x, v_first


class TimeMixBiDirectional(nn.Module):
    """
    Time mixing layer with integrated bidirectional processing from proven RWKV-v7
    """
    
    def __init__(self, dim: int, n_head: int, head_size: int, layer_id: int):
        super().__init__()
        
        self.dim = dim
        self.n_head = n_head
        self.head_size = head_size
        self.layer_id = layer_id
        
        # Learnable parameters for time shift
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        # Projections
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        
        # Gating for bidirectional fusion
        self.gate = nn.Linear(dim, n_head, bias=False)
        
        # Layer normalization
        self.ln_x = nn.GroupNorm(n_head, dim, eps=64e-5)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # RWKV-style initialization
        nn.init.uniform_(self.receptance.weight, -0.5/math.sqrt(self.dim), 0.5/math.sqrt(self.dim))
        nn.init.uniform_(self.key.weight, -0.05/math.sqrt(self.dim), 0.05/math.sqrt(self.dim))
        nn.init.uniform_(self.value.weight, -0.5/math.sqrt(self.dim), 0.5/math.sqrt(self.dim))
        nn.init.zeros_(self.output.weight)
    
    def forward(self, x: torch.Tensor, v_first: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, C = x.shape
        
        # Time shift operation
        xx = self.time_shift(x) - x
        
        # Projections
        r = self.receptance(x + xx)  # Receptance
        k = self.key(x + xx)         # Key  
        v = self.value(x + xx)       # Value
        
        # Store v_first for first layer
        if self.layer_id == 0:
            v_first = v
        
        # Reshape for multi-head processing
        r = r.view(B, T, self.n_head, self.head_size)
        k = k.view(B, T, self.n_head, self.head_size) 
        v = v.view(B, T, self.n_head, self.head_size)
        
        # Bidirectional processing using the proven approach
        # Forward pass
        x_forward = self._wkv_forward(r, k, v)
        
        # Backward pass (flip sequence)
        r_backward = torch.flip(r, dims=[1])
        k_backward = torch.flip(k, dims=[1])
        v_backward = torch.flip(v, dims=[1])
        x_backward = self._wkv_forward(r_backward, k_backward, v_backward)
        x_backward = torch.flip(x_backward, dims=[1])
        
        # Learnable gating fusion
        gate_weights = torch.sigmoid(self.gate(xx)).unsqueeze(-1)  # [B, T, n_head, 1]
        x_combined = gate_weights * x_forward + (1 - gate_weights) * x_backward
        
        # Output projection
        x_out = self.output(x_combined.reshape(B, T, C))
        
        return x_out, v_first
    
    def _wkv_forward(self, r: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Simplified WKV computation - in practice you'd use the optimized CUDA kernel"""
        B, T, H, N = r.shape
        
        # This is a simplified version - use the actual RWKV WKV computation from proven script
        wkv = torch.zeros_like(v)
        
        # For now, using a simple causal attention approximation
        # In practice, replace with the actual RWKV WKV computation
        for t in range(T):
            if t == 0:
                wkv[:, t] = v[:, t]
            else:
                wkv[:, t] = (wkv[:, t-1] + v[:, t]) / 2  # Simplified
        
        return r * wkv


class ChannelMix(nn.Module):
    """Channel mixing from proven RWKV-v7"""
    
    def __init__(self, dim: int, layer_id: int):
        super().__init__()
        
        self.dim = dim
        self.layer_id = layer_id
        
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        # Key-value projection with expansion
        self.key = nn.Linear(dim, dim * 4, bias=False)
        self.value = nn.Linear(dim * 4, dim, bias=False)
        
        # Initialize weights
        nn.init.uniform_(self.key.weight, -0.5/math.sqrt(dim), 0.5/math.sqrt(dim))
        nn.init.zeros_(self.value.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xx = self.time_shift(x) - x
        k = x + xx  # Time-shifted features
        
        k = torch.relu(self.key(k)) ** 2  # Squared ReLU activation
        return self.value(k)


class SeparationHead(nn.Module):
    """Enhanced separation head with frequency-aware processing"""
    
    def __init__(self, hidden_dim: int, output_dim: int, n_groups: int = 8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_groups = n_groups
        
        # Multi-scale processing
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=n_groups)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=n_groups)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=n_groups)
        
        # Gated fusion
        self.fusion_gate = nn.Linear(hidden_dim * 3, 3)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Residual scale
        self.residual_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, features: torch.Tensor, input_mix: torch.Tensor) -> torch.Tensor:
        B, T, C = features.shape
        
        # Transpose for conv processing
        features_t = features.transpose(1, 2)  # [B, C, T]
        
        # Multi-scale convolution
        conv1_out = self.conv1(features_t).transpose(1, 2)  # [B, T, C]
        conv2_out = self.conv2(features_t).transpose(1, 2)
        conv3_out = self.conv3(features_t).transpose(1, 2)
        
        # Gated fusion
        fused = torch.cat([conv1_out, conv2_out, conv3_out], dim=-1)
        gate_weights = torch.softmax(self.fusion_gate(fused), dim=-1)  # [B, T, 3]
        
        # Weighted combination
        combined = (gate_weights[..., 0:1] * conv1_out + 
                   gate_weights[..., 1:2] * conv2_out + 
                   gate_weights[..., 2:3] * conv3_out)
        
        # Output projection with residual connection
        output = self.output_proj(combined)
        
        # Residual connection to input mix
        output = input_mix + self.residual_scale * output
        
        return output


# Usage example
def create_separator_model():
    """Create the enhanced RWKV-v7 separator"""
    model = RWKVv7AudioSeparator(
        in_dim=1024,      # DAC latent dimension
        hidden_dim=1024,  # Model dimension
        n_layers=12,      # Number of layers
        n_head=16,        # Number of heads
        head_size=64,     # Head size
        dropout=0.1,
        num_sources=2     # Separate into 2 sources
    )
    return model

# Test the model
if __name__ == "__main__":
    model = create_separator_model()
    
    # Test with DAC embeddings [batch, time, channels]
    test_input = torch.randn(2, 512, 1024)  # [B, T, C]
    
    with torch.no_grad():
        outputs = model(test_input)
    
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")