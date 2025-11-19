"""
Minimal decoder-only Transformer blocks in Flax/JAX, commented for learning.

The model mirrors a GPT-style architecture:
- Token embeddings + learned positional embeddings
- Stack of Pre-LayerNorm decoder blocks with causal self-attention
- Final LayerNorm
- Weight tying between input embeddings and output logits projection

Tensor shape conventions used below:
- B: batch size
- T: sequence length (time/positions)
- D: hidden size / embedding dimension (d_model)
- V: vocabulary size
"""

import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn
from typing import Any, Callable

class MLP(nn.Module):
        """Transformer feed-forward network (a.k.a. MLP block).

        Structure: Dense(D -> 4D), GELU, Dense(4D -> D) by default.
        The expansion factor can be adjusted with `mlp_ratio`.

        Args:
            d_model: Hidden size D.
            mlp_ratio: Expansion factor for the intermediate hidden size.

        Input shape:  (B, T, D)
        Output shape: (B, T, D)
        """

        d_model: int
        mlp_ratio: int = 4
        mlp_dropout: float = 0.0
        kernel_init: Any = nn.initializers.lecun_normal()
        compute_dtype: Any = jnp.float32

        @nn.compact
        def __call__(self, x, *, deterministic: bool = True):
            hidden = int(self.d_model * self.mlp_ratio)  # e.g., 4*D
            x = nn.Dense(
                hidden,
                kernel_init=self.kernel_init,
                dtype=self.compute_dtype,
            )(x)
            x = nn.gelu(x)
            x = nn.Dropout(self.mlp_dropout)(x, deterministic=deterministic)
            x = nn.Dense(
                self.d_model,
                kernel_init=self.kernel_init,
                dtype=self.compute_dtype,
            )(x)
            # Optionally a second dropout after the projection (common in some impls):
            # x = nn.Dropout(self.mlp_dropout)(x, deterministic=deterministic)
            return x
        
def apply_rotary_pos_emb(x, positions):
    """
    Apply Rotary Position Embeddings (RoPE) to input tensor.
    
    Args:
        x: Input tensor of shape (B, T, n_heads, head_dim)
        positions: Position indices, shape (T,)
    
    Returns:
        Tensor with same shape as x with RoPE applied
    """
    # Get dimensions
    *batch_dims, seq_len, dim = x.shape
    head_dim = dim
    
    # Create frequency bands: theta_i = 10000^(-2i/d) for i in [0, d/2)
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2).astype(jnp.float32) / head_dim))
    
    # Compute position * frequency for each position
    # positions shape: (T,) -> (T, 1)
    # inv_freq shape: (head_dim/2,) -> (1, head_dim/2)
    # Result: (T, head_dim/2)
    freqs = jnp.outer(positions.astype(jnp.float32), inv_freq)
    
    # Create cos and sin components: (T, head_dim/2)
    cos_emb = jnp.cos(freqs)
    sin_emb = jnp.sin(freqs)
    
    # Split x into first half and second half along the last dimension
    # x shape: (B, T, n_heads, head_dim)
    # x1, x2 shape: (B, T, n_heads, head_dim/2)
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]
    
    # Add dimensions to cos/sin for broadcasting
    # Need to broadcast (T, head_dim/2) to (B, T, n_heads, head_dim/2)
    # Add dimensions for batch and heads: (T, head_dim/2) -> (1, T, 1, head_dim/2)
    cos_emb = cos_emb[None, :, None, :]  # (1, T, 1, head_dim/2)
    sin_emb = sin_emb[None, :, None, :]  # (1, T, 1, head_dim/2)
    
    # Apply rotation:
    # [x1_rotated]   [cos  -sin] [x1]
    # [x2_rotated] = [sin   cos] [x2]
    # This is equivalent to: x1*cos - x2*sin, x1*sin + x2*cos
    rotated = jnp.concatenate([
        x1 * cos_emb - x2 * sin_emb,
        x1 * sin_emb + x2 * cos_emb
    ], axis=-1)
    
    return rotated

class RoPEDecoderBlock(nn.Module):
    """Decoder block with RoPE applied in self-attention."""
    
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    kernel_init: Any = nn.initializers.lecun_normal()
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, positions, *, mask=None, deterministic: bool = True):
        B, T, D = x.shape
        head_dim = D // self.n_heads
        
        # --- Self-Attention with RoPE ---
        h = nn.LayerNorm(dtype=jnp.float32)(x)
        
        # QKV projections
        qkv = nn.Dense(
            3 * self.d_model,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.compute_dtype,
        )(h)
        
        # Split into Q, K, V and reshape for multi-head attention
        # (B, T, 3*D) -> 3 x (B, T, n_heads, head_dim)
        qkv = qkv.reshape(B, T, 3, self.n_heads, head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)  # (B, T, n_heads, head_dim) each
        
        # Apply RoPE to queries and keys
        q = apply_rotary_pos_emb(q, positions)
        k = apply_rotary_pos_emb(k, positions)
        
        # Standard attention computation
        # Move head dimension: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))
        
        # Attention scores: (B, n_heads, T, T)
        scale = 1.0 / jnp.sqrt(head_dim)
        attn_weights = jnp.einsum('bhqd,bhkd->bhqk', q, k) * scale
        
        # Apply causal mask
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e10)
        
        attn_weights = nn.softmax(attn_weights, axis=-1)
        attn_weights = nn.Dropout(self.attn_dropout)(attn_weights, deterministic=deterministic)
        
        # Apply attention to values: (B, n_heads, T, head_dim)
        h = jnp.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, D)
        h = jnp.transpose(h, (0, 2, 1, 3)).reshape(B, T, D)
        
        # Output projection
        h = nn.Dense(
            self.d_model,
            use_bias=False,
            kernel_init=self.kernel_init,
            dtype=self.compute_dtype,
        )(h)
        
        x = x + h  # residual
        
        # --- MLP sublayer ---
        h = nn.LayerNorm(dtype=jnp.float32)(x)
        h = MLP(
            d_model=self.d_model,
            mlp_ratio=self.mlp_ratio,
            mlp_dropout=self.mlp_dropout,
            kernel_init=self.kernel_init,
            compute_dtype=self.compute_dtype,
        )(h, deterministic=deterministic)
        x = x + h
        return x

class DecoderOnlyTransformer(nn.Module):
    """GPT-style decoder-only Transformer for language modeling.

    Components:
      - Token embeddings: maps token ids to D-dim vectors
      - Learned positional embeddings: adds position information (0..T-1)
      - N stacked decoder blocks with causal self-attention
      - Final LayerNorm
      - Output projection:
          * If tie_weights=True (default), reuse token embedding matrix E to
            compute logits via x @ E^T (implemented via einsum).
          * Else, use a separate linear head to project to V logits.

    Args:
      vocab_size: Vocabulary size V.
      d_model: Hidden size D.
      n_layers: Number of decoder blocks.
      n_heads: Attention heads per block.
      max_len: Maximum supported sequence length for positional embeddings.
    """

    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    mlp_ratio: int = 4

    # --- knobs ---
    emb_dropout: float = 0.0
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    param_dtype: Any = jnp.float32
    compute_dtype: Any = jnp.float32

    # initializers
    kernel_init: Callable = nn.initializers.lecun_normal()
    proj_init:   Callable = nn.initializers.normal(stddev=1e-4)

    def setup(self):
        # Token embedding (same as before)
        self.tok_embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=self.kernel_init,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

        # Decoder stack with RoPE
        self.blocks = [
            RoPEDecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                attn_dropout=self.attn_dropout,
                mlp_dropout=self.mlp_dropout,
                kernel_init=self.kernel_init,
                compute_dtype=self.compute_dtype,
            ) for _ in range(self.n_layers)
        ]

        # Final norm and projection (same as before)
        self.layerNorm_final = nn.LayerNorm(dtype=jnp.float32)
        self.project_to_vocab = nn.Dense(
            features=self.vocab_size,
            use_bias=False,
            kernel_init=self.proj_init,
            dtype=self.compute_dtype,
        )

    @nn.compact
    def __call__(self, idx, *, deterministic: bool = True):
        B, T = idx.shape

        # Only token embeddings, no positional embeddings
        x = self.tok_embed(idx).astype(self.compute_dtype)
        x = nn.Dropout(self.emb_dropout)(x, deterministic=deterministic)

        # Create position indices for RoPE
        positions = jnp.arange(T)  # (T,)

        # Causal mask
        mask = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))

        # Decoder stack - now pass positions to each block
        for blk in self.blocks:
            x = blk(x, positions, mask=mask, deterministic=deterministic)

        # Final norm and projection
        x = self.layerNorm_final(x)
        logits = self.project_to_vocab(x)
        return logits