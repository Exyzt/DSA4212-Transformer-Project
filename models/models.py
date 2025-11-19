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

class DecoderBlock(nn.Module):
    """A single decoder block (Pre-LayerNorm + Self-Attn + MLP + residuals).

    Pre-LayerNorm improves training stability. Residual connections are used after
    attention and MLP sublayers. The attention is causal when a causal mask is passed
    (so each position can only attend to previous or current positions).

    Args:
      d_model: Hidden size D.
      n_heads: Number of attention heads.

    Input/Output shape: (B, T, D)
    """

    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    kernel_init: Any = nn.initializers.lecun_normal()
    compute_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, *, mask=None, deterministic: bool = True):
        # --- Self-Attention sublayer ---
        h = nn.LayerNorm(dtype=jnp.float32)(x)  # norms in fp32
        h = nn.SelfAttention(
            num_heads=self.n_heads,
            use_bias=False,
            kernel_init=self.kernel_init,
            dropout_rate=self.attn_dropout,
            dtype=self.compute_dtype,
        )(h, mask=mask, deterministic=deterministic)
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
        x = x + h  # residual
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
    max_len: int
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
        # Token embedding table E: (V, D)
        self.tok_embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.d_model,
            embedding_init=self.kernel_init,
            dtype=self.compute_dtype,
            param_dtype=self.param_dtype,
        )

        # Learned positional embeddings P: (max_len, D)
        self.positional_embed = self.param(
            "positional_embed",
            nn.initializers.normal(stddev=0.02),
            (self.max_len, self.d_model),
        )

        # Decoder stack
        self.blocks = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                attn_dropout=self.attn_dropout,
                mlp_dropout=self.mlp_dropout,
                kernel_init=self.kernel_init,
                compute_dtype=self.compute_dtype,
            ) for _ in range(self.n_layers)
        ]

        # Final norm (keep norms in fp32)
        self.layerNorm_final = nn.LayerNorm(dtype=jnp.float32)

        # Output head: (D -> V)
        self.project_to_vocab = nn.Dense(
            features=self.vocab_size,
            use_bias=False,
            kernel_init=self.proj_init,
            dtype=self.compute_dtype,
        )

    @nn.compact
    def __call__(self, idx, *, deterministic: bool = True):
        """
        idx: (B, T) token IDs
        returns logits: (B, T, V)
        """
        B, T = idx.shape

        # Token + positional embeddings  -> (B, T, D)
        x = self.tok_embed(idx).astype(self.compute_dtype) + self.positional_embed[:T]
        x = nn.Dropout(self.emb_dropout)(x, deterministic=deterministic)

        # Strictly causal mask (lower-triangular)
        mask = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))

        # Decoder stack
        for blk in self.blocks:
            x = blk(x, mask=mask, deterministic=deterministic)

        # Final norm and projection
        x = self.layerNorm_final(x)
        logits = self.project_to_vocab(x)  # (B, T, V)
        return logits