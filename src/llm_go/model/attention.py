"""Multi-head self-attention with Rotary Positional Embeddings (RoPE)."""

from __future__ import annotations

import math

import tensorflow as tf
import keras
from keras import layers

from llm_go.config import ModelConfig


class RotaryEmbedding(layers.Layer):
    """Rotary Position Embedding (RoPE) — Su et al. 2023."""

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10_000.0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute frequency matrix [max_seq_len, dim/2]
        freqs = self._compute_freqs(max_seq_len)
        # Store as non-trainable weight for serialisation
        self._freqs = self.add_weight(
            name="rope_freqs",
            shape=freqs.shape,
            initializer=keras.initializers.Constant(freqs.numpy()),
            trainable=False,
        )

    def _compute_freqs(self, seq_len: int) -> tf.Tensor:
        half = self.dim // 2
        inv_freq = 1.0 / (
            self.theta ** (tf.cast(tf.range(0, half, 2), tf.float32) / half)
        )
        positions = tf.cast(tf.range(seq_len), tf.float32)
        # [seq_len, half//2]
        freqs = tf.einsum("i,j->ij", positions, inv_freq)
        # [seq_len, half]
        return tf.concat([freqs, freqs], axis=-1)

    def _rotate_half(self, x: tf.Tensor) -> tf.Tensor:
        half = tf.shape(x)[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return tf.concat([-x2, x1], axis=-1)

    def call(self, x: tf.Tensor, seq_len: int | None = None) -> tf.Tensor:
        """Apply RoPE to query or key tensor of shape [B, H, T, D]."""
        t = tf.shape(x)[2] if seq_len is None else seq_len
        freqs = self._freqs[:t]                         # [T, D]
        cos = tf.cos(freqs)[tf.newaxis, tf.newaxis]     # [1, 1, T, D]
        sin = tf.sin(freqs)[tf.newaxis, tf.newaxis]
        return x * cos + self._rotate_half(x) * sin

    def get_config(self) -> dict:
        return {**super().get_config(), "dim": self.dim,
                "max_seq_len": self.max_seq_len, "theta": self.theta}


class MultiHeadAttention(layers.Layer):
    """Causal multi-head self-attention with RoPE and optional dropout."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.attn_drop = config.attention_dropout

        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = layers.Dense(config.d_model, use_bias=False, name="q")
        self.k_proj = layers.Dense(config.d_model, use_bias=False, name="k")
        self.v_proj = layers.Dense(config.d_model, use_bias=False, name="v")
        self.out_proj = layers.Dense(config.d_model, use_bias=False, name="out")

        self.rope = RotaryEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            name="rope",
        )
        self.dropout = layers.Dropout(config.attention_dropout)

    def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
        """[B, T, D] → [B, H, T, D/H]"""
        b, t, _ = tf.unstack(tf.shape(x)[:3])
        x = tf.reshape(x, [b, t, self.n_heads, self.head_dim])
        return tf.transpose(x, [0, 2, 1, 3])

    def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
        """[B, H, T, D/H] → [B, T, D]"""
        b, h, t, d = tf.unstack(tf.shape(x))
        x = tf.transpose(x, [0, 2, 1, 3])
        return tf.reshape(x, [b, t, h * d])

    def _causal_mask(self, seq_len: tf.Tensor) -> tf.Tensor:
        """Upper-triangular mask: -inf for future positions."""
        mask = 1 - tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
        return mask * -1e9

    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        t = tf.shape(x)[1]

        q = self._split_heads(self.q_proj(x))  # [B, H, T, Dh]
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        q = self.rope(q)
        k = self.rope(k)

        # Scaled dot-product attention
        scores = tf.matmul(q, k, transpose_b=True) * self.scale  # [B, H, T, T]
        scores += self._causal_mask(t)

        if mask is not None:
            # mask: [B, 1, 1, T], 0=keep -inf=ignore
            scores += mask

        weights = tf.nn.softmax(scores, axis=-1)
        weights = self.dropout(weights, training=training)

        context = tf.matmul(weights, v)           # [B, H, T, Dh]
        context = self._merge_heads(context)       # [B, T, D]
        return self.out_proj(context)

    def get_config(self) -> dict:
        # Store only serialisable primitives; ModelConfig is reconstructed externally
        return {
            **super().get_config(),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "attn_drop": self.attn_drop,
        }
