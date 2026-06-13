"""GoLLM: GPT-style decoder-only transformer for Go code generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from llm_go.config import ModelConfig
from llm_go.model.attention import MultiHeadAttention


class RMSNorm(layers.Layer):
    """Root Mean Square Layer Normalisation (faster than LayerNorm, used in LLaMA/Mistral)."""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps

    def build(self, input_shape: Any) -> None:
        self.gamma = self.add_weight(
            name="gamma", shape=(self.dim,), initializer="ones", trainable=True
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x / rms)

    def get_config(self) -> dict:
        return {**super().get_config(), "dim": self.dim, "eps": self.eps}


class FeedForward(layers.Layer):
    """SwiGLU feed-forward block (used by PaLM, LLaMA, Mistral)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.gate_proj = layers.Dense(d_ff, use_bias=False, name="gate")
        self.up_proj   = layers.Dense(d_ff, use_bias=False, name="up")
        self.down_proj = layers.Dense(d_model, use_bias=False, name="down")
        self.dropout   = layers.Dropout(dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # SwiGLU: down(swish(gate(x)) * up(x))
        gate = tf.nn.silu(self.gate_proj(x))
        up   = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up, training=training))

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update({
            "d_model": self.down_proj.units,
            "d_ff":    self.gate_proj.units,
        })
        return cfg


class TransformerBlock(layers.Layer):
    """Single decoder block: pre-norm attention + pre-norm FFN."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.attn      = MultiHeadAttention(config, name="attn")
        self.ffn       = FeedForward(config.d_model, config.d_ff, config.dropout, name="ffn")
        self.norm1     = RMSNorm(config.d_model, config.layer_norm_eps, name="norm1")
        self.norm2     = RMSNorm(config.d_model, config.layer_norm_eps, name="norm2")
        self.drop_path = layers.Dropout(config.dropout)

    def call(
        self,
        x: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        # Pre-norm residual attention
        x = x + self.drop_path(
            self.attn(self.norm1(x), mask=mask, training=training),
            training=training,
        )
        # Pre-norm residual FFN
        x = x + self.drop_path(self.ffn(self.norm2(x), training=training), training=training)
        return x


class GoLLM(keras.Model):
    """
    Go-specialised large language model.

    Architecture: decoder-only GPT with:
      - Token + optional type embeddings
      - RoPE positional encoding (per attention layer)
      - RMSNorm + SwiGLU feed-forward
      - Tied input/output embeddings
      - Autoregressive causal masking
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        self.token_emb = layers.Embedding(
            config.vocab_size,
            config.d_model,
            embeddings_initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            name="token_emb",
        )
        self.emb_dropout = layers.Dropout(config.dropout)

        self.blocks = [
            TransformerBlock(config, name=f"block_{i}")
            for i in range(config.n_layers)
        ]

        self.norm_out = RMSNorm(config.d_model, config.layer_norm_eps, name="norm_out")

        # Output projection — weights tied to token_emb if requested
        if not config.tie_embeddings:
            self.lm_head = layers.Dense(config.vocab_size, use_bias=False, name="lm_head")
        else:
            self.lm_head = None  # handled in call() via matmul with emb weights

    def call(
        self,
        input_ids: tf.Tensor,
        mask: tf.Tensor | None = None,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Args:
            input_ids: [B, T] int32
            mask:      [B, 1, 1, T] float32 additive attention mask (0 keep, -inf ignore)
            training:  bool

        Returns:
            logits: [B, T, vocab_size]
        """
        x = self.token_emb(input_ids)                       # [B, T, D]
        x = self.emb_dropout(x, training=training)

        for block in self.blocks:
            x = block(x, mask=mask, training=training)

        x = self.norm_out(x)

        if self.lm_head is not None:
            return self.lm_head(x)

        # Tied embeddings: logits = x @ E^T
        emb_w = tf.cast(self.token_emb.embeddings, x.dtype)  # [V, D]
        return tf.matmul(x, emb_w, transpose_b=True)          # [B, T, V]

    @tf.function(reduce_retracing=True)
    def train_step_fn(
        self,
        input_ids: tf.Tensor,
        optimizer: keras.optimizers.Optimizer,
    ) -> dict[str, tf.Tensor]:
        """Single training step with gradient accumulation-ready signature."""
        labels = input_ids[:, 1:]       # next-token targets
        inputs = input_ids[:, :-1]

        with tf.GradientTape() as tape:
            logits = self(inputs, training=True)        # [B, T-1, V]
            loss = self._cross_entropy_loss(logits, labels)

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss, "perplexity": tf.exp(loss)}

    def _cross_entropy_loss(
        self, logits: tf.Tensor, labels: tf.Tensor
    ) -> tf.Tensor:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.cast(labels, tf.int32),
            logits=tf.cast(logits, tf.float32),
        )
        # Mask padding tokens (id=0)
        pad_mask = tf.cast(tf.not_equal(labels, self.config.pad_token_id), tf.float32)
        return tf.reduce_sum(loss * pad_mask) / tf.reduce_sum(pad_mask)

    @tf.function
    def generate(
        self,
        prompt_ids: tf.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> tf.Tensor:
        """Greedy/nucleus sampling with repetition penalty (no beam search)."""
        generated = prompt_ids  # [B, T]

        for _ in tf.range(max_new_tokens):
            # Truncate to max_seq_len
            ctx = generated[:, -self.config.max_seq_len:]
            logits = self(ctx, training=False)[:, -1, :]  # [B, V]

            # Repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)

            logits = logits / temperature

            # Top-k filter
            if top_k > 0:
                values, _ = tf.math.top_k(logits, k=top_k)
                threshold = values[:, -1:]
                logits = tf.where(logits < threshold, tf.fill(tf.shape(logits), -1e9), logits)

            # Top-p (nucleus) filter
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)

            probs = tf.nn.softmax(logits, axis=-1)
            next_id = tf.cast(
                tf.squeeze(tf.random.categorical(tf.math.log(probs), 1), axis=-1),
                tf.int32,
            )  # [B]
            generated = tf.concat([generated, next_id[:, tf.newaxis]], axis=1)

            # Stop on EOS
            if tf.reduce_all(tf.equal(next_id, self.config.eos_token_id)):
                break

        return generated

    def _apply_repetition_penalty(
        self, logits: tf.Tensor, generated: tf.Tensor, penalty: float
    ) -> tf.Tensor:
        # Penalise already-generated token ids
        b = tf.shape(generated)[0]
        for bi in tf.range(b):
            ids = generated[bi]
            scores = tf.gather(logits[bi], ids)
            scores = tf.where(scores > 0, scores / penalty, scores * penalty)
            logits = tf.tensor_scatter_nd_update(
                logits, tf.stack([tf.fill([tf.shape(ids)[0]], bi), ids], axis=1), scores
            )
        return logits

    def _top_p_filter(self, logits: tf.Tensor, top_p: float) -> tf.Tensor:
        probs = tf.nn.softmax(logits, axis=-1)
        sorted_probs = tf.sort(probs, direction="DESCENDING", axis=-1)
        cumprobs = tf.cumsum(sorted_probs, axis=-1, exclusive=True)
        sorted_indices = tf.argsort(probs, direction="DESCENDING", axis=-1)
        mask = tf.cast(cumprobs < top_p, probs.dtype)
        # Scatter mask back to original order
        inv = tf.argsort(sorted_indices, axis=-1)
        mask = tf.gather(mask, inv, batch_dims=1)
        logits = tf.where(mask == 0, tf.fill(tf.shape(logits), -1e9), logits)
        return logits

    def count_params(self) -> dict[str, int]:
        total = sum(np.prod(v.shape) for v in self.trainable_variables)
        emb   = np.prod(self.token_emb.embeddings.shape)
        return {"total": int(total), "embedding": int(emb), "non_embedding": int(total - emb)}

    def save_pretrained(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.config.save(d / "config.json")
        self.save_weights(str(d / "model.weights.h5"))

    @classmethod
    def from_pretrained(cls, directory: str | Path) -> "GoLLM":
        d = Path(directory)
        config = ModelConfig.load(d / "config.json")
        model = cls(config)
        # Build with dummy input so weights exist before loading
        dummy = tf.zeros([1, 8], dtype=tf.int32)
        model(dummy)
        model.load_weights(str(d / "model.weights.h5"))
        return model

    def get_config(self) -> dict:
        return {"config": self.config.to_dict()}

    @classmethod
    def from_config(cls, cfg: dict) -> "GoLLM":
        return cls(ModelConfig(**cfg["config"]))
