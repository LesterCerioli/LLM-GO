import pytest
import numpy as np
import tensorflow as tf

from llm_go.config import ModelConfig
from llm_go.model.attention import RotaryEmbedding, MultiHeadAttention
from llm_go.model.transformer import GoLLM, TransformerBlock, RMSNorm, FeedForward


@pytest.fixture
def tiny_config():
    return ModelConfig(
        vocab_size=256,
        max_seq_len=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.0,
        attention_dropout=0.0,
    )


class TestRMSNorm:
    def test_output_shape(self, tiny_config):
        norm = RMSNorm(tiny_config.d_model)
        x = tf.random.normal([2, 8, tiny_config.d_model])
        y = norm(x)
        assert y.shape == x.shape

    def test_normalises(self, tiny_config):
        norm = RMSNorm(tiny_config.d_model)
        x = tf.random.normal([1, 4, tiny_config.d_model]) * 100
        y = norm(x)
        # RMS of output should be close to 1 (scaled by gamma=1)
        rms = tf.sqrt(tf.reduce_mean(tf.square(y)))
        assert abs(float(rms) - 1.0) < 0.2


class TestRotaryEmbedding:
    def test_shape_preserved(self, tiny_config):
        rope = RotaryEmbedding(dim=tiny_config.d_model // tiny_config.n_heads,
                               max_seq_len=tiny_config.max_seq_len)
        # [B, H, T, Dh]
        x = tf.random.normal([2, tiny_config.n_heads, 8,
                               tiny_config.d_model // tiny_config.n_heads])
        y = rope(x)
        assert y.shape == x.shape

    def test_different_positions_differ(self, tiny_config):
        dh = tiny_config.d_model // tiny_config.n_heads
        rope = RotaryEmbedding(dim=dh, max_seq_len=tiny_config.max_seq_len)
        x = tf.ones([1, 1, 10, dh])
        y = rope(x)
        # Positions should differ
        assert not np.allclose(y[:, :, 0, :].numpy(), y[:, :, 5, :].numpy())


class TestMultiHeadAttention:
    def test_output_shape(self, tiny_config):
        attn = MultiHeadAttention(tiny_config)
        x    = tf.random.normal([2, 8, tiny_config.d_model])
        y    = attn(x, training=False)
        assert y.shape == (2, 8, tiny_config.d_model)

    def test_causal_masking(self, tiny_config):
        """Output at position i must not depend on positions > i."""
        attn = MultiHeadAttention(tiny_config)
        x    = tf.random.normal([1, 4, tiny_config.d_model])

        with tf.GradientTape() as tape:
            tape.watch(x)
            y = attn(x, training=False)

        grads = tape.jacobian(y[0, 0], x[0])  # [D_out, T, D_in]
        # Position 0 output should have zero gradient from positions 1,2,3
        future_grads = grads[:, 1:, :]
        assert np.allclose(future_grads.numpy(), 0, atol=1e-6)


class TestTransformerBlock:
    def test_residual_shape(self, tiny_config):
        block = TransformerBlock(tiny_config)
        x     = tf.random.normal([2, 8, tiny_config.d_model])
        y     = block(x, training=False)
        assert y.shape == x.shape


class TestGoLLM:
    def test_forward_pass_shape(self, tiny_config):
        model = GoLLM(tiny_config)
        ids   = tf.random.uniform([2, 16], minval=0, maxval=tiny_config.vocab_size, dtype=tf.int32)
        logits = model(ids, training=False)
        assert logits.shape == (2, 16, tiny_config.vocab_size)

    def test_param_count_nonzero(self, tiny_config):
        model = GoLLM(tiny_config)
        ids   = tf.zeros([1, 4], dtype=tf.int32)
        model(ids)
        counts = model.count_params()
        assert counts["total"] > 0
        assert counts["non_embedding"] > 0

    def test_save_load_roundtrip(self, tiny_config, tmp_path):
        model = GoLLM(tiny_config)
        ids   = tf.zeros([1, 4], dtype=tf.int32)
        logits_before = model(ids)
        model.save_pretrained(str(tmp_path / "ckpt"))

        loaded = GoLLM.from_pretrained(str(tmp_path / "ckpt"))
        logits_after = loaded(ids)
        np.testing.assert_allclose(
            logits_before.numpy(), logits_after.numpy(), atol=1e-5
        )

    def test_generation_length(self, tiny_config):
        model = GoLLM(tiny_config)
        prompt = tf.constant([[1, 10, 20]], dtype=tf.int32)
        output = model.generate(prompt, max_new_tokens=5, temperature=1.0, top_k=0, top_p=1.0)
        assert output.shape[1] <= 3 + 5 + 1   # prompt + max_new + possible EOS
