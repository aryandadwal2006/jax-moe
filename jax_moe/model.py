from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import linen as nn


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 13
    max_length: int = 32
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    num_experts: int = 4
    mlp_ratio: int = 4

    @property
    def ffn_size(self) -> int:
        return self.hidden_size * self.mlp_ratio


class CausalSelfAttention(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        batch, length, hidden = x.shape
        if hidden % self.config.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        head_dim = hidden // self.config.num_heads
        qkv = nn.Dense(3 * hidden, use_bias=False, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(batch, length, self.config.num_heads, head_dim)
        k = k.reshape(batch, length, self.config.num_heads, head_dim)
        v = v.reshape(batch, length, self.config.num_heads, head_dim)
        scores = jnp.einsum("bthd,bshd->bhts", q, k) * (head_dim ** -0.5)
        causal = jnp.tril(jnp.ones((length, length), dtype=jnp.bool_))
        scores = jnp.where(causal[None, None, :, :], scores, jnp.finfo(scores.dtype).min)
        weights = jax.nn.softmax(scores, axis=-1)
        attended = jnp.einsum("bhts,bshd->bthd", weights, v)
        attended = attended.reshape(batch, length, hidden)
        return nn.Dense(hidden, name="out")(attended)


class DenseFeedForward(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.config.ffn_size, name="up")(x)
        x = nn.silu(x)
        return nn.Dense(self.config.hidden_size, name="down")(x)


class Top1MoE(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        hidden = self.config.hidden_size
        ffn = self.config.ffn_size
        router_logits = nn.Dense(self.config.num_experts, use_bias=False, name="router")(x)
        expert_ids = jnp.argmax(router_logits, axis=-1)
        gates = jax.nn.softmax(router_logits, axis=-1)
        gate = jnp.take_along_axis(gates, expert_ids[..., None], axis=-1)
        up = self.param("expert_up", nn.initializers.xavier_uniform(), (self.config.num_experts, hidden, ffn))
        down = self.param("expert_down", nn.initializers.xavier_uniform(), (self.config.num_experts, ffn, hidden))
        selected_up = up[expert_ids]
        selected_down = down[expert_ids]
        intermediate = jnp.einsum("bth,bthf->btf", x, selected_up)
        intermediate = nn.silu(intermediate)
        output = jnp.einsum("btf,btfh->bth", intermediate, selected_down)
        return output * gate


class TransformerBlock(nn.Module):
    config: ModelConfig
    use_moe: bool

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        attended = CausalSelfAttention(self.config, name="attention")(nn.LayerNorm(name="attention_norm")(x))
        x = x + attended
        normalized = nn.LayerNorm(name="ffn_norm")(x)
        if self.use_moe:
            feed_forward = Top1MoE(self.config, name="moe")(normalized)
        else:
            feed_forward = DenseFeedForward(self.config, name="mlp")(normalized)
        return x + feed_forward


class TransformerLM(nn.Module):
    config: ModelConfig
    use_moe: bool = True

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        length = tokens.shape[1]
        if length > self.config.max_length:
            raise ValueError("input sequence exceeds max_length")
        x = nn.Embed(self.config.vocab_size, self.config.hidden_size, name="token_embedding")(tokens)
        positions = self.param("position_embedding", nn.initializers.normal(0.02), (self.config.max_length, self.config.hidden_size))
        x = x + positions[None, :length, :]
        for index in range(self.config.num_layers):
            x = TransformerBlock(self.config, self.use_moe, name=f"block_{index}")(x)
        x = nn.LayerNorm(name="final_norm")(x)
        return nn.Dense(self.config.vocab_size, name="lm_head")(x)
