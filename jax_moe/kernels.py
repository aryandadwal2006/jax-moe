import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def selected_expert_mlp(x: jax.Array, expert_up: jax.Array, expert_down: jax.Array, expert_ids: jax.Array) -> jax.Array:
    selected_up = expert_up[expert_ids]
    selected_down = expert_down[expert_ids]
    hidden = jnp.einsum("th,thf->tf", x, selected_up)
    hidden = jax.nn.silu(hidden)
    return jnp.einsum("tf,tfh->th", hidden, selected_down)


def ragged_expert_mlp(x: jax.Array, expert_up: jax.Array, expert_down: jax.Array, expert_ids: jax.Array) -> jax.Array:
    order = jnp.argsort(expert_ids, stable=True)
    group_sizes = jnp.bincount(expert_ids, length=expert_up.shape[0]).astype(jnp.int32)
    sorted_x = x[order]
    hidden = jax.lax.ragged_dot(sorted_x, expert_up, group_sizes)
    hidden = jax.nn.silu(hidden)
    sorted_output = jax.lax.ragged_dot(hidden, expert_down, group_sizes)
    return sorted_output[jnp.argsort(order)]


def fused_expert_mlp(
    x: jax.Array,
    expert_up: jax.Array,
    expert_down: jax.Array,
    expert_ids: jax.Array,
    block_h: int = 32,
    block_f: int = 64,
) -> jax.Array:
    if x.ndim != 2 or expert_ids.ndim != 1:
        raise ValueError("x must be rank 2 and expert_ids must be rank 1")
    tokens, hidden_size = x.shape
    ffn_size = expert_up.shape[-1]
    if expert_up.shape != (expert_down.shape[0], hidden_size, ffn_size):
        raise ValueError("expert_up must have shape [experts, hidden_size, ffn_size]")
    if expert_down.shape[1:] != (ffn_size, hidden_size):
        raise ValueError("expert_down must have shape [experts, ffn_size, hidden_size]")
    if expert_ids.shape[0] != tokens:
        raise ValueError("expert_ids must contain one id per token")
    if block_h < 1 or block_f < 1 or hidden_size % block_h or ffn_size % block_f:
        raise ValueError("projection dimensions must be divisible by block_h and block_f")

    def kernel(x_ref, up_ref, down_ref, ids_ref, output_ref):
        token = pl.program_id(0)
        expert = ids_ref[token]
        intermediate = jnp.zeros((ffn_size,), dtype=jnp.float32)
        for start in range(0, hidden_size, block_h):
            x_block = x_ref[token, start:start + block_h].astype(jnp.float32)
            up_block = up_ref[expert, start:start + block_h, :].astype(jnp.float32)
            intermediate = intermediate + jnp.sum(x_block[:, None] * up_block, axis=0)
        intermediate = jax.nn.silu(intermediate)
        result = jnp.zeros((hidden_size,), dtype=jnp.float32)
        for start in range(0, ffn_size, block_f):
            ffn_block = intermediate[start:start + block_f]
            down_block = down_ref[expert, start:start + block_f, :].astype(jnp.float32)
            result = result + jnp.sum(ffn_block[:, None] * down_block, axis=0)
        output_ref[token, :] = result.astype(x_ref.dtype)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden_size), x.dtype),
        grid=(tokens,),
    )(x, expert_up, expert_down, expert_ids)
