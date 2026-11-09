import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def selected_expert_mlp(x: jax.Array, expert_up: jax.Array, expert_down: jax.Array, expert_ids: jax.Array) -> jax.Array:
    selected_up = expert_up[expert_ids]
    selected_down = expert_down[expert_ids]
    hidden = jnp.einsum("th,thf->tf", x, selected_up)
    hidden = jax.nn.silu(hidden)
    return jnp.einsum("tf,tfh->th", hidden, selected_down)


def fused_expert_mlp(
    x: jax.Array,
    expert_up: jax.Array,
    expert_down: jax.Array,
    expert_ids: jax.Array,
    block_out: int = 32,
) -> jax.Array:
    if x.ndim != 2 or expert_ids.ndim != 1:
        raise ValueError("x must be rank 2 and expert_ids must be rank 1")
    if block_out < 1 or block_out & (block_out - 1):
        raise ValueError("block_out must be a power of two")
    tokens, hidden_size = x.shape
    ffn_size = expert_up.shape[-1]
    if expert_up.shape != (expert_down.shape[0], hidden_size, ffn_size):
        raise ValueError("expert_up must have shape [experts, hidden_size, ffn_size]")
    if expert_down.shape[1:] != (ffn_size, hidden_size):
        raise ValueError("expert_down must have shape [experts, ffn_size, hidden_size]")
    if hidden_size & (hidden_size - 1) or ffn_size & (ffn_size - 1):
        raise ValueError("hidden_size and ffn_size must be powers of two")
    if expert_ids.shape[0] != tokens:
        raise ValueError("expert_ids must contain one id per token")

    def kernel(x_ref, up_ref, down_ref, ids_ref, output_ref):
        token = pl.program_id(0)
        output_start = pl.program_id(1) * block_out
        h_idx = pl.arange(0, hidden_size)
        f_idx = pl.arange(0, ffn_size)
        o_idx = output_start + pl.arange(0, block_out)
        expert = pl.load(ids_ref + token)
        x_row = pl.load(x_ref + token * hidden_size + h_idx).astype(jnp.float32)
        up = pl.load(
            up_ref + expert * hidden_size * ffn_size + h_idx[:, None] * ffn_size + f_idx[None, :]
        ).astype(jnp.float32)
        intermediate = pl.sum(x_row[:, None] * up, axis=0)
        intermediate = intermediate / (1.0 + jnp.exp(-intermediate))
        down = pl.load(
            down_ref + expert * ffn_size * hidden_size + f_idx[:, None] * hidden_size + o_idx[None, :],
            mask=o_idx[None, :] < hidden_size,
            other=0,
        ).astype(jnp.float32)
        result = pl.sum(intermediate[:, None] * down, axis=0)
        pl.store(output_ref + token * hidden_size + o_idx, result, mask=o_idx < hidden_size)

    grid = (tokens, (hidden_size + block_out - 1) // block_out)
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tokens, hidden_size), x.dtype),
        grid=grid,
    )(x, expert_up, expert_down, expert_ids)
