import time

import jax
import jax.numpy as jnp

from jax_moe.kernels import fused_expert_mlp, selected_expert_mlp


def _measure(function, repeats: int) -> float:
    result = function()
    result.block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        result = function()
        result.block_until_ready()
    return (time.perf_counter() - start) / repeats


def benchmark_fused(tokens: int = 128, hidden_size: int = 128, ffn_size: int = 512, experts: int = 4, repeats: int = 50, seed: int = 0):
    key_x, key_up, key_down, key_ids = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = jax.random.normal(key_x, (tokens, hidden_size), dtype=jnp.float32)
    up = jax.random.normal(key_up, (experts, hidden_size, ffn_size), dtype=jnp.float32)
    down = jax.random.normal(key_down, (experts, ffn_size, hidden_size), dtype=jnp.float32)
    expert_ids = jax.random.randint(key_ids, (tokens,), 0, experts)
    reference = lambda: selected_expert_mlp(x, up, down, expert_ids)
    fused = lambda: fused_expert_mlp(x, up, down, expert_ids)
    reference_time = _measure(reference, repeats)
    fused_time = _measure(fused, repeats)
    return {
        "reference_ms": reference_time * 1000,
        "fused_ms": fused_time * 1000,
        "speedup": reference_time / fused_time,
    }


if __name__ == "__main__":
    print(benchmark_fused())
