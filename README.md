# JAX Mixture-of-Experts Transformer

A Flax decoder-only Transformer for multi-digit addition with dense and top-1 MoE feed-forward layers.

- `data.py` generates addition sequences from a restricted digit/operator vocabulary.
- `model.py` implements causal attention, dense MLPs, and token-routed experts.
- `train.py` trains either architecture with Optax and masked next-token loss.
- `scaling.py` exports parameter and active-compute grids using a Chinchilla-style `6NT` estimate.
- `kernels.py` fuses expert up-projection, SiLU, and down-projection in a Pallas kernel.
- `benchmark.py` compares the Pallas kernel with a `jax.lax.ragged_dot` baseline.

Install JAX for the selected CPU, CUDA, or TPU runtime, then install `requirements.txt`.

`python -m jax_moe.train --architecture moe`
`python -m jax_moe.scaling --output scaling.csv`
