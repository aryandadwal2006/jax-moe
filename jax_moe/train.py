import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training import train_state

from jax_moe.data import PAD_ID, make_dataset
from jax_moe.model import ModelConfig, TransformerLM


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    dataset = make_dataset(args.samples, args.max_digits, args.seed)
    sequence_length = dataset.shape[1] - 1
    config = ModelConfig(
        max_length=sequence_length,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
    )
    model = TransformerLM(config, use_moe=args.architecture == "moe")
    sample = jnp.zeros((args.batch_size, sequence_length), dtype=jnp.int32)
    params = model.init(jax.random.PRNGKey(args.seed), sample)["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(args.learning_rate),
    )

    @jax.jit
    def train_step(current_state, inputs, targets, mask):
        def loss_fn(current_params):
            logits = current_state.apply_fn({"params": current_params}, inputs)
            losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return jnp.sum(losses * mask) / jnp.maximum(jnp.sum(mask), 1.0)

        loss, gradients = jax.value_and_grad(loss_fn)(current_state.params)
        return current_state.apply_gradients(grads=gradients), loss

    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(dataset), size=args.batch_size)
        batch = dataset[indices]
        inputs = jnp.asarray(batch[:, :-1])
        targets = jnp.asarray(batch[:, 1:])
        mask = (targets != PAD_ID).astype(jnp.float32)
        state, loss = train_step(state, inputs, targets, mask)
        if step == 1 or step % args.log_every == 0:
            print(f"step={step} loss={float(loss):.5f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialization.to_bytes(state.params))
    print(f"saved={output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=("dense", "moe"), default="moe")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-digits", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="checkpoints/model.msgpack")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
