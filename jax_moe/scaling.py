import argparse
import csv
from pathlib import Path


def parameter_counts(hidden_size: int, num_layers: int, num_experts: int, vocab_size: int = 13, mlp_ratio: int = 4):
    ffn_size = hidden_size * mlp_ratio
    shared = 2 * vocab_size * hidden_size + num_layers * (4 * hidden_size * hidden_size + 4 * hidden_size)
    dense_ffn = 2 * hidden_size * ffn_size
    dense_total = shared + num_layers * dense_ffn
    moe_total = shared + num_layers * (num_experts * dense_ffn + hidden_size * num_experts)
    moe_active = shared + num_layers * (dense_ffn + hidden_size * num_experts)
    return dense_total, moe_total, moe_active


def build_grid(hidden_sizes=(64, 128, 256), layer_counts=(2, 4), expert_counts=(4, 8), token_budgets=(100_000, 500_000, 1_000_000)):
    rows = []
    for hidden_size in hidden_sizes:
        for num_layers in layer_counts:
            for tokens in token_budgets:
                dense_total, _, _ = parameter_counts(hidden_size, num_layers, 1)
                rows.append({
                    "architecture": "dense",
                    "hidden_size": hidden_size,
                    "layers": num_layers,
                    "experts": 1,
                    "tokens": tokens,
                    "total_parameters": dense_total,
                    "active_parameters": dense_total,
                    "estimated_training_flops": 6 * dense_total * tokens,
                })
                for num_experts in expert_counts:
                    _, moe_total, moe_active = parameter_counts(hidden_size, num_layers, num_experts)
                    rows.append({
                        "architecture": "moe",
                        "hidden_size": hidden_size,
                        "layers": num_layers,
                        "experts": num_experts,
                        "tokens": tokens,
                        "total_parameters": moe_total,
                        "active_parameters": moe_active,
                        "estimated_training_flops": 6 * moe_active * tokens,
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="scaling.csv")
    args = parser.parse_args()
    rows = build_grid()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote={output} rows={len(rows)}")


if __name__ == "__main__":
    main()
