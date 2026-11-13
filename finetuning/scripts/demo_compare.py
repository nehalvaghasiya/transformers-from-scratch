"""Compare actual SQL from the frozen base and the explicitly selected adapter."""

import argparse
from pathlib import Path

import torch

from finetuning.lora_sql.adapters import load_adapter
from finetuning.tinysql.evaluate import generate_many
from finetuning.tinysql.train import load_model


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", type=Path, default=root / "adapters/aug_r8_fast/adapter.pt"
    )
    parser.add_argument("--base", type=Path, default=root / "base_small")
    args = parser.parse_args()

    torch.set_num_threads(2)
    base, tokenizer = load_model(args.base)

    queries = [
        "retrieve names for the customers in oh",
        "i need number of completed orders for ny customers",
        "i want 5 expensive orders in pending status of at least 40",
    ]
    before = generate_many(base, tokenizer, queries)
    load_adapter(base, args.adapter)
    after = generate_many(base, tokenizer, queries)

    for question, original, adapted in zip(queries, before, after, strict=True):
        print(f"\nQuestion: {question}\nFrozen base: {original}\nLoRA: {adapted}")


if __name__ == "__main__":
    main()
