"""Produce new training-only synthetic paraphrases; never copy val/test/challenge questions.

All gold SQL is generated from a fixed grammar; no evaluation rows feed this process.
The augmentation is explicitly synthetic and NOT a new independent data source.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from finetuning.tinysql.utils import Record, read_jsonl, write_jsonl

PARAPHRASES = {
    "state_customers": [
        "show customers from {state} alphabetically",
        "find customer names in {state}",
        "get names of all customers from {state}",
        "list customer names located in {state}",
        "get the names of customers in state {state}",
        "display names of customers who live in {state}",
    ],
    "order_lookup": [
        "get top {limit} orders marked {status} where total is at least {amount}",
        "find highest {limit} orders above {amount} for status {status}",
        "list the largest {limit} orders in status {status} worth at least {amount}",
        "top {limit} {status} orders with total minimum {amount}",
        "show {limit} {status} orders of total at least {amount}",
        "find {limit} biggest orders for {status} with value at least {amount}",
    ],
    "order_count": [
        "show number of orders marked {status}",
        "give total count of orders in {status}",
        "count orders with {status} status",
        "calculate number of {status} orders",
        "get the order count for {status}",
        "how many {status} orders are there",
    ],
    "order_sum": [
        "show sum of total for {status} orders",
        "calculate total revenue for orders marked {status}",
        "get total value of {status} orders",
        "add up order amounts in {status}",
        "sum values of {status} orders",
        "give total amount of all {status} orders",
    ],
    "top_customers": [
        "get top {limit} customers for {status} spending",
        "list {limit} leading customers by {status} order totals",
        "show largest {limit} customers by {status} revenue",
        "find {limit} customers with highest total for {status} orders",
        "who are top {limit} customers in {status} orders by spending",
        "sort customers by total {status} spending show {limit}",
    ],
    "product_lookup": [
        "get {category} product names priced no more than {amount}",
        "list {category} items with price up to {amount}",
        "show products from {category} with maximum price {amount}",
        "find products in {category} at most {amount}",
        "display {category} products that cost {amount} or less",
        "get names of {category} items below {amount} inclusive",
    ],
    "product_avg": [
        "get average price of items in {category}",
        "find the mean price of {category} products",
        "calculate mean cost of {category} products",
        "average price for products in category {category}",
        "show average cost for {category} products",
        "give mean product price in {category}",
    ],
    "state_order_count": [
        "count {status} orders where customers are from {state}",
        "get number of orders in {status} for {state} customers",
        "calculate number of {status} orders from state {state}",
        "how many orders for {state} customers are {status}",
        "show count of {status} orders placed in {state}",
        "find order count for customers from {state} with status {status}",
    ],
}


def create(
    src: str | Path = "data/train.jsonl",
    out: str | Path = "runs/augmentation/train.jsonl",
    seed: int = 123,
    repeat: int = 2,
    original_fraction: float = 0.5,
) -> Record:
    rng = random.Random(seed)
    base = read_jsonl(src)
    result: list[Record] = []
    for row in base:
        if rng.random() < original_fraction:
            result.append(dict(row, id="lora-original-" + str(len(result))))
        for _ in range(repeat):
            question = rng.choice(PARAPHRASES[row["family"]]).format(**row["values"])
            result.append(
                dict(
                    row,
                    id=f"lora-aug-{len(result)}",
                    question=question,
                    template_id="augmentation",
                    source="train_only_paraphrase",
                )
            )
    rng.shuffle(result)
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, result)
    validation = Path(src).with_name("val.jsonl")
    if validation.is_file() and validation.resolve() != output.with_name("val.jsonl").resolve():
        shutil.copy2(validation, output.with_name("val.jsonl"))
    return {
        "count": len(result),
        "source_examples": len(base),
        "family_counts": dict(Counter(z["family"] for z in result)),
        "warning": "synthetic paraphrases; original val/test/challenge held out",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/train.jsonl")
    parser.add_argument("--out", default="runs/augmentation/train.jsonl")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    print(json.dumps(create(**vars(args)), indent=2))
