"""Deterministic synthetic NL->read-only SQL data, SQL semantics, and strict splits."""

from __future__ import annotations

import json
import random
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = (
    "customers(id,name,state); orders(id,customer_id,total,status); "
    "products(id,name,category,price)"
)
STATES = ["tx", "ca", "ny", "wa", "fl", "oh", "az", "ga", "co", "il", "or", "nc"]
STATUS = ["completed", "pending", "cancelled", "refunded"]
CATEGORIES = [
    "books",
    "toys",
    "electronics",
    "clothing",
    "office",
    "sports",
    "beauty",
    "home",
]
AMOUNTS = [10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750]
LIMITS = [2, 3, 4, 5, 6, 8, 10]

# Template selection is by split, not randomly across the whole corpus.
TEMPLATES = {
    "state_customers": [
        "list customers in {state}",
        "show customer names from {state}",
        "which customers live in {state}",
        "find all customers located in {state}",
        "customer names for state {state}",
        "get customers from {state} alphabetically",
        "show names of customers in {state}",
        "display customers who are in {state}",
        "customers living in state {state}",
        "give me names of customers from {state}",
        "i need customers based in {state}",
        "retrieve names for the customers in {state}",
    ],
    "order_lookup": [
        "show the top {limit} {status} orders above {amount}",
        "find {limit} highest value {status} orders with total at least {amount}",
        "list {limit} {status} orders costing {amount} or more",
        "get the largest {limit} {status} orders over {amount} inclusive",
        "which {status} orders have totals of at least {amount} show {limit}",
        "top {limit} orders with status {status} and minimum total {amount}",
        "retrieve {limit} biggest {status} orders where total is at least {amount}",
        "display {limit} {status} orders with value no less than {amount}",
        "return the {limit} highest {status} orders worth at least {amount}",
        "give ids of {limit} largest {status} orders with total minimum {amount}",
        "list {limit} orders marked {status} totaling {amount} or higher",
        "i want {limit} expensive orders in {status} status of at least {amount}",
    ],
    "order_count": [
        "how many orders are {status}",
        "count all {status} orders",
        "number of orders with status {status}",
        "total count of {status} orders",
        "give the number of {status} orders",
        "get count of orders marked {status}",
        "calculate how many orders are {status}",
        "show order count for {status}",
        "tell me how many {status} orders exist",
        "what is the count of orders in {status} status",
        "count the orders that are {status}",
        "i need the number of orders labeled {status}",
    ],
    "order_sum": [
        "total revenue from {status} orders",
        "sum the total amounts of {status} orders",
        "what is the sum of order totals for {status}",
        "add up values of all {status} orders",
        "calculate total order value where status is {status}",
        "aggregate spending across {status} orders",
        "get sum of totals for orders marked {status}",
        "what is overall amount of {status} orders",
        "find total value of all {status} orders",
        "sum up the amounts for {status} orders",
        "give the revenue of orders in {status} status",
        "what do {status} orders add up to",
    ],
    "top_customers": [
        "top {limit} customers by spending on {status} orders",
        "who spent the most on {status} orders show {limit}",
        "rank the {limit} highest spenders for {status} orders",
        "list {limit} customers with greatest {status} order totals",
        "find {limit} customers generating the highest {status} order revenue",
        "show {limit} biggest customers by total value of {status} orders",
        "return top {limit} purchasers by {status} order sum",
        "which {limit} customers have largest spending in {status} orders",
        "give me the {limit} leading customers for {status} order value",
        "display {limit} customers ordered by the sum of {status} purchases",
        "get the {limit} customers spending most on orders marked {status}",
        "sort customers by {status} spending and take {limit}",
    ],
    "product_lookup": [
        "list {category} products costing at most {amount}",
        "show products in {category} priced no higher than {amount}",
        "find {category} items below {amount} inclusive",
        "display {category} products priced up to {amount}",
        "products of category {category} that cost {amount} or less",
        "get {category} product names with price not above {amount}",
        "show low priced {category} products with maximum {amount}",
        "retrieve {category} items costing no more than {amount}",
        "list names of {category} products under or equal to {amount}",
        "what {category} products cost at most {amount}",
        "give affordable {category} items with price ceiling {amount}",
        "find products classified {category} whose price is at most {amount}",
    ],
    "product_avg": [
        "average price of {category} products",
        "what is mean cost of {category} items",
        "calculate average price for category {category}",
        "show the mean price for {category}",
        "get average cost of products in {category}",
        "compute avg price of {category} products",
        "tell me the average product price in {category}",
        "mean price of all {category} items",
        "what do {category} products cost on average",
        "give the average price in {category} category",
        "retrieve average cost for {category}",
        "find mean pricing for products labeled {category}",
    ],
    "state_order_count": [
        "how many {status} orders came from {state} customers",
        "count {status} orders placed by customers in {state}",
        "number of {status} orders for state {state}",
        "total {status} orders belonging to {state} residents",
        "find count of {status} orders where customer state is {state}",
        "count the orders marked {status} from {state} customers",
        "show how many orders are {status} for customers in {state}",
        "get number of {status} orders for customers based in {state}",
        "calculate {status} order count in state {state}",
        "how many orders in {status} status from {state}",
        "return order count for status {status} and customer state {state}",
        "i need number of {status} orders for {state} customers",
    ],
}


def sql_for(family: str, values: dict[str, Any]) -> str:
    v = values

    if family == "state_customers":
        return f"SELECT name FROM customers WHERE state = '{v['state']}' ORDER BY name ASC;"

    if family == "order_lookup":
        return (
            f"SELECT id FROM orders WHERE status = '{v['status']}' AND total >= "
            f"{v['amount']} ORDER BY total DESC, id ASC LIMIT {v['limit']};"
        )

    if family == "order_count":
        return f"SELECT COUNT(*) FROM orders WHERE status = '{v['status']}';"

    if family == "order_sum":
        return f"SELECT SUM(total) FROM orders WHERE status = '{v['status']}';"

    if family == "top_customers":
        return (
            f"SELECT c.name, SUM(o.total) AS spent FROM customers c JOIN orders o "
            f"ON c.id = o.customer_id WHERE o.status = '{v['status']}' GROUP BY "
            f"c.id, c.name ORDER BY spent DESC, c.name ASC LIMIT {v['limit']};"
        )

    if family == "product_lookup":
        return (
            f"SELECT name FROM products WHERE category = '{v['category']}' AND "
            f"price <= {v['amount']} ORDER BY price ASC, name ASC;"
        )

    if family == "product_avg":
        return f"SELECT AVG(price) FROM products WHERE category = '{v['category']}';"

    if family == "state_order_count":
        return (
            f"SELECT COUNT(*) FROM orders o JOIN customers c ON o.customer_id = "
            f"c.id WHERE c.state = '{v['state']}' AND o.status = '{v['status']}';"
        )

    raise ValueError(family)


def values_for(family: str, rng: random.Random) -> dict[str, Any]:
    fields = {
        "state_customers": ["state"],
        "order_lookup": ["status", "amount", "limit"],
        "order_count": ["status"],
        "order_sum": ["status"],
        "top_customers": ["status", "limit"],
        "product_lookup": ["category", "amount"],
        "product_avg": ["category"],
        "state_order_count": ["status", "state"],
    }[family]
    choices: dict[str, list[Any]] = {
        "state": STATES,
        "status": STATUS,
        "category": CATEGORIES,
        "amount": AMOUNTS,
        "limit": LIMITS,
    }

    return {k: rng.choice(choices[k]) for k in fields}


def build_dataset(
    out: Path,
    n_train: int = 6400,
    n_val: int = 320,
    n_test: int = 320,
    n_challenge: int = 160,
    seed: int = 42,
) -> dict[str, Any]:
    """Templates 0-7 train, 8-9 validation, 10-11 test. Challenge uses "
    "unseen clauses and rearrangements."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    plans = {
        "train": (n_train, range(0, 8)),
        "val": (n_val, range(8, 10)),
        "test": (n_test, range(10, 12)),
    }
    used = set()
    stats = {}

    for split, (n, template_ids) in plans.items():
        rows: list[dict[str, Any]] = []
        trials = 0

        while len(rows) < n:
            trials += 1

            if trials > n * 100:
                raise RuntimeError("insufficient unique examples")

            family = rng.choice(list(TEMPLATES))
            values = values_for(family, rng)
            t = rng.choice(list(template_ids))
            question = TEMPLATES[family][t].format(**values)

            if question in used and split != "train":
                continue

            used.add(question)
            rows.append(
                {
                    "id": f"{split}-{len(rows):05d}",
                    "family": family,
                    "template_id": t,
                    "question": question,
                    "sql": sql_for(family, values),
                    "values": values,
                }
            )

        rng.shuffle(rows)
        (out / f"{split}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
        stats[split] = {
            "count": len(rows),
            "families": dict(Counter(x["family"] for x in rows)),
        }

    # Historical challenge phrasings: these were inspected during prior development.
    challenge_templates = {
        "state_customers": "i am interested in names of people from {state} sort by name",
        "order_lookup": (
            "i need {limit} ids for orders at least {amount} in {status} sorted largest first"
        ),
        "order_count": "provide a count for status {status} in orders",
        "order_sum": "add every order total where status equals {status}",
        "top_customers": "for {status} sales identify highest spending {limit} customers",
        "product_lookup": "find names and order by cheapness in {category} with max price {amount}",
        "product_avg": "compute the typical mean price of category {category}",
        "state_order_count": (
            "number of orders with {status} status whose customers live in {state}"
        ),
    }
    rows = []
    used_ch = set()
    trials = 0

    while len(rows) < n_challenge:
        trials += 1

        if trials > n_challenge * 100:
            raise RuntimeError("insufficient challenge examples")

        family = rng.choice(list(TEMPLATES))
        values = values_for(family, rng)
        question = challenge_templates[family].format(**values)

        if question in used_ch or question in used:
            continue

        used_ch.add(question)
        rows.append(
            {
                "id": f"challenge-{len(rows):05d}",
                "family": family,
                "template_id": 99,
                "question": question,
                "sql": sql_for(family, values),
                "values": values,
            }
        )

    (out / "challenge.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    stats["challenge"] = {
        "count": len(rows),
        "families": dict(Counter(x["family"] for x in rows)),
    }
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "schema": SCHEMA,
                "splits": stats,
                "note": (
                    "Synthetic domain-specific grammar. Questions are unique across "
                    "splits. Fixed template ranges held out."
                ),
            },
            indent=2,
        )
    )

    return stats


def create_db(seed: int = 2026) -> sqlite3.Connection:
    rng = random.Random(seed)
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE customers(id INTEGER PRIMARY KEY,name TEXT,state TEXT);
        CREATE TABLE orders(id INTEGER PRIMARY KEY,customer_id INTEGER,total INTEGER,status TEXT);
        CREATE TABLE products(id INTEGER PRIMARY KEY,name TEXT,category TEXT,price INTEGER);
    """
    )
    first = [
        "alex",
        "sam",
        "jordan",
        "taylor",
        "avery",
        "riley",
        "casey",
        "morgan",
        "jamie",
        "cameron",
    ]

    for i in range(1, 101):
        db.execute(
            "INSERT INTO customers VALUES (?,?,?)",
            (i, f"{rng.choice(first)}_{i}", rng.choice(STATES)),
        )

    for i in range(1, 801):
        db.execute(
            "INSERT INTO orders VALUES (?,?,?,?)",
            (
                i,
                rng.randint(1, 100),
                rng.choice(AMOUNTS) + rng.randint(0, 70),
                rng.choice(STATUS),
            ),
        )

    for i in range(1, 241):
        db.execute(
            "INSERT INTO products VALUES (?,?,?,?)",
            (
                i,
                f"product_{i}",
                rng.choice(CATEGORIES),
                rng.choice(AMOUNTS) + rng.randint(0, 20),
            ),
        )

    db.commit()

    return db


def validate_read_only(sql: str) -> bool:
    import re

    cleaned = sql.strip()

    if not re.fullmatch(r"(?is)\s*select\b[^;]*;?\s*", cleaned):
        return False

    if re.search(
        r"(?i)\b(insert|update|delete|drop|attach|pragma|alter|create|replace|vacuum|load_extension)\b",
        cleaned,
    ):
        return False

    if "--" in cleaned or "/*" in cleaned:
        return False

    return True


def execute_sql(
    sql: str, db: sqlite3.Connection, timeout_steps: int = 40000
) -> list[Any]:
    if not validate_read_only(sql):
        raise ValueError("Only single read-only SELECT queries supported")

    progress = [0]

    def watchdog() -> bool:
        progress[0] += 1

        return progress[0] > timeout_steps

    db.set_progress_handler(watchdog, 100)

    try:
        return db.execute(sql).fetchall()
    finally:
        db.set_progress_handler(None, 0)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    p.add_argument("--train", type=int, default=6400)
    p.add_argument("--val", type=int, default=320)
    p.add_argument("--test", type=int, default=320)
    p.add_argument("--challenge", type=int, default=160)
    a = p.parse_args()
    print(
        json.dumps(
            build_dataset(Path(a.out), a.train, a.val, a.test, a.challenge), indent=2
        )
    )
