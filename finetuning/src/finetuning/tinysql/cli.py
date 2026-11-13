"""Quick terminal inference and safe SQLite demonstration."""

import argparse

import torch

from .data import SCHEMA, create_db, execute_sql
from .train import load_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?", default="how many orders are completed")
    p.add_argument("--artifacts", default="base_small")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-new-tokens", type=int, default=85)
    a = p.parse_args()

    torch.set_num_threads(2)
    m, t = load_model(a.artifacts, a.device)

    ids = m.generate(t.format_input(a.question), t.eos_id, a.max_new_tokens)
    sql = t.decode(ids)

    print("SCHEMA:", SCHEMA, "\nQUESTION:", a.question, "\nGENERATED SQL:", sql)

    with create_db() as db:
        try:
            print("RESULT (first 20 rows):", execute_sql(sql, db)[:20])
        except Exception as ex:
            print("EXECUTION FAILED:", str(ex))


if __name__ == "__main__":
    main()
