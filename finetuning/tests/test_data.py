import pytest

from finetuning.tinysql.data import (
    build_dataset,
    create_db,
    execute_sql,
    sql_for,
    validate_read_only,
)
from finetuning.tinysql.utils import read_jsonl


def test_dataset_is_disjoint_and_template_heldout(tmp_path):
    build_dataset(tmp_path, n_train=80, n_val=40, n_test=40, n_challenge=16)
    splits = {
        x: read_jsonl(tmp_path / f"{x}.jsonl")
        for x in ["train", "val", "test", "challenge"]
    }

    for a, bb in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert set(x["question"] for x in splits[a]).isdisjoint(
            x["question"] for x in splits[bb]
        )

    assert set(r["template_id"] for r in splits["train"]) <= set(range(8))
    assert set(r["template_id"] for r in splits["val"]) <= set(range(8, 10))
    assert set(r["template_id"] for r in splits["test"]) <= set(range(10, 12))
    assert all(r["template_id"] == 99 for r in splits["challenge"])


def test_sql_gold_executes_all_families():
    db = create_db()
    examples = {
        "state_customers": {"state": "tx"},
        "order_lookup": {"status": "completed", "amount": 50, "limit": 3},
        "order_count": {"status": "pending"},
        "order_sum": {"status": "refunded"},
        "top_customers": {"status": "completed", "limit": 3},
        "product_lookup": {"category": "books", "amount": 100},
        "product_avg": {"category": "toys"},
        "state_order_count": {"state": "tx", "status": "completed"},
    }

    for k, v in examples.items():
        sql = sql_for(k, v)

        assert validate_read_only(sql)
        assert isinstance(execute_sql(sql, db), list)

    db.close()


@pytest.mark.parametrize(
    "bad",
    [
        "DROP TABLE customers",
        "SELECT 1; DELETE FROM customers",
        "PRAGMA database_list",
        "SELECT load_extension('evil')",
        "SELECT 1 -- comment",
        "SELECT 1 /* x */",
        "UPDATE orders SET total=0",
    ],
)
def test_sql_safety(bad):
    assert not validate_read_only(bad)

    with pytest.raises(ValueError):
        execute_sql(bad, create_db())


def test_seed_determinism(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    build_dataset(a, 60, 24, 24, 16, seed=42)
    build_dataset(b, 60, 24, 24, 16, seed=42)

    for x in ["train", "val", "test", "challenge"]:
        assert (a / f"{x}.jsonl").read_bytes() == (b / f"{x}.jsonl").read_bytes()
