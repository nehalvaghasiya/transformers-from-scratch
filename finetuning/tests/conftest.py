"""CPU test defaults; file fixtures resolve explicitly without changing cwd."""

import pytest
import torch


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


@pytest.fixture(scope="session")
def sql_data(tmp_path_factory):
    """Regenerate the reference corpus without depending on ignored local data."""
    from finetuning.lora_sql.augment import create
    from finetuning.tinysql.data import build_dataset

    root = tmp_path_factory.mktemp("sql-data")
    build_dataset(root / "data", 6400, 320, 320, 160, seed=42)
    create(root / "data/train.jsonl", root / "data_lora/train.jsonl")
    return root


@pytest.fixture(scope="session")
def prepared_sql(tmp_path_factory):
    """Train tiny temporary artifacts for export tests; these are not benchmark weights."""
    from types import SimpleNamespace

    from finetuning.lora_sql.train_lora import train_lora
    from finetuning.tinysql.data import build_dataset
    from finetuning.tinysql.train import train

    root = tmp_path_factory.mktemp("sql-artifacts")
    build_dataset(root / "data", 80, 40, 40, 16, seed=42)
    train(
        data_dir=root / "data",
        out=root / "base",
        epochs=1,
        batch=8,
        steps=2,
        dim=16,
        layers=1,
        heads=2,
        device="cpu",
        num_threads=1,
        verbose=False,
    )
    train_lora(
        base=root / "base",
        data=root / "data",
        out=root / "adapter",
        epochs=1,
        batch=8,
        steps=2,
        rank=2,
        alpha=4,
        device="cpu",
        num_threads=1,
    )
    return SimpleNamespace(base=root / "base", adapter=root / "adapter", data=root / "data")
