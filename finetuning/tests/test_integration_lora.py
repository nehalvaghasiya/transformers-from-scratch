import json
from pathlib import Path

import pytest
import torch

from finetuning.lora_sql.adapters import load_adapter
from finetuning.tinysql.train import load_model
from finetuning.tinysql.utils import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_saved_adapter_loads_and_changes_logits():
    required = [
        ROOT / "base_small/model.pt",
        ROOT / "base_small/tokenizer.json",
        ROOT / "adapters/aug_r8_fast/adapter.pt",
        ROOT / "adapters/aug_r8_fast/compatibility.json",
    ]

    if not all(path.is_file() for path in required):
        pytest.skip(
            "Historical checkpoints are external artifacts; see finetuning/README.md"
        )

    torch.set_num_threads(2)
    base, tok = load_model(ROOT / "base_small")
    adapted, _ = load_model(ROOT / "base_small")
    load_adapter(adapted, ROOT / "adapters/aug_r8_fast/adapter.pt")
    ids = torch.tensor([tok.format_input("how many orders are completed")])

    assert (base(ids)[0] - adapted(ids)[0]).abs().max().item() > 1e-5
    assert all(
        not p.requires_grad for k, p in adapted.named_parameters() if "lora_" not in k
    )


def test_no_exact_train_eval_question_overlap(sql_data):
    train = {r["question"] for r in read_jsonl(sql_data / "data_lora/train.jsonl")}

    for split in ["val", "test", "challenge"]:
        assert not (
            train
            & {r["question"] for r in read_jsonl(sql_data / f"data/{split}.jsonl")}
        )


def test_measured_reports_are_consistent():
    for split, n in [("val", 320), ("test", 320), ("challenge", 160)]:
        x = json.loads((ROOT / f"reports/{split}_final.json").read_text())

        assert x["sample_count"] == n

        for label in ("frozen_base", "lora_raw", "word_tfidf_slots"):
            assert x["model_scores"][label]["count"] == n
