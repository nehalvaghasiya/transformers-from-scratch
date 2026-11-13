"""Regressions for inference parity, coverage and adapter compatibility."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from finetuning.lora_sql.adapters import (
    LoRAConfig,
    LoRALinear,
    attach_lora,
    load_adapter,
    save_adapter,
)
from finetuning.lora_sql.publish import prepare_hf
from finetuning.lora_sql.train_lora import train_lora
from finetuning.tinysql.evaluate import report, results_for
from finetuning.tinysql.model import ModelConfig, TinySQLLM
from finetuning.tinysql.train import train

ROOT = Path(__file__).resolve().parents[1]


def tiny():
    torch.manual_seed(17)

    return TinySQLLM(ModelConfig(vocab_size=24, context=12, dim=16, layers=1, heads=2))


def saved(tmp_path):
    model = tiny()
    config = LoRAConfig(rank=2)
    attach_lora(model, config)
    save_adapter(model, tmp_path / "adapter.pt", config)

    return tmp_path / "adapter.pt"


def test_adapter_inherits_dtype_and_device():
    layer = torch.nn.Linear(4, 6, dtype=torch.float64)
    adapter = LoRALinear(layer, 2, 4, 0)
    inputs = torch.ones(3, 4, dtype=torch.float64)

    assert torch.equal(layer(inputs), adapter(inputs))

    adapter(inputs).sum().backward()

    assert adapter.lora_a.dtype == torch.float64
    assert adapter.lora_b.grad is not None

    meta = LoRALinear(torch.nn.Linear(4, 6, device="meta"), 2, 4, 0)

    assert meta.lora_a.device.type == meta.lora_b.device.type == "meta"


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA execution is not available"
)
def test_cuda_adapter_update():
    layer = LoRALinear(torch.nn.Linear(4, 6, device="cuda"), 2, 4, 0)
    layer(torch.ones(2, 4, device="cuda")).sum().backward()

    assert layer.lora_b.grad is not None and layer.lora_b.grad.is_cuda


def test_wrong_base_and_tokenizer_rejected_before_mutation(tmp_path):
    path = saved(tmp_path)
    other = tiny()

    with torch.no_grad():
        other.tok_emb.weight[0, 0] += 0.1

    with pytest.raises(ValueError, match="base fingerprint"):
        load_adapter(other, path)

    assert all(p.requires_grad for p in other.parameters())

    other = tiny()
    other.tokenizer_sha256 = "wrong"

    with pytest.raises(ValueError, match="tokenizer"):
        load_adapter(other, path)


def test_adapter_tensor_shape_validation(tmp_path):
    path = saved(tmp_path)
    data = torch.load(path, weights_only=True)
    name = next(iter(data["state_dict"]))
    data["state_dict"][name] = torch.zeros(1)
    torch.save(data, path)
    model = tiny()

    with pytest.raises(ValueError, match="shapes"):
        load_adapter(model, path)

    assert all(p.requires_grad for p in model.parameters())


def test_legacy_adapter_requires_manifest(tmp_path):
    path = saved(tmp_path)
    data = torch.load(path, weights_only=True)
    del data["format"]
    torch.save(data, path)

    with pytest.raises(ValueError, match="manifest"):
        load_adapter(tiny(), path)


def test_generation_parity_and_original_context_boundary():
    model = tiny()
    prompts = [[1, 3], [1, 4, 5], [1] * 11]
    batched = model.generate_batch(prompts, eos_id=-1, max_new_tokens=85)

    assert batched == [model.generate(p, -1, 85) for p in prompts]
    assert [len(x) for x in batched] == [10, 9, 1]

    with pytest.raises(ValueError, match="context"):
        model.generate([1] * 12, -1)

    with pytest.raises(ValueError):
        model.generate([], -1)


def evaluation_rows():
    return [
        {
            "id": str(i),
            "family": "count",
            "question": "count",
            "sql": "SELECT COUNT(*) FROM orders;",
        }
        for i in range(2)
    ]


def test_missing_duplicate_and_misordered_predictions_fail():
    rows = evaluation_rows()

    with pytest.raises(ValueError, match="one prediction"):
        results_for(rows, ["SELECT 1;"])

    with pytest.raises(ValueError, match="unique"):
        results_for([rows[0], rows[0]], ["SELECT 1;", "SELECT 1;"])

    scored = results_for(rows, [r["sql"] for r in rows])

    with pytest.raises(ValueError, match="IDs"):
        report(scored[::-1], ["0", "1"])

    with pytest.raises(ValueError, match="IDs"):
        report(scored[:1], ["0", "1"])


def test_failures_keep_full_denominator_and_literal_exact_match():
    rows = evaluation_rows()
    scored = results_for(rows, [rows[0]["sql"], None])

    assert report(scored)["execution_accuracy"] == 0.5

    row = dict(rows[0], sql="SELECT 'New York';")
    scores = results_for([row], ["SELECT 'newyork';"])

    assert not scores[0]["exact_match"]
    assert not scores[0]["execution_accuracy"]


def test_training_refuses_overwrite(tmp_path):
    with pytest.raises(FileExistsError):
        train(out=tmp_path)

    with pytest.raises(FileExistsError):
        train_lora(out=tmp_path)


def test_export_does_not_borrow_unrelated_metrics(tmp_path, prepared_sql):
    from finetuning.lora_sql.publish import model_card

    bogus = tmp_path / "report.json"
    bogus.write_text(json.dumps({"adapter_sha256": "another-checkpoint"}))

    with pytest.raises(ValueError, match="does not identify"):
        model_card(prepared_sql.adapter, tmp_path / "README.md", (bogus,))

    bundle = prepare_hf(prepared_sql.adapter, tmp_path / "bundle", prepared_sql.base)

    assert (bundle / "base/model.pt").is_file()
    assert (bundle / "finetuning/tinysql/model.py").is_file()
    assert "No execution-accuracy report" in (bundle / "README.md").read_text()


def test_export_cli_accepts_the_active_report(tmp_path, prepared_sql):
    from finetuning.lora_sql.evaluate_lora import evaluate

    report_path = tmp_path / "test.json"
    measured = evaluate(
        base=prepared_sql.base,
        adapter=prepared_sql.adapter / "adapter.pt",
        data=prepared_sql.data,
        split="val",
        count=8,
        baselines=False,
        output=report_path,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "finetuning.lora_sql.publish",
            "--base",
            str(prepared_sql.base),
            "--adapter",
            str(prepared_sql.adapter),
            "--out",
            str(tmp_path / "export"),
            "--report",
            str(report_path),
        ],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr

    score = measured["model_scores"]["lora_raw"]["execution_accuracy"]

    assert f"adapter {score:.2%}" in (tmp_path / "export/README.md").read_text()


def test_training_artifacts_record_provenance_and_load_together(tmp_path, sql_data):
    from finetuning.tinysql.train import load_model
    from finetuning.tinysql.utils import file_hash

    base_dir = tmp_path / "base"
    base_meta = train(
        data_dir=sql_data / "data",
        out=base_dir,
        epochs=1,
        batch=8,
        steps=2,
        dim=16,
        layers=1,
        heads=2,
        num_threads=1,
        device="cpu",
        verbose=False,
    )
    original_hash = file_hash(base_dir / "model.pt")
    adapter_dir = tmp_path / "adapter"
    adapter_meta = train_lora(
        base=base_dir,
        data=sql_data / "data",
        out=adapter_dir,
        epochs=1,
        batch=8,
        steps=2,
        rank=2,
        num_threads=1,
        device="cpu",
    )
    model, tokenizer = load_model(base_dir)
    load_adapter(model, adapter_dir / "adapter.pt")

    assert file_hash(base_dir / "model.pt") == original_hash
    assert base_meta["completed_steps"] == adapter_meta["completed_steps"] == 2
    assert adapter_meta["base_params"] == base_meta["parameters"]
    assert adapter_meta["tokenizer_sha256"] == file_hash(base_dir / "tokenizer.json")
    assert adapter_meta["source_hashes"] and adapter_meta["data_hashes"]

    ids = torch.tensor([tokenizer.format_input("how many orders are completed")])

    assert torch.isfinite(model(ids)[0]).all()


def test_augmentation_regenerates_training_and_preserves_validation(tmp_path, sql_data):
    from finetuning.lora_sql.augment import create
    from finetuning.tinysql.utils import file_hash

    manifest = json.loads((ROOT / "experiments/data-manifest.json").read_text())

    for name, expected in manifest["sha256"].items():
        assert file_hash(sql_data / name) == expected

    result = create(sql_data / "data/train.jsonl", tmp_path / "train.jsonl")

    assert result["count"] == 15930
    assert (tmp_path / "train.jsonl").read_bytes() == (
        sql_data / "data_lora/train.jsonl"
    ).read_bytes()
    assert (tmp_path / "val.jsonl").read_bytes() == (
        sql_data / "data/val.jsonl"
    ).read_bytes()
