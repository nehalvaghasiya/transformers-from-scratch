"""Small CPU checks for the scientific behavior, not just command plumbing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pretraining.artifact import export_artifact, fetch_artifact
from pretraining.config import ExperimentConfig
from pretraining.data import TicketDataset, batchify, generate_dataset, read_rows
from pretraining.evaluate import paired_interval, score
from pretraining.inference import load_artifact, predict
from pretraining.model import ModelConfig, TinyTicketLM, make_pool_head, pool_prompt
from pretraining.tokenizer import WordTokenizer
from pretraining.train import train


@pytest.fixture(autouse=True)
def cpu_threads() -> None:
    torch.set_num_threads(1)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    generate_dataset(
        tmp_path, n_train=36, n_val=18, n_test=18, n_challenge=18, version="ticket-v2"
    )

    return tmp_path


def test_v2_regeneration_matches_reference_manifest(tmp_path: Path) -> None:
    manifest = json.loads(
        (Path(__file__).parents[1] / "reports/ticket-v2/data-manifest.json").read_text()
    )
    generated = generate_dataset(tmp_path, version="ticket-v2")

    for split, path in generated.items():
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == manifest["splits"][split]["sha256"]
        )
        assert len(read_rows(path)) == manifest["splits"][split]["examples"]


def test_v2_has_disjoint_text_and_template_groups(data_dir: Path) -> None:
    splits = [
        read_rows(data_dir / f"{split}.csv")
        for split in ("train", "val", "test", "challenge")
    ]
    texts = [row["text"] for rows in splits for row in rows]

    assert len(texts) == len(set(texts))

    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            assert {row["template_id"] for row in left}.isdisjoint(
                row["template_id"] for row in right
            )


def test_tokenizer_is_training_only_and_roundtrips(tmp_path: Path) -> None:
    tokenizer = WordTokenizer.fit(["Invoice issue."])

    assert tokenizer.encode_text("neverobserved") == [tokenizer.unk_id]

    tokenizer.save(tmp_path / "vocab.json")

    assert WordTokenizer.load(tmp_path / "vocab.json").vocab == tokenizer.vocab


def test_label_loss_masking_and_gradient() -> None:
    row = {"text": "Invoice issue.", "intent": "billing_issue", "priority": "low"}
    tokenizer = WordTokenizer.fit([row["text"]])
    dataset = TicketDataset([row], tokenizer, 32)
    x, y = batchify([dataset[0]], tokenizer.pad_id)
    first = len(tokenizer.encode_prompt(row["text"])) - 1

    assert (y[0, :first] == -100).all()
    assert y[0, first:].tolist() == tokenizer.answer_ids("billing_issue", "low")

    model = TinyTicketLM(ModelConfig(len(tokenizer.vocab), 32, 1, 2, 16, 0.0))
    _, loss = model(x, y)

    assert loss is not None and torch.isfinite(loss)

    loss.backward()

    assert model.token_emb.weight is model.lm_head.weight
    assert model.token_emb.weight.grad is not None
    assert torch.isfinite(model.token_emb.weight.grad).all()


def test_causality_and_save_load(tmp_path: Path) -> None:
    model = TinyTicketLM(ModelConfig(32, 16, 1, 2, 16, 0)).eval()
    a, _ = model(torch.tensor([[1, 2, 3, 4, 5]]))
    b, _ = model(torch.tensor([[1, 2, 3, 8, 9]]))

    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-6)

    model.save_model(tmp_path)
    loaded = TinyTicketLM.load_model(tmp_path)
    c, _ = loaded(torch.tensor([[1, 2, 3, 4, 5]]))

    assert torch.equal(a, c)


def test_pooling_excludes_separator_answers_and_padding() -> None:
    tokens = torch.tensor([[2, 8, 3, 10, 0], [2, 3, 0, 0, 0]])
    hidden = torch.arange(10, dtype=torch.float32).reshape(2, 5, 1)
    pooled = pool_prompt(hidden, tokens, pad_id=0, sep_id=3)

    assert pooled.tolist() == [[0.5], [5.0]]


def test_batching_invariance_rejections_and_dual_head(tmp_path: Path) -> None:
    tokenizer = WordTokenizer.fit(["Invoice issue. No rush. Please reset my password."])
    model = TinyTicketLM(ModelConfig(len(tokenizer.vocab), 32, 1, 2, 16, 0)).eval()
    head = make_pool_head(16)
    texts = ["Invoice.", "Please reset my password.", "", "long " * 100]

    for auxiliary in (None, head):
        alone = predict(model, tokenizer, texts, batch_size=1, pool_head=auxiliary)
        batched = predict(model, tokenizer, texts, pool_head=auxiliary)

        assert alone == batched
        assert batched[2]["error"] == "empty input"
        assert batched[3]["error"] == "input too long"

    model.save_model(tmp_path)
    tokenizer.save(tmp_path / "vocab.json")
    torch.save(head.state_dict(), tmp_path / "pool_head.pt")
    (tmp_path / "run_info.json").write_text('{"pooling":"exclude_separator"}')
    loaded, vocab, heads, legacy = load_artifact(tmp_path)

    assert not legacy and heads is not None
    assert predict(loaded, vocab, texts, pool_head=heads) == predict(
        model, tokenizer, texts, pool_head=head
    )


def test_metrics_keep_rejected_examples_and_missing_classes() -> None:
    rows = [
        {"text": "a", "template_id": "a", "intent": "billing_issue", "priority": "low"},
        {"text": "b", "template_id": "b", "intent": "billing_issue", "priority": "low"},
    ]
    predictions = [
        {"intent": "billing_issue", "priority": "low"},
        {"intent": None, "priority": None, "error": "input too long"},
    ]
    metrics = score(rows, predictions)

    assert metrics["joint_accuracy"] == 0.5
    assert metrics["rejected_rate"] == 0.5
    assert metrics["intent_macro_f1"] == pytest.approx((2 / 3) / 6)
    assert metrics["intent_per_class"]["access_reset"]["support"] == 0

    with pytest.raises(ValueError):
        score(rows, predictions[:1])

    interval = paired_interval(rows, predictions, predictions, samples=20)

    assert interval["lower_95"] == interval["upper_95"] == 0


def test_epoch_recovery_matches_uninterrupted_training(
    data_dir: Path, tmp_path: Path
) -> None:
    config = ExperimentConfig(
        data_dir=str(data_dir),
        runs_dir=str(tmp_path / "runs"),
        threads=1,
        n_layer=1,
        n_head=2,
        n_embd=16,
        epochs=2,
        batch_size=18,
        pool_weight=2.0,
    )
    full = train(config, "full")
    train(config, "resumed", stop_after_epoch=1)
    resumed = train(config, "resumed", resume=True)

    for name in ("model.pt", "pool_head.pt"):
        a = torch.load(full / name, weights_only=True)
        b = torch.load(resumed / name, weights_only=True)

        assert all(torch.equal(a[key], b[key]) for key in a)

    assert (full / "history.json").read_bytes() == (
        resumed / "history.json"
    ).read_bytes()

    with pytest.raises(ValueError, match="identical"):
        train(replace(config, learning_rate=0.001), "resumed", resume=True)

    with pytest.raises(FileExistsError):
        train(config, "full")


def test_config_rejects_invalid_settings() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(schedule="unknown")

    with pytest.raises(ValueError):
        ExperimentConfig(learning_rate=float("nan"))


def test_artifact_roundtrip_and_checksum_rejection(tmp_path: Path) -> None:
    source = tmp_path / "source"
    tokenizer = WordTokenizer.fit(["Invoice issue."])
    model = TinyTicketLM(ModelConfig(len(tokenizer.vocab), 32, 1, 2, 16, 0)).eval()
    model.save_model(source)
    tokenizer.save(source / "vocab.json")
    (source / "run_info.json").write_text('{"pooling":"exclude_separator"}')
    archive = tmp_path / "release.tar.gz"
    manifest = export_artifact(source, archive)
    restored = tmp_path / "restored"
    fetch_artifact(archive.as_uri(), str(manifest["sha256"]), restored)
    loaded, vocab, _, _ = load_artifact(restored)

    assert predict(loaded, vocab, ["Invoice."]) == predict(
        model, tokenizer, ["Invoice."]
    )

    with pytest.raises(ValueError, match="checksum"):
        fetch_artifact(archive.as_uri(), "0" * 64, tmp_path / "bad")

    assert not (tmp_path / "bad").exists()


def test_sampler_stream_is_independent_of_model_rng(data_dir: Path) -> None:
    from torch.utils.data import DataLoader

    rows = read_rows(data_dir / "train.csv")
    tokenizer = WordTokenizer.fit([row["text"] for row in rows])
    dataset = TicketDataset(rows, tokenizer, 112)

    def first_batch(extra_draws: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(42)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, generator=generator)
        torch.rand(extra_draws)

        # Global random draws must not change the dedicated sampler stream.
        return next(iter(loader))[0]

    assert torch.equal(first_batch(0), first_batch(1000))
