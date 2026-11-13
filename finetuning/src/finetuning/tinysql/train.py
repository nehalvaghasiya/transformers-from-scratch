"""Reproducible next-token SQL training with masked prompt loss and real validation."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .model import ModelConfig, TinySQLLM
from .tokenizer import WordSQLTokenizer
from .utils import Record, device_for, file_hash, provenance, read_jsonl, seed_all


class SQLDataset(Dataset[tuple[list[int], list[int]]]):
    def __init__(self, records: list[Record], tok: WordSQLTokenizer, context: int) -> None:
        self.rows = [tok.format_train(row, context) for row in records]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        return self.rows[i]


def collate(samples: list[tuple[list[int], list[int]]], pad_id: int) -> tuple[Tensor, Tensor]:
    T = max(len(x) for x, y in samples)
    x = torch.full((len(samples), T), pad_id, dtype=torch.long)
    y = torch.full((len(samples), T), -100, dtype=torch.long)
    for i, (xi, yi) in enumerate(samples):
        x[i, : len(xi)] = torch.tensor(xi)
        y[i, : len(yi)] = torch.tensor(yi)
    return x, y


def teacher_eval(model: TinySQLLM, loader: Iterable[tuple[Tensor, Tensor]], device: str) -> float:
    model.eval()
    total = 0.0
    weight = 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            assert loss is not None
            count = int((y != -100).sum())
            total += loss.item() * count
            weight += count
    if not weight:
        raise ValueError("No supervised validation tokens")
    return total / weight


def train(
    data_dir: str | Path = "data",
    out: str | Path = "runs/base",
    epochs: int = 20,
    batch: int = 64,
    lr: float = 0.001,
    seed: int = 42,
    dim: int = 128,
    layers: int = 2,
    heads: int = 4,
    context: int = 160,
    dropout: float = 0.0,
    steps: int = 0,
    patience: int = 10,
    device: str | None = None,
    num_threads: int = 4,
    verbose: bool = True,
) -> Record:
    if Path(out).exists():
        raise FileExistsError(out)
    if (
        min(epochs, batch, num_threads) < 1
        or steps < 0
        or patience < 0
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("Invalid training settings")
    t0 = time.time()
    seed_all(seed)
    if num_threads > 0:
        torch.set_num_threads(num_threads)
    device = device or device_for()
    data_dir = Path(data_dir)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows = read_jsonl(data_dir / "train.jsonl")
    valrows = read_jsonl(data_dir / "val.jsonl")
    tok = WordSQLTokenizer.fit(rows)
    tok.save(out / "tokenizer.json")
    cfg = ModelConfig(
        vocab_size=len(tok.vocab),
        context=context,
        dim=dim,
        layers=layers,
        heads=heads,
        dropout=dropout,
    )
    model = TinySQLLM(cfg).to(device)
    tr_ds = SQLDataset(rows, tok, context)
    vl_ds = SQLDataset(valrows, tok, context)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        tr_ds,
        batch_size=batch,
        shuffle=True,
        collate_fn=partial(collate, pad_id=tok.pad_id),
        generator=g,
        num_workers=0,
    )
    val_loader = DataLoader(
        vl_ds,
        batch_size=batch,
        shuffle=False,
        collate_fn=partial(collate, pad_id=tok.pad_id),
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = steps if steps else epochs * len(train_loader)
    warmup = max(1, int(total_steps * 0.05))

    def multiplier(step: int) -> float:
        if step < warmup:
            return max(1e-3, (step + 1) / warmup)
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    history = []
    best = float("inf")
    bad = 0
    step = 0
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        seen = 0
        loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            tokens = int((yb != -100).sum())
            loss_sum += loss.item() * tokens
            seen += tokens
            if steps and step >= steps:
                break
        train_loss = loss_sum / max(seen, 1)
        val_loss = teacher_eval(model, val_loader, device)
        rec = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        if val_loss < best - 1e-4:
            best = val_loss
            bad = 0
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "seed": seed,
                    "step": step,
                    "best_val_loss": best,
                },
                out / "model.pt",
            )
        else:
            bad += 1
        if verbose:
            print(json.dumps(rec), flush=True)
        if (patience and bad >= patience) or (steps and step >= steps):
            break
    meta: Record = {
        "seed": seed,
        "parameters": model.count_parameters(),
        "config": asdict(cfg),
        "train_examples": len(rows),
        "validation_examples": len(valrows),
        "best_epoch": best_epoch,
        "best_validation_loss": best,
        "duration_seconds": round(time.time() - t0, 2),
        "device": device,
        "max_steps": total_steps,
        "completed_steps": step,
        "checkpoint_selection": "minimum held-out template validation SQL token cross-entropy",
        "training_note": (
            "Decoder-only transformer initialized from scratch; SQL target tokens only."
        ),
    }
    meta.update(
        {
            "training_config": {
                "epochs": epochs,
                "batch": batch,
                "learning_rate": lr,
                "steps": steps,
                "patience": patience,
                "num_threads": num_threads,
                "weight_decay": 0.01,
            },
            "tokenizer_sha256": file_hash(out / "tokenizer.json"),
            **provenance(data_dir),
        }
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    return meta


def load_model(
    artifacts: str | Path = "base_small", device: str = "cpu"
) -> tuple[TinySQLLM, WordSQLTokenizer]:
    p = Path(artifacts)
    tok = WordSQLTokenizer.load(p / "tokenizer.json")
    ck = torch.load(p / "model.pt", map_location="cpu", weights_only=True)
    cfg = ModelConfig(**ck["config"])
    if cfg.vocab_size != len(tok.vocab):
        raise ValueError("Tokenizer/model vocabulary mismatch")
    with torch.random.fork_rng(devices=[]):
        model = TinySQLLM(cfg)
    model.load_state_dict(ck["state_dict"])
    model.tokenizer_sha256 = file_hash(p / "tokenizer.json")
    model.to(device).eval()
    return model, tok


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="runs/base")
    for flag, typ, default in [
        ("epochs", int, 20),
        ("batch", int, 64),
        ("lr", float, 0.001),
        ("seed", int, 42),
        ("dim", int, 128),
        ("layers", int, 2),
        ("heads", int, 4),
        ("context", int, 160),
        ("dropout", float, 0.0),
        ("steps", int, 0),
        ("patience", int, 10),
        ("num-threads", int, 4),
    ]:
        p.add_argument("--" + flag, type=typ, default=default)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    print(json.dumps(train(**vars(a)), indent=2))
