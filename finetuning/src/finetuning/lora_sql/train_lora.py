"""Frozen-base LoRA experiments; checkpoints selected by validation target-token CE.
Test/challenge splits MUST NOT be used for hyperparameter selection.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from finetuning.tinysql.train import SQLDataset, collate, load_model, teacher_eval
from finetuning.tinysql.utils import Record, file_hash, provenance, read_jsonl, seed_all

from .adapters import LoRAConfig, adapter_parameters, attach_lora, save_adapter


def train_lora(
    base: str | Path = "base_small",
    data: str | Path = "data",
    out: str | Path = "runs/adapter",
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    lr: float = 0.001,
    batch: int = 64,
    epochs: int = 6,
    seed: int = 42,
    patience: int = 3,
    steps: int = 0,
    device: str | None = None,
    num_threads: int = 2,
    targets: tuple[str, ...] = ("qkv", "proj", "ff.0", "ff.2"),
) -> Record:
    if Path(out).exists():
        raise FileExistsError(out)
    if (
        min(batch, epochs, num_threads) < 1
        or steps < 0
        or patience < 0
        or not math.isfinite(lr)
        or lr <= 0
    ):
        raise ValueError("Invalid training settings")
    start = time.time()
    seed_all(seed)
    torch.set_num_threads(num_threads)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_model, tok = load_model(base, device)
    base_model.eval()
    base_count = base_model.count_parameters()
    frozen_checksum = {k: v.detach().clone() for k, v in base_model.state_dict().items()}
    cfg = LoRAConfig(rank, alpha, dropout, tuple(targets))
    attach_lora(base_model, cfg)
    trainable = sum(p.numel() for p in adapter_parameters(base_model))
    total = sum(p.numel() for p in base_model.parameters())
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    rows = read_jsonl(Path(data) / "train.jsonl")
    val = read_jsonl(Path(data) / "val.jsonl")
    train_ds = SQLDataset(rows, tok, base_model.config.context)
    val_ds = SQLDataset(val, tok, base_model.config.context)
    g = torch.Generator().manual_seed(seed)

    def make(records: SQLDataset, shuffle: bool) -> DataLoader[Any]:
        return DataLoader(
            records,
            batch_size=batch,
            shuffle=shuffle,
            generator=g if shuffle else None,
            collate_fn=partial(collate, pad_id=tok.pad_id),
            num_workers=0,
        )

    loader = make(train_ds, True)
    vloader = make(val_ds, False)
    optim = torch.optim.AdamW(adapter_parameters(base_model), lr=lr, weight_decay=0.0)
    total_steps = steps or (epochs * len(loader))
    warmup = max(1, int(total_steps * 0.05))

    def scale(step: int) -> float:
        if step < warmup:
            return max(0.001, (step + 1) / warmup)
        x = min(1, (step - warmup) / max(1, total_steps - warmup))
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * x))

    schedule = torch.optim.lr_scheduler.LambdaLR(optim, scale)
    baseline_val = teacher_eval(base_model, vloader, device)
    best = baseline_val
    best_epoch = 0
    bad = 0
    history = []
    step = 0
    # save zero-update as fallback, allowing honest reporting if LoRA fails
    save_adapter(base_model, output / "adapter.pt", cfg, base_ref=base)
    for epoch in range(1, epochs + 1):
        base_model.train()
        seen = 0
        weighted = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            _, loss = base_model(x, y)
            assert loss is not None
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter_parameters(base_model), 1.0)
            optim.step()
            schedule.step()
            step += 1
            n = (y != -100).sum().item()
            seen += n
            weighted += loss.detach().item() * n
            if steps and step >= steps:
                break
        val_ce = teacher_eval(base_model, vloader, device)
        record = {
            "epoch": epoch,
            "step": step,
            "train_sql_ce": weighted / max(1, seen),
            "val_sql_ce": val_ce,
            "lr": optim.param_groups[0]["lr"],
            "seconds": round(time.time() - start, 2),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if val_ce < best - 1e-5:
            best = val_ce
            best_epoch = epoch
            bad = 0
            save_adapter(base_model, output / "adapter.pt", cfg, base_ref=base)
        else:
            bad += 1
        if (patience and bad >= patience) or (steps and step >= steps):
            break
    # verify exact unchanged frozen base weights within LoRA wrappers
    current = {
        k.replace(".original.", "."): v
        for k, v in base_model.state_dict().items()
        if ".lora_" not in k
    }
    if any(not torch.equal(current[k].cpu(), v.cpu()) for k, v in frozen_checksum.items()):
        raise AssertionError("Frozen base changed")
    summary: Record = {
        "base": str(base),
        "base_params": base_count,
        "total_model_parameters": sum(p.numel() for p in base_model.parameters()),
        "trainable_parameters": trainable,
        "adapter_to_base_fraction": trainable / base_count,
        "adapter_to_total_fraction": trainable / total,
        "config": asdict(cfg),
        "seed": seed,
        "learning_rate": lr,
        "batch_size": batch,
        "training_data": str(data),
        "base_val_sql_ce": baseline_val,
        "best_val_sql_ce": best,
        "best_epoch": best_epoch,
        "completed_steps": step,
        "duration_seconds": round(time.time() - start, 2),
        "device": device,
        "model_selection": "minimum validation SQL cross-entropy; test/challenge not used",
    }
    summary.update(
        {
            "training_config": {
                "epochs": epochs,
                "steps": steps,
                "patience": patience,
                "num_threads": num_threads,
                "weight_decay": 0.0,
            },
            "base_model_sha256": file_hash(Path(base) / "model.pt"),
            "tokenizer_sha256": file_hash(Path(base) / "tokenizer.json"),
            **provenance(data),
        }
    )
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "metadata.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name, typ, default in [
        ("base", str, "base_small"),
        ("data", str, "data"),
        ("out", str, "runs/adapter"),
        ("rank", int, 8),
        ("alpha", float, 16.0),
        ("dropout", float, 0.0),
        ("lr", float, 0.001),
        ("batch", int, 64),
        ("epochs", int, 6),
        ("seed", int, 42),
        ("patience", int, 3),
        ("steps", int, 0),
        ("device", str, None),
        ("num_threads", int, 2),
    ]:
        p.add_argument("--" + name.replace("_", "-"), type=typ, default=default)
    a = p.parse_args()
    train_lora(**vars(a))
