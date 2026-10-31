"""One answer-token training loop, with optional auxiliary heads and epoch recovery."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import random
import resource
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import TicketDataset, batchify, read_rows
from .model import ModelConfig, TinyTicketLM, make_pool_head, pool_prompt
from .tokenizer import WordTokenizer


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes() -> dict[str, str]:
    scientific_modules = {
        "model",
        "tokenizer",
        "data",
        "config",
        "train",
        "inference",
        "baselines",
        "evaluate",
        "benchmark",
        "report",
    }
    return {
        path.name: file_hash(path)
        for path in sorted(Path(__file__).parent.glob("*.py"))
        if path.stem in scientific_modules
    }


def lr_multiplier(step: int, total: int, warmup: int, schedule: str) -> float:
    if schedule == "flat":
        return 1.0
    if step < warmup:
        return max(step / max(warmup, 1), 1e-3)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate_loss(
    model: TinyTicketLM, loader: Iterable[tuple[Tensor, Tensor]], device: str
) -> float:
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        _, loss = model(x.to(device), y.to(device))
        assert loss is not None
        valid = int((y != -100).sum())
        total += loss.item() * valid
        count += valid
    if not count:
        raise ValueError("No supervised validation tokens")
    return total / count


def train(
    config: ExperimentConfig,
    run_id: str,
    *,
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> Path:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("Run ID must be one directory name")
    directory = Path(config.runs_dir) / run_id
    if directory.exists() and not resume:
        raise FileExistsError(f"Run already exists: {directory}")
    if resume and not (directory / "recovery.pt").is_file():
        raise FileNotFoundError("No epoch-boundary recovery checkpoint")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)
    data_dir = Path(config.data_dir)
    data_hashes = {
        split: file_hash(data_dir / f"{split}.csv")
        for split in ("train", "val", "test", "challenge")
    }
    rows = read_rows(data_dir / "train.csv")
    if config.limit_train:
        rows = rows[: config.limit_train]
    tokenizer = WordTokenizer.fit([row["text"] for row in rows])
    architecture = ModelConfig(
        vocab_size=len(tokenizer.vocab),
        max_seq_len=config.max_seq_len,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
    )
    train_data = TicketDataset(rows, tokenizer, architecture.max_seq_len)
    val_data = TicketDataset(read_rows(data_dir / "val.csv"), tokenizer, architecture.max_seq_len)
    generators = [torch.Generator().manual_seed(config.seed + offset) for offset in (0, 1)]
    collate = partial(batchify, pad_id=tokenizer.pad_id)
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generators[0] if config.loader_rng == "isolated" else None,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate,
        generator=generators[1] if config.loader_rng == "isolated" else None,
    )
    model = TinyTicketLM(architecture).to(config.device)
    initial_hash = hashlib.sha256(
        b"".join(parameter.detach().cpu().numpy().tobytes() for parameter in model.parameters())
    ).hexdigest()
    head = make_pool_head(config.n_embd).to(config.device) if config.pool_weight else None
    parameters = list(model.parameters()) + (list(head.parameters()) if head is not None else [])
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    total_steps = min(config.max_steps or config.epochs * len(loader), config.epochs * len(loader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            lr_multiplier,
            total=total_steps,
            warmup=int(config.warmup_ratio * total_steps),
            schedule=config.schedule,
        ),
    )
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = config.fingerprint()
    code = source_hashes()
    history: list[dict[str, float | int]] = []
    trace: list[dict[str, float | int | str]] = []
    best_val, first_epoch, step, elapsed_before = float("inf"), 1, 0, 0.0
    if resume:
        state = torch.load(directory / "recovery.pt", map_location="cpu", weights_only=True)
        if (
            state["config_hash"] != fingerprint
            or state["data_hashes"] != data_hashes
            or state["source_hashes"] != code
        ):
            raise ValueError("Resume requires identical config, data and source code")
        model.load_state_dict(state["model"])
        if head is not None:
            head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        for generator, rng in zip(generators, state["loader_rng"], strict=True):
            generator.set_state(rng)
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, trace = state["history"], state["trace"]
        best_val, step = state["best_val"], state["step"]
        first_epoch, elapsed_before = state["epoch"] + 1, state["elapsed_seconds"]
        if step >= total_steps:
            raise ValueError("This run has already completed its configured update budget")
    else:
        torch.save(model.state_dict(), directory / "initial.pt")
        tokenizer.save(directory / "vocab.json")
        write_json(directory / "experiment.json", asdict(config))
    lookup_intent = torch.full((len(tokenizer.vocab),), -1, dtype=torch.long, device=config.device)
    lookup_priority = torch.full_like(lookup_intent, -1)
    for index, token in enumerate(tokenizer.intent_ids):
        lookup_intent[token] = index
    for index, token in enumerate(tokenizer.priority_ids):
        lookup_priority[token] = index
    start = time.perf_counter()
    info: dict[str, Any] = {
        "run_id": run_id,
        "status": "running",
        "seed": config.seed,
        "dataset": config.dataset,
        "config_hash": fingerprint,
        "data_hashes": data_hashes,
        "source_hashes": code,
        "pooling": "exclude_separator",
        "parameter_count": model.count_parameters(),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "initial_parameters_sha256": initial_hash,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "threads": config.threads,
        "device": config.device,
        "train_examples": len(rows),
        "architecture": asdict(architecture),
        "selection": "minimum validation answer-token cross-entropy; earliest epoch on ties",
    }
    if resume:
        info["selected_epoch"] = state["selected_epoch"]
    root = Path(__file__).resolve().parents[3]
    if (root / "uv.lock").exists():
        info["lock_sha256"] = file_hash(root / "uv.lock")
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
        )
        info["git_commit"] = revision.stdout.strip()
        info["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
            ).stdout
        )
    write_json(directory / "run_info.json", info)
    for epoch in range(first_epoch, config.epochs + 1):
        if step >= total_steps:
            break
        model.train()
        if head is not None:
            head.train()
        sums = {"answer_ce": 0.0, "auxiliary_ce": 0.0, "total_loss": 0.0}
        count = 0
        for x_cpu, y_cpu in loader:
            x, y = x_cpu.to(config.device), y_cpu.to(config.device)
            hidden = model.encode(x)
            logits = model.lm_head(hidden)
            answer = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
            )
            auxiliary = torch.zeros((), device=config.device)
            if head is not None:
                pooled = pool_prompt(hidden, x, tokenizer.pad_id, tokenizer.sep_id)
                first = (y != -100).long().argmax(dim=1)
                indices = torch.arange(x.size(0), device=config.device)
                auxiliary = F.cross_entropy(
                    head["intent"](pooled), lookup_intent[y[indices, first]]
                ) + F.cross_entropy(
                    head["priority"](pooled), lookup_priority[y[indices, first + 1]]
                )
            loss = answer + config.pool_weight * auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            if step < 3:
                trace.append(
                    {
                        "step": step,
                        "batch_sha256": hashlib.sha256(x_cpu.numpy().tobytes()).hexdigest(),
                        "answer_ce": answer.item(),
                        "auxiliary_ce": auxiliary.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "gradient_norm": float(norm),
                    }
                )
            optimizer.step()
            scheduler.step()
            valid = int((y != -100).sum())
            for key, value in (
                ("answer_ce", answer),
                ("auxiliary_ce", auxiliary),
                ("total_loss", loss),
            ):
                sums[key] += value.item() * valid
            count += valid
            step += 1
            if step >= total_steps:
                break
        val_loss = evaluate_loss(model, val_loader, config.device)
        history.append(
            {
                "epoch": epoch,
                "steps": step,
                **{key: value / count for key, value in sums.items()},
                "val_answer_ce": val_loss,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            info["selected_epoch"] = epoch
            model.save_model(directory)
            if head is not None:
                torch.save(head.state_dict(), directory / "pool_head.pt")
        elapsed = elapsed_before + time.perf_counter() - start
        torch.save(
            {
                "config_hash": fingerprint,
                "data_hashes": data_hashes,
                "source_hashes": code,
                "model": model.state_dict(),
                "head": head.state_dict() if head is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng": torch.get_rng_state(),
                "python_rng": random.getstate(),
                "loader_rng": [generator.get_state() for generator in generators],
                "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
                "history": history,
                "trace": trace,
                "best_val": best_val,
                "selected_epoch": info["selected_epoch"],
                "epoch": epoch,
                "step": step,
                "elapsed_seconds": elapsed,
            },
            directory / "recovery.pt",
        )
        info.update({"steps": step, "best_val_answer_ce": best_val, "elapsed_seconds": elapsed})
        write_json(directory / "history.json", history)
        write_json(directory / "trace.json", trace)
        print(json.dumps({"run": run_id, **history[-1]}), flush=True)
        if stop_after_epoch is not None and epoch >= stop_after_epoch and step < total_steps:
            info["status"] = "interrupted"
            write_json(directory / "run_info.json", info)
            return directory
    info.update(
        {
            "status": "completed",
            "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
    )
    if config.device == "cuda":
        info["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        info["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    info["artifact_hashes"] = {
        path.name: file_hash(path) for path in directory.glob("*.pt") if path.name != "recovery.pt"
    }
    write_json(directory / "run_info.json", info)
    return directory
