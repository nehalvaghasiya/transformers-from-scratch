"""Small TOML experiment contract; paths are relative to the configuration file."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "ticket-v2"
    data_dir: str = "../data/ticket-v2"
    runs_dir: str = "../runs"
    seed: int = 42
    data_seed: int = 2026
    threads: int = 2
    device: str = "cpu"
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    max_seq_len: int = 112
    dropout: float = 0.1
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 8e-4
    weight_decay: float = 0.01
    schedule: str = "cosine"
    warmup_ratio: float = 0.1
    pool_weight: float = 2.0
    max_steps: int = 0
    limit_train: int = 0
    # Canonical runs isolate loader RNG; historical experiments did not.
    loader_rng: str = "isolated"

    def __post_init__(self) -> None:
        if self.dataset not in {"ticket-v1", "ticket-v2"} or self.device not in {"cpu", "cuda"}:
            raise ValueError("Unsupported dataset or device")
        if self.schedule not in {"flat", "cosine"} or self.loader_rng not in {"isolated", "legacy"}:
            raise ValueError("Unsupported schedule or loader RNG")
        if (
            min(
                self.threads,
                self.epochs,
                self.batch_size,
                self.n_layer,
                self.n_head,
                self.n_embd,
                self.max_seq_len,
            )
            < 1
        ):
            raise ValueError("Counts must be positive")
        if self.max_steps < 0 or self.limit_train < 0 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("Invalid run limits or warmup")
        if not 0 <= self.dropout < 1 or self.n_embd % self.n_head:
            raise ValueError("Invalid dropout or attention dimensions")
        if (
            any(
                not math.isfinite(value) or value < 0
                for value in (self.learning_rate, self.weight_decay, self.pool_weight)
            )
            or self.learning_rate == 0
        ):
            raise ValueError(
                "Optimizer settings must be finite and nonnegative; LR must be positive"
            )

    def fingerprint(self) -> str:
        values = asdict(self)
        for key in ("data_dir", "runs_dir"):
            values.pop(key)
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def read_config(path: str | Path) -> ExperimentConfig:
    path = Path(path).resolve()
    values = tomllib.loads(path.read_text())
    for key in ("data_dir", "runs_dir"):
        values[key] = str(
            (
                path.parent / values.get(key, ExperimentConfig.__dataclass_fields__[key].default)
            ).resolve()
        )
    return ExperimentConfig(**values)
