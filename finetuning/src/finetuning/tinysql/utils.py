"""JSON records, seeding and provenance for the SQL experiment."""

from __future__ import annotations

import hashlib
import json
import platform
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

type Record = dict[str, Any]


def read_jsonl(path: str | Path) -> list[Record]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[Record]) -> None:
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def provenance(data_dir: str | Path) -> Record:
    root = Path(__file__).parents[1]
    return {
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": torch.get_num_threads(),
        },
        "source_hashes": {
            str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))
        },
        "data_hashes": {p.name: file_hash(p) for p in sorted(Path(data_dir).glob("*.jsonl"))},
    }
