"""Training-only tokenizer: preserve SQL symbols and full numeric literals."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .data import SCHEMA
from .utils import Record

PATTERN = r"[a-z_]+|\d+|>=|<=|!=|[^\s]"
SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokens(text: str) -> list[str]:
    return re.findall(PATTERN, text.lower())


def prompt(question: str) -> str:
    return "schema: " + SCHEMA + " question: " + question.lower() + " sql:"


class WordSQLTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self.vocab = list(vocab)
        self.stoi = {s: i for i, s in enumerate(vocab)}
        if len(self.vocab) != len(self.stoi) or not set(SPECIAL) <= self.stoi.keys():
            raise ValueError("Invalid vocabulary")
        self.pad_id = self.stoi["<pad>"]
        self.unk_id = self.stoi["<unk>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    @classmethod
    def fit(cls, records: list[Record]) -> WordSQLTokenizer:
        counts: Counter[str] = Counter()
        for r in records:
            counts.update(tokens(prompt(r["question"])))
            counts.update(tokens(r["sql"]))
        return cls(
            SPECIAL
            + [
                k
                for k, v in sorted(counts.items(), key=lambda x: (-x[1], x[0]))
                if k not in SPECIAL
            ]
        )

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens(text)]

    def format_input(self, question: str) -> list[int]:
        return [self.bos_id] + self.encode(prompt(question))

    def format_train(self, row: Record, context: int = 160) -> tuple[list[int], list[int]]:
        p = self.format_input(row["question"])
        out = self.encode(row["sql"]) + [self.eos_id]
        seq = p + out
        if len(seq) > context:
            raise ValueError(f"Sequence {len(seq)} too long for context {context}")
        x = seq[:-1]
        y = [-100] * (len(p) - 1) + out
        assert len(x) == len(y)
        return x, y

    def decode(self, ids: list[int]) -> str:
        decoded = [self.vocab[x] if 0 <= x < len(self.vocab) else "<unk>" for x in ids]
        decoded = [x for x in decoded if x not in ("<pad>", "<bos>", "<eos>")]
        txt = " ".join(decoded)
        txt = re.sub(r"'\s+([a-z_]+)\s+'", r"'\1'", txt)
        return txt.strip()

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"vocab": self.vocab}, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> WordSQLTokenizer:
        return cls(json.loads(Path(path).read_text())["vocab"])
