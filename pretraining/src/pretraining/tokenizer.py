"""Auditable word vocabulary fitted exclusively on training text."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .data import INTENTS, PRIORITIES

PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]", re.IGNORECASE)
SPECIAL = ["<pad>", "<unk>", "<bos>", "<sep>", "<eos>"]
LABELS = [f"<intent:{value}>" for value in INTENTS] + [
    f"<priority:{value}>" for value in PRIORITIES
]


class WordTokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError("Vocabulary IDs must be unique and contiguous")
        if not set(SPECIAL + LABELS) <= vocab.keys():
            raise ValueError("Vocabulary is missing required special/label tokens")
        self.vocab = vocab
        self.itos = {value: key for key, value in vocab.items()}
        self.pad_id, self.unk_id, self.bos_id, self.sep_id, self.eos_id = (
            vocab[token] for token in SPECIAL
        )
        self.intent_ids = [vocab[f"<intent:{value}>"] for value in INTENTS]
        self.priority_ids = [vocab[f"<priority:{value}>"] for value in PRIORITIES]

    @classmethod
    def fit(cls, texts: list[str], min_count: int = 1, max_vocab: int = 3000) -> WordTokenizer:
        if min_count < 1 or max_vocab < len(SPECIAL + LABELS):
            raise ValueError("Invalid vocabulary limits")
        counts = Counter(token for text in texts for token in PATTERN.findall(text.lower()))
        base = SPECIAL + LABELS
        ordered = sorted(
            (
                (token, count)
                for token, count in counts.items()
                if count >= min_count and token not in base
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
        tokens = base + [token for token, _ in ordered[: max_vocab - len(base)]]
        return cls({token: index for index, token in enumerate(tokens)})

    def encode_text(self, text: str) -> list[int]:
        return [self.vocab.get(token, self.unk_id) for token in PATTERN.findall(text.lower())]

    def encode_prompt(self, text: str) -> list[int]:
        return [self.bos_id] + self.encode_text("support ticket: " + text) + [self.sep_id]

    def answer_ids(self, intent: str, priority: str) -> list[int]:
        return [self.vocab[f"<intent:{intent}>"], self.vocab[f"<priority:{priority}>"], self.eos_id]

    def decode_label(self, token_id: int, prefix: str) -> str | None:
        match = re.fullmatch(r"<" + prefix + r":([^>]+)>", self.itos.get(token_id, ""))
        return match.group(1) if match else None

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.vocab, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WordTokenizer:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
