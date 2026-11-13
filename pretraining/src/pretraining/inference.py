"""Constrained label generation; independent of training and plotting dependencies."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .model import TinyTicketLM, make_pool_head, pool_prompt
from .tokenizer import WordTokenizer

type Prediction = dict[str, str | None]


def load_artifact(
    folder: str | Path, device: str = "cpu"
) -> tuple[TinyTicketLM, WordTokenizer, nn.ModuleDict | None, bool]:
    folder = Path(folder)
    model = TinyTicketLM.load_model(folder, device)
    tokenizer = WordTokenizer.load(folder / "vocab.json")
    head = None

    if (folder / "pool_head.pt").exists():
        with torch.random.fork_rng(devices=[]):
            head = make_pool_head(model.config.n_embd).to(device)

        head.load_state_dict(
            torch.load(folder / "pool_head.pt", map_location=device, weights_only=True)
        )
        head.eval()

    metadata = (
        json.loads((folder / "run_info.json").read_text())
        if (folder / "run_info.json").exists()
        else {}
    )
    legacy_pooling = (
        metadata.get("pooling", "legacy_include_separator")
        == "legacy_include_separator"
    )

    return model, tokenizer, head, legacy_pooling


@torch.no_grad()
def predict(
    model: TinyTicketLM,
    tokenizer: WordTokenizer,
    texts: list[str],
    device: str = "cpu",
    batch_size: int = 64,
    pool_head: nn.ModuleDict | None = None,
    alpha: float = 2.0,
    legacy_pooling: bool = False,
) -> list[Prediction]:
    if batch_size < 1 or alpha < 0:
        raise ValueError("Batch size must be positive and alpha nonnegative")

    model.eval()

    if pool_head is not None:
        pool_head.eval()

    results: list[Prediction] = []

    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        prompts = [tokenizer.encode_prompt(text) for text in chunk]
        output: list[Prediction] = [
            {
                "intent": None,
                "priority": None,
                "error": "empty input" if not text.strip() else "input too long",
            }
            for text in chunk
        ]
        good = [
            index
            for index, prompt in enumerate(prompts)
            if chunk[index].strip() and len(prompt) + 2 <= model.config.max_seq_len
        ]
        sequences = [prompts[index] for index in good]
        generated: list[list[int]] = []

        for step, allowed in enumerate((tokenizer.intent_ids, tokenizer.priority_ids)):
            if not sequences:
                break

            tokens = pad_sequence(
                [torch.tensor(sequence, dtype=torch.long) for sequence in sequences],
                batch_first=True,
                padding_value=tokenizer.pad_id,
            ).to(device)
            hidden = model.encode(tokens)
            positions = torch.tensor(
                [len(sequence) - 1 for sequence in sequences], device=device
            )
            label_logits = model.lm_head(hidden)[
                torch.arange(len(sequences), device=device), positions
            ][:, allowed]

            if step == 0 and pool_head is not None:
                if legacy_pooling:
                    mask = tokens != tokenizer.pad_id
                    pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(
                        1, keepdim=True
                    ).clamp_min(1)
                else:
                    pooled = pool_prompt(
                        hidden, tokens, tokenizer.pad_id, tokenizer.sep_id
                    )

                label_logits = label_logits + alpha * pool_head["intent"](pooled)

            labels = [allowed[index] for index in label_logits.argmax(dim=-1).tolist()]
            generated.append(labels)
            sequences = [
                sequence + [label]
                for sequence, label in zip(sequences, labels, strict=True)
            ]

        for index, original in enumerate(good):
            output[original] = {
                "intent": tokenizer.decode_label(generated[0][index], "intent"),
                "priority": tokenizer.decode_label(generated[1][index], "priority"),
            }

        results.extend(output)

    return results
