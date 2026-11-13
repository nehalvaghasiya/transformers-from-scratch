"""Metrics retain rejected examples; final scoring is separate from training."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from .data import INTENTS, PRIORITIES, Ticket, read_rows
from .inference import Prediction, load_artifact, predict


def score(rows: list[Ticket], predictions: list[Prediction]) -> dict[str, Any]:
    if not rows or len(rows) != len(predictions):
        raise ValueError("Need one prediction per nonempty evaluation row")

    result: dict[str, Any] = {"examples": len(rows)}

    for key, labels in (("intent", INTENTS), ("priority", PRIORITIES)):
        per_class = {}
        confusion = [[0] * (len(labels) + 1) for _ in labels]

        for row, prediction in zip(rows, predictions, strict=True):
            if row[key] not in labels:
                raise ValueError("Unknown ground-truth label")

            value = prediction.get(key)
            confusion[labels.index(row[key])][
                labels.index(value) if value in labels else len(labels)
            ] += 1

        for label in labels:
            tp = sum(
                row[key] == label and prediction.get(key) == label
                for row, prediction in zip(rows, predictions, strict=True)
            )
            support = sum(row[key] == label for row in rows)
            predicted = sum(prediction.get(key) == label for prediction in predictions)
            precision = tp / predicted if predicted else 0.0
            recall = tp / support if support else 0.0
            per_class[label] = {
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0,
            }

        result[f"{key}_accuracy"] = sum(
            row[key] == prediction.get(key)
            for row, prediction in zip(rows, predictions, strict=True)
        ) / len(rows)
        result[f"{key}_macro_f1"] = statistics.mean(
            value["f1"] for value in per_class.values()
        )
        result[f"{key}_per_class"] = per_class
        result[f"{key}_confusion"] = {
            "labels": list(labels),
            "prediction_labels": [*labels, "<invalid>"],
            "counts": confusion,
        }

    result["joint_accuracy"] = sum(
        all(row[key] == prediction.get(key) for key in ("intent", "priority"))
        for row, prediction in zip(rows, predictions, strict=True)
    ) / len(rows)
    result["rejected_rate"] = sum(
        bool(prediction.get("error")) for prediction in predictions
    ) / len(rows)

    return result


def paired_interval(
    rows: list[Ticket],
    left: list[Prediction],
    right: list[Prediction],
    samples: int = 2000,
) -> dict[str, Any]:
    """Paired template-cluster bootstrap of joint-accuracy difference, left minus right."""
    if len(rows) != len(left) or len(rows) != len(right) or samples < 1:
        raise ValueError("Bootstrap inputs must align")

    groups: dict[str, list[int]] = {}

    for index, row in enumerate(rows):
        groups.setdefault(row["template_id"], []).append(index)

    if not groups:
        raise ValueError("No template groups")

    differences = [
        int(all(row[key] == a.get(key) for key in ("intent", "priority")))
        - int(all(row[key] == b.get(key) for key in ("intent", "priority")))
        for row, a, b in zip(rows, left, right, strict=True)
    ]
    group_sums = [
        (sum(differences[index] for index in indices), len(indices))
        for indices in groups.values()
    ]
    rng = random.Random(2026)
    values = []

    for _ in range(samples):
        selected = rng.choices(group_sums, k=len(groups))
        values.append(
            sum(value for value, _ in selected) / sum(size for _, size in selected)
        )

    values.sort()

    return {
        "difference": statistics.mean(differences),
        "lower_95": values[int(0.025 * (samples - 1))],
        "upper_95": values[int(0.975 * (samples - 1))],
        "template_groups": len(groups),
        "samples": samples,
        "seed": 2026,
        "method": "paired template-cluster bootstrap; synthetic-template uncertainty only",
    }


def write_predictions(
    path: Path, rows: list[Ticket], predictions: list[Prediction]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as stream:
        fields = [
            "id",
            "template_id",
            "true_intent",
            "pred_intent",
            "true_priority",
            "pred_priority",
            "error",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()

        for row, prediction in zip(rows, predictions, strict=True):
            writer.writerow(
                {
                    "id": row["id"],
                    "template_id": row["template_id"],
                    "true_intent": row["intent"],
                    "pred_intent": prediction.get("intent"),
                    "true_priority": row["priority"],
                    "pred_priority": prediction.get("priority"),
                    "error": prediction.get("error", ""),
                }
            )


def evaluate_artifact(
    artifact: Path,
    data_dir: Path,
    split: str,
    output: Path,
    mode: str = "generative",
    device: str = "cpu",
) -> dict[str, Any]:
    if split not in {"val", "test", "challenge"} or mode not in {"generative", "dual"}:
        raise ValueError("Invalid evaluation split or mode")

    model, tokenizer, head, legacy = load_artifact(artifact, device)

    if mode == "dual" and head is None:
        raise ValueError("This artifact has no auxiliary heads")

    rows = read_rows(data_dir / f"{split}.csv")
    texts = [row["text"] for row in rows]
    kwargs = {"pool_head": head if mode == "dual" else None, "legacy_pooling": legacy}
    predictions = predict(model, tokenizer, texts, device=device, **kwargs)  # type: ignore[arg-type]
    metrics = score(rows, predictions)

    tokens = [token for text in texts for token in tokenizer.encode_text(text)]
    metrics["known_vocabulary_fraction"] = sum(
        token != tokenizer.unk_id for token in tokens
    ) / max(1, len(tokens))
    # Single-ticket timings include tokenization and decoding, after one warm-up.
    predict(model, tokenizer, texts[:1], device=device, **kwargs)  # type: ignore[arg-type]
    timings = []

    for text in texts[:10]:
        for _ in range(3):
            if device == "cuda":
                import torch

                torch.cuda.synchronize()

            start = time.perf_counter()
            predict(model, tokenizer, [text], device=device, **kwargs)  # type: ignore[arg-type]

            if device == "cuda":
                torch.cuda.synchronize()

            timings.append(1000 * (time.perf_counter() - start))

    timings.sort()
    metrics["latency_ms"] = {
        "p50": statistics.median(timings),
        "p95": timings[min(len(timings) - 1, int(0.95 * len(timings)))],
        "samples": len(timings),
        "scope": "one ticket; warm tokenization plus decoding",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_predictions(output / f"{split}-{mode}-predictions.csv", rows, predictions)
    result = {
        "split": split,
        "mode": mode,
        "artifact": str(artifact),
        "model_sha256": hashlib.sha256(
            (artifact / "model.pt").read_bytes()
        ).hexdigest(),
        "data_sha256": hashlib.sha256(
            (data_dir / f"{split}.csv").read_bytes()
        ).hexdigest(),
        **metrics,
    }
    (output / f"{split}-{mode}.json").write_text(json.dumps(result, indent=2) + "\n")

    return result
