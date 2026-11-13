"""Shared SQL generation and execution metrics; failures stay in the denominator."""

from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import torch

from .data import create_db, execute_sql
from .model import TinySQLLM
from .tokenizer import WordSQLTokenizer
from .utils import Record


@torch.inference_mode()
def generate_many(
    model: TinySQLLM,
    tok: WordSQLTokenizer,
    questions: list[str],
    max_new_tokens: int = 85,
    device: str = "cpu",
    batch: int = 16,
) -> list[str]:
    requested = torch.device(device)
    actual = next(model.parameters()).device

    if requested.type != actual.type or (
        requested.index is not None and requested.index != actual.index
    ):
        raise ValueError("Requested device differs from model device")

    prompts = [tok.format_input(question) for question in questions]
    good = [
        i
        for i, p in enumerate(prompts)
        if questions[i].strip() and len(p) < model.config.context
    ]
    predictions = [""] * len(questions)
    generated = model.generate_batch(
        [prompts[i] for i in good], tok.eos_id, max_new_tokens, batch
    )

    for index, tokens in zip(good, generated, strict=True):
        predictions[index] = tok.decode(tokens)

    return predictions


def validate_ids(rows: list[Record], expected_ids: list[str] | None = None) -> None:
    ids = [row["id"] for row in rows]

    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Evaluation requires nonempty rows with unique IDs")

    if expected_ids is not None and ids != expected_ids:
        raise ValueError("Prediction IDs or ordering do not match evaluation rows")


def _normalize(sql: str) -> str:
    # Preserve literal contents, including spaces and case, in exact-match scoring.
    parts = re.split(r"('(?:''|[^'])*')", sql.strip())

    return "".join(
        part if i % 2 else re.sub(r"\s+", "", part.lower())
        for i, part in enumerate(parts)
    )


def results_for(
    rows: list[Record],
    preds: Sequence[str | None],
    db_seeds: tuple[int, ...] = (2026, 2027),
) -> list[Record]:
    validate_ids(rows)

    if len(rows) != len(preds) or not db_seeds:
        raise ValueError("Need one prediction per row and at least one database")

    databases = [create_db(seed) for seed in db_seeds]
    results = []

    try:
        for row, prediction in zip(rows, preds, strict=True):
            sql = prediction or ""
            valid = correct = True
            message = None

            for db in databases:
                try:
                    truth = execute_sql(row["sql"], db)
                except (ValueError, sqlite3.Error) as error:
                    raise RuntimeError("Invalid gold SQL " + row["sql"]) from error

                try:
                    guess = execute_sql(sql, db)
                except (ValueError, sqlite3.Error) as error:
                    valid = correct = False
                    message = str(error)[:180]
                    break

                if truth != guess:
                    correct = False

            results.append(
                {
                    "id": row["id"],
                    "family": row["family"],
                    "question": row["question"],
                    "reference_sql": row["sql"],
                    "predicted_sql": sql,
                    "exact_match": _normalize(sql) == _normalize(row["sql"]),
                    "valid_sql": valid,
                    "execution_accuracy": correct,
                    "error": message,
                }
            )
    finally:
        for db in databases:
            db.close()

    return results


def report(rows: list[Record], expected_ids: list[str] | None = None) -> Record:
    validate_ids(rows, expected_ids)
    scores: Record = {"count": len(rows)}

    for metric in ("exact_match", "valid_sql", "execution_accuracy"):
        scores[metric] = float(np.mean([row[metric] for row in rows]))

    families: dict[str, list[Record]] = defaultdict(list)

    for row in rows:
        families[row["family"]].append(row)

    scores["by_family"] = {
        family: {
            "n": len(items),
            "execution_accuracy": float(
                np.mean([row["execution_accuracy"] for row in items])
            ),
        }
        for family, items in sorted(families.items())
    }

    return scores
