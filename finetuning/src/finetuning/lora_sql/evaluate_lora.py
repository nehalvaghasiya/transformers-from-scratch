"""Compare frozen backbone, LoRA, and classical retrieval using identical questions.
Never use test/challenge results to pick LoRA hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from finetuning.tinysql.baselines import make_baselines
from finetuning.tinysql.evaluate import generate_many, report, results_for, validate_ids
from finetuning.tinysql.train import load_model
from finetuning.tinysql.utils import Record, file_hash, provenance, read_jsonl, write_jsonl

from .adapters import load_adapter


def evaluate(
    adapter: str | Path = "adapters/aug_r8_fast/adapter.pt",
    base: str | Path = "base_small",
    data: str | Path = "data",
    split: str = "val",
    count: int = 0,
    output: str | Path = "runs/evaluation/val_lora.json",
    device: str = "cpu",
    batch: int = 16,
    seed: int = 19,
    baselines: bool = True,
) -> Record:
    if split not in {"val", "test", "challenge"} or count < 0 or batch < 1:
        raise ValueError("Invalid split, count or batch size")
    if Path(output).exists():
        raise FileExistsError(output)
    if device == "cpu":
        torch.set_num_threads(2)
    t0 = time.time()
    rows = read_jsonl(Path(data) / f"{split}.jsonl")
    validate_ids(rows)
    rng = random.Random(seed)
    if count and count < len(rows):
        # fixed stratified sample (same indices for all models)
        families = sorted({x["family"] for x in rows})
        by = {f: [r for r in rows if r["family"] == f] for f in families}
        selected = []
        for f in families:
            rng.shuffle(by[f])
            selected.extend(by[f][: max(1, count // len(families))])
        if len(selected) < count:
            remain = [r for r in rows if r["id"] not in {x["id"] for x in selected}]
            rng.shuffle(remain)
            selected.extend(remain[: count - len(selected)])
        rows = selected[:count]
    qs = [r["question"] for r in rows]
    model, tok = load_model(base, device)
    base_preds = generate_many(model, tok, qs, device=device, batch=batch)
    load_adapter(model, adapter)
    lora_preds = generate_many(model, tok, qs, device=device, batch=batch)
    all_predictions = {"frozen_base": base_preds, "lora_raw": lora_preds}
    if baselines:
        baseline = make_baselines(data)
        for k, v in baseline.items():
            all_predictions[k] = v.predict_many(qs)
    detail = {k: results_for(rows, pred) for k, pred in all_predictions.items()}
    scores = {k: report(v, [r["id"] for r in rows]) for k, v in detail.items()}
    # paired bootstrap CI for gain (correctness difference) across examples
    before = np.array([int(x["execution_accuracy"]) for x in detail["frozen_base"]])
    after = np.array([int(x["execution_accuracy"]) for x in detail["lora_raw"]])
    difference = after - before
    rng_np = np.random.default_rng(seed)
    boots = np.array(
        [rng_np.choice(difference, len(difference), replace=True).mean() for _ in range(2000)]
    )
    summary: Record = {
        "split": split,
        "sample_count": len(rows),
        "seed_for_subset": seed,
        "base": str(base),
        "adapter": str(adapter),
        "model_scores": scores,
        "paired_gain_percent_points": 100 * float(difference.mean()),
        "paired_bootstrap_95pct_ci_pp": [float(x * 100) for x in np.percentile(boots, [2.5, 97.5])],
        "duration_seconds": round(time.time() - t0, 2),
        "note": (
            "All predictions evaluated against gold queries on two SQLite DBs. "
            "Held-out splits were known from earlier project; scores are "
            "exploratory."
        ),
    }
    summary.update(
        {
            "database_seeds": [2026, 2027],
            "decoding": {
                "mode": "greedy",
                "max_new_tokens": 85,
                "context_policy": "stop at original context boundary",
            },
            "base_model_sha256": file_hash(Path(base) / "model.pt"),
            "tokenizer_sha256": file_hash(Path(base) / "tokenizer.json"),
            "adapter_sha256": file_hash(adapter),
            **provenance(data),
        }
    )
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=2))
    write_jsonl(
        dest.with_name(dest.stem + "_predictions.jsonl"),
        [
            {
                "id": r["id"],
                "question": r["question"],
                "gold_sql": r["sql"],
                **{
                    k: {
                        "sql": detail[k][i]["predicted_sql"],
                        "correct": detail[k][i]["execution_accuracy"],
                    }
                    for k in detail
                },
            }
            for i, r in enumerate(rows)
        ],
    )
    print(
        json.dumps(
            {
                "split": split,
                "n": len(rows),
                "accuracies": {k: round(x["execution_accuracy"], 4) for k, x in scores.items()},
                "gain_pp": round(summary["paired_gain_percent_points"], 2),
                "95ci_pp": summary["paired_bootstrap_95pct_ci_pp"],
                "time": summary["duration_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for flag, typ, default in [
        ("adapter", str, "adapters/aug_r8_fast/adapter.pt"),
        ("base", str, "base_small"),
        ("data", str, "data"),
        ("split", str, "val"),
        ("count", int, 0),
        ("output", str, "runs/evaluation/val_lora.json"),
        ("device", str, "cpu"),
        ("batch", int, 16),
        ("seed", int, 19),
    ]:
        p.add_argument("--" + flag, type=typ, default=default)
    p.add_argument("--no-baselines", action="store_true")
    a = p.parse_args()
    d = vars(a)
    d["baselines"] = not d.pop("no_baselines")
    evaluate(**d)
