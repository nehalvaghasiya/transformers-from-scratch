"""Frozen three-seed benchmark: select on validation before reading final predictions."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from .baselines import fit_baselines
from .config import read_config
from .data import read_rows
from .evaluate import evaluate_artifact, paired_interval, score, write_predictions
from .inference import load_artifact, predict
from .train import file_hash, source_hashes, write_json

SEEDS = (42, 7, 123)
VARIANTS = (("base", "flat", 0.0), ("cosine", "cosine", 0.0), ("pool", "cosine", 2.0))


def run_benchmark(config_path: Path, output: Path) -> None:
    config = read_config(config_path)

    if config.dataset != "ticket-v2" or config.max_steps or config.limit_train:
        raise ValueError(
            "The full benchmark requires ticket-v2 without shortened training"
        )

    torch.set_num_threads(config.threads)
    data_dir = Path(config.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "dataset": config.dataset,
        "data_hashes": {
            split: file_hash(data_dir / f"{split}.csv")
            for split in ("train", "val", "test", "challenge")
        },
        "source_hashes": source_hashes(),
        "seeds": SEEDS,
        "variants": VARIANTS,
        "config": asdict(config),
        "dual_alpha": 2.0,
        "selection": (
            "maximum mean validation joint accuracy, then intent macro-F1, "
            "then minimum mean validation answer CE; lexical variant/mode tie break; "
            "demo seed by validation CE then seed"
        ),
    }
    contract = json.loads(json.dumps(contract))
    contract_file = output / "contract.json"

    if contract_file.exists() and json.loads(contract_file.read_text()) != contract:
        raise ValueError("Benchmark contract changed; use a new report/run directory")

    write_json(contract_file, contract)
    validation = read_rows(data_dir / "val.csv")
    candidates: list[dict[str, Any]] = []
    paths: dict[tuple[str, int], Path] = {}

    for variant, schedule, pool in VARIANTS:
        for seed in SEEDS:
            current = replace(config, seed=seed, schedule=schedule, pool_weight=pool)
            run_id = f"{config.dataset}-{variant}-{seed}"
            artifact = Path(config.runs_dir) / run_id
            info_file = artifact / "run_info.json"
            completed = (
                info_file.exists()
                and json.loads(info_file.read_text()).get("status") == "completed"
            )

            if completed:
                previous = json.loads(info_file.read_text())

                if (
                    previous["config_hash"] != current.fingerprint()
                    or previous["source_hashes"] != contract["source_hashes"]
                    or previous["data_hashes"] != contract["data_hashes"]
                ):
                    raise ValueError(f"Existing run differs: {run_id}")
            else:
                child = (
                    "import json,sys; from pretraining.config import ExperimentConfig; "
                    "from pretraining.train import train; "
                    "train(ExperimentConfig(**json.loads(sys.argv[1])),sys.argv[2],"
                    "resume=sys.argv[3]=='True')"
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        child,
                        json.dumps(asdict(current)),
                        run_id,
                        str((artifact / "recovery.pt").exists()),
                    ],
                    check=True,
                    env={
                        **os.environ,
                        "OMP_NUM_THREADS": str(config.threads),
                        "MKL_NUM_THREADS": str(config.threads),
                    },
                )

            paths[(variant, seed)] = artifact
            model, tokenizer, head, legacy = load_artifact(artifact, config.device)
            info = json.loads(info_file.read_text())

            for mode in ("generative", "dual") if pool else ("generative",):
                predictions = predict(
                    model,
                    tokenizer,
                    [row["text"] for row in validation],
                    device=config.device,
                    pool_head=head if mode == "dual" else None,
                    legacy_pooling=legacy,
                )
                candidates.append(
                    {
                        "variant": variant,
                        "mode": mode,
                        "seed": seed,
                        "val_answer_ce": info["best_val_answer_ce"],
                        **score(validation, predictions),
                    }
                )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for candidate in candidates:
        grouped.setdefault((candidate["variant"], candidate["mode"]), []).append(
            candidate
        )

    winner = min(
        grouped,
        key=lambda key: (
            -statistics.mean(row["joint_accuracy"] for row in grouped[key]),
            -statistics.mean(row["intent_macro_f1"] for row in grouped[key]),
            statistics.mean(row["val_answer_ce"] for row in grouped[key]),
            key,
        ),
    )
    selected = min(grouped[winner], key=lambda row: (row["val_answer_ce"], row["seed"]))
    selected_path = paths[(winner[0], selected["seed"])]
    baselines = fit_baselines(read_rows(data_dir / "train.csv"))
    baseline_validation = {
        baseline.name: score(
            validation, baseline.predict([row["text"] for row in validation])
        )
        for baseline in baselines
    }
    baseline_name = min(
        baseline_validation,
        key=lambda name: (
            -baseline_validation[name]["joint_accuracy"],
            -baseline_validation[name]["intent_macro_f1"],
            name,
        ),
    )
    # Written before final test/challenge inference. Later reruns use the same contract.
    selection = {
        "variant": winner[0],
        "mode": winner[1],
        "seed": selected["seed"],
        "artifact": str(selected_path),
        "comparison_baseline": baseline_name,
        "candidates": candidates,
        "baseline_validation": baseline_validation,
        "test_used_for_selection": False,
    }
    write_json(output / "selection.json", selection)
    results = []
    selected_predictions = {}
    baseline_predictions = {}

    for split in ("test", "challenge"):
        rows = read_rows(data_dir / f"{split}.csv")
        texts = [row["text"] for row in rows]

        for baseline in baselines:
            predictions = baseline.predict(texts)
            results.append(
                {
                    "split": split,
                    "system": baseline.name,
                    "seed": None,
                    **score(rows, predictions),
                }
            )
            write_predictions(
                output / f"{split}-{baseline.name}.csv", rows, predictions
            )

            if baseline.name == baseline_name:
                baseline_predictions[split] = predictions

        for variant, _, pool in VARIANTS:
            for seed in SEEDS:
                artifact = paths[(variant, seed)]

                for mode in ("generative", "dual") if pool else ("generative",):
                    metrics = evaluate_artifact(
                        artifact,
                        data_dir,
                        split,
                        artifact / "evaluation",
                        mode,
                        config.device,
                    )
                    system = f"{variant}-{mode}"
                    results.append({"system": system, "seed": seed, **metrics})

                    if (
                        variant == winner[0]
                        and mode == winner[1]
                        and seed == selected["seed"]
                    ):
                        model, tokenizer, head, legacy = load_artifact(
                            artifact, config.device
                        )
                        selected_predictions[split] = predict(
                            model,
                            tokenizer,
                            texts,
                            device=config.device,
                            pool_head=head if mode == "dual" else None,
                            legacy_pooling=legacy,
                        )

                    write_predictions_path = (
                        artifact / "evaluation" / f"{split}-{mode}-predictions.csv"
                    )
                    (output / f"{split}-{system}-{seed}.csv").write_bytes(
                        write_predictions_path.read_bytes()
                    )

        for seed in SEEDS:
            artifact = paths[("base", seed)]
            model, tokenizer, _, _ = load_artifact(artifact, config.device)
            model.load_state_dict(
                torch.load(
                    artifact / "initial.pt",
                    map_location=config.device,
                    weights_only=True,
                )
            )
            predictions = predict(model, tokenizer, texts, device=config.device)
            results.append(
                {
                    "split": split,
                    "system": "random-init",
                    "seed": seed,
                    **score(rows, predictions),
                }
            )

    intervals = {
        split: paired_interval(
            read_rows(data_dir / f"{split}.csv"),
            selected_predictions[split],
            baseline_predictions[split],
        )
        for split in ("test", "challenge")
    }
    report = {
        "dataset": config.dataset,
        "selection": {
            key: value
            for key, value in selection.items()
            if key not in {"candidates", "baseline_validation"}
        },
        "results": results,
        "paired_intervals": intervals,
        "runs": {
            f"{variant}-{seed}": json.loads((artifact / "run_info.json").read_text())
            for (variant, seed), artifact in paths.items()
        },
    }
    write_json(output / "benchmark.json", report)
    from .report import render_report

    render_report(output, paths, data_dir, selected_predictions)
    print(
        f"Benchmark complete: {output}; selected {selected_path} ({winner[1]})",
        flush=True,
    )
