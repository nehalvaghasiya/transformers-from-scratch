"""Figures and Markdown tables derived from evaluated results, with no copied scores."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from .data import read_rows
from .inference import Prediction


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in results:
        groups.setdefault((row["system"], row["split"]), []).append(row)
    return [
        {
            "system": system,
            "split": split,
            "runs": len(rows),
            **{
                metric: statistics.mean(row[metric] for row in rows)
                for metric in (
                    "intent_accuracy",
                    "intent_macro_f1",
                    "priority_accuracy",
                    "priority_macro_f1",
                    "joint_accuracy",
                )
            },
            "joint_std": statistics.stdev(row["joint_accuracy"] for row in rows)
            if len(rows) > 1
            else 0.0,
        }
        for (system, split), rows in sorted(groups.items())
    ]


def render_report(
    output: Path,
    paths: dict[tuple[str, int], Path],
    data_dir: Path,
    selected_predictions: dict[str, list[Prediction]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads((output / "benchmark.json").read_text())
    summary = summarize(report["results"])
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "| System | Test joint accuracy | Challenge joint accuracy | Test intent macro-F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for system in sorted({row["system"] for row in summary}):
        test = next(row for row in summary if row["system"] == system and row["split"] == "test")
        challenge = next(
            row for row in summary if row["system"] == system and row["split"] == "challenge"
        )

        def percent(row: dict[str, Any]) -> str:
            return f"{100 * row['joint_accuracy']:.2f}%" + (
                f" ± {100 * row['joint_std']:.2f}" if row["runs"] > 1 else ""
            )

        lines.append(
            f"| {system} | {percent(test)} | {percent(challenge)} | {test['intent_macro_f1']:.3f} |"
        )
    (output / "results.md").write_text(
        "\n".join(lines) + "\n\n± denotes sample standard deviation across three fixed seeds. "
        "Classical baselines are single fitted systems.\n"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fbfaf7",
            "axes.facecolor": "#fbfaf7",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for split, ax in zip(("test", "challenge"), axes, strict=True):
        rows = [row for row in summary if row["split"] == split]
        ax.barh(
            [row["system"] for row in rows],
            [100 * row["joint_accuracy"] for row in rows],
            xerr=[100 * row["joint_std"] for row in rows],
            color=["#b66d42" if "tfidf" in row["system"] else "#326b70" for row in rows],
            capsize=3,
        )
        ax.set(xlim=(0, 100), xlabel="Joint accuracy (%)", title=split.title())
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("Ticket-v2 · Transformer and classical baselines", fontsize=16)
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    seed = report["selection"]["seed"]
    for variant in ("base", "cosine", "pool"):
        history = json.loads((paths[(variant, seed)] / "history.json").read_text())
        for ax, key in zip(axes, ("answer_ce", "val_answer_ce"), strict=True):
            ax.plot(
                [row["epoch"] for row in history],
                [row[key] for row in history],
                label=variant,
                marker=".",
            )
            ax.set(xlabel="Epoch", ylabel="Answer-token cross-entropy", yscale="log")
            ax.grid(alpha=0.2)
    axes[0].set_title(f"Training answer CE · seed {seed}")
    axes[1].set_title("Validation answer CE")
    axes[1].legend()
    fig.savefig(output / "learning-curves.png", dpi=160)
    plt.close(fig)
    errors: list[dict[str, Any]] = []
    for split, predictions in selected_predictions.items():
        rows = read_rows(data_dir / f"{split}.csv")
        from .evaluate import score

        metrics = score(rows, predictions)
        for key in ("intent", "priority"):
            confusion = metrics[f"{key}_confusion"]
            fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
            ax.imshow(confusion["counts"], cmap="YlGnBu")
            ax.set_xticks(
                range(len(confusion["prediction_labels"])),
                confusion["prediction_labels"],
                rotation=35,
                ha="right",
            )
            ax.set_yticks(range(len(confusion["labels"])), confusion["labels"])
            ax.set(xlabel="Predicted", ylabel="True", title=f"Selected Transformer · {split} {key}")
            for i, counts in enumerate(confusion["counts"]):
                for j, value in enumerate(counts):
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
            fig.savefig(output / f"{split}-{key}-confusion.png", dpi=140)
            plt.close(fig)
        wrong = [
            (row, prediction)
            for row, prediction in zip(rows, predictions, strict=True)
            if any(row[key] != prediction.get(key) for key in ("intent", "priority"))
        ]
        errors.extend(
            {
                "split": split,
                "ticket": row["text"],
                "truth": {key: row[key] for key in ("intent", "priority")},
                "prediction": prediction,
            }
            for row, prediction in wrong[:5]
        )
    (output / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")


def model_card(artifact: Path, report: Path) -> None:
    info = json.loads((artifact / "run_info.json").read_text())
    results = (report / "results.md").read_text()
    text = f"""---
license: mit
language: en
tags: [from-scratch, synthetic-data, text-classification]
---
# Ticket Transformer trained from scratch

Run `{info["run_id"]}`; seed {info["seed"]}; {info["parameter_count"]:,} Transformer
parameters; {info["trainable_parameters"]:,} total trainable parameters.
Word-level tokenizer fitted on training text. Answer-token causal cross-entropy,
with optional auxiliary pooled classifiers. Label tokens are constraint-decoded
and mapped to JSON; this is a task-specific model, not free-form JSON generation.

## Benchmark

{results}

These aggregate results describe a three-seed experiment, not just this artifact.
See the accompanying benchmark report for individual seeds, selection, uncertainty
and failures. All data is synthetic; no real-ticket accuracy is established.

## Loading

Install this repository's locked CPU environment and use
`pretraining predict --model ARTIFACT --text 'My invoice is wrong. No rush.'`.
The `run_info.json` records source, data and artifact hashes. Architecture and
vocabulary are stored with weights. Weights were trained from random initialization.

## Limitations

Six fixed intents, three urgency classes, English synthetic templates, short inputs.
Urgency phrases are shared across splits. Unknown words become UNK. Multiple intents,
implicit urgency, typos, long tickets and real data can cause failures. Confidence
is not calibrated. Training and inference CUDA memory has not been validated here.
"""
    (artifact / "README.md").write_text(text)
