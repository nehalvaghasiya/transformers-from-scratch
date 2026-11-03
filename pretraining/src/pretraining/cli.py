"""Command dispatch for implemented ticket-training workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a ticket Transformer from scratch."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="Generate versioned synthetic datasets")
    prepare = data.add_subparsers(dest="data_command", required=True).add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    training = commands.add_parser("train", help="Train without accessing final evaluation")
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--run-id", required=True)
    training.add_argument("--seed", type=int)
    training.add_argument("--resume", type=Path, help="The run's recovery.pt at an epoch boundary")
    evaluation = commands.add_parser("evaluate", help="Score an explicit checkpoint")
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--model", type=Path, required=True)
    evaluation.add_argument("--split", choices=["val", "test", "challenge"], required=True)
    evaluation.add_argument("--mode", choices=["generative", "dual"], default="generative")
    evaluation.add_argument("--output", type=Path)
    prediction = commands.add_parser("predict", help="Generate intent and priority labels")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--text", required=True)
    prediction.add_argument("--mode", choices=["generative", "dual"], default="generative")
    demo = commands.add_parser("demo", help="Launch the local Gradio application")
    demo.add_argument("--model", type=Path, required=True)
    demo.add_argument("--report", type=Path)
    demo.add_argument("--port", type=int, default=7860)
    benchmark = commands.add_parser("benchmark", help="Run the frozen three-seed experiment matrix")
    benchmark.add_argument("--config", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    artifact = commands.add_parser("artifact", help="Export or retrieve a checksummed model bundle")
    operations = artifact.add_subparsers(dest="artifact_command", required=True)
    export = operations.add_parser("export")
    export.add_argument("--model", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    fetch = operations.add_parser("fetch")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--sha256", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "artifact":
        from .artifact import export_artifact, fetch_artifact

        if args.artifact_command == "export":
            print(json.dumps(export_artifact(args.model, args.output), indent=2))
        else:
            fetch_artifact(args.url, args.sha256, args.output)
        return
    if args.command in {"data", "train", "evaluate", "benchmark"}:
        from .config import read_config

        config = read_config(args.config)
        if args.command == "data":
            from .data import prepare_dataset

            print(
                json.dumps(
                    prepare_dataset(Path(config.data_dir), config.dataset, config.data_seed),
                    indent=2,
                )
            )
        elif args.command == "train":
            from .train import train

            if args.seed is not None:
                config = replace(config, seed=args.seed)
            if (
                args.resume is not None
                and args.resume.resolve()
                != (Path(config.runs_dir) / args.run_id / "recovery.pt").resolve()
            ):
                parser.error("--resume must name this run's recovery.pt")
            print(train(config, args.run_id, resume=args.resume is not None))
        elif args.command == "evaluate":
            import torch

            from .evaluate import evaluate_artifact

            torch.set_num_threads(config.threads)
            print(
                json.dumps(
                    evaluate_artifact(
                        args.model,
                        Path(config.data_dir),
                        args.split,
                        args.output or args.model / "evaluation",
                        args.mode,
                        config.device,
                    ),
                    indent=2,
                )
            )
        else:
            from .benchmark import run_benchmark

            run_benchmark(args.config, args.output)
        return
    import torch

    torch.set_num_threads(2)
    if args.command == "demo":
        from .demo import launch

        launch(args.model, args.report, args.port)
    else:
        from .inference import load_artifact, predict

        model, tokenizer, head, legacy = load_artifact(args.model)
        if args.mode == "dual" and head is None:
            parser.error("This artifact has no auxiliary heads")
        print(
            json.dumps(
                predict(
                    model,
                    tokenizer,
                    [args.text],
                    pool_head=head if args.mode == "dual" else None,
                    legacy_pooling=legacy,
                )[0],
                indent=2,
            )
        )
