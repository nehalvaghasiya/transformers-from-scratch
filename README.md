# Transformers from Scratch & Fine-Tuning

An ML portfolio with readable implementations, reproducible experiments and measured comparisons.

| Track | Status | What it demonstrates |
| --- | --- | --- |
| [Pretraining / training from scratch](pretraining/README.md) | **Implemented and evaluated** | A 656K-parameter causal Transformer, training-only tokenizer, nine training runs, classical baselines, checkpoint recovery and a working demo |
| [Fine-tuning](finetuning/README.md) | Planned | LoRA adaptation of SmolLM2 for structured job extraction |

One repository, two independent Python packages. The ticket model is trained from
random weights to predict intent and priority; it is a small task-specific language
model. WikiText pretraining remains a separate future experiment.

![Actual ticket Transformer demo](pretraining/docs/assets/demo-preview.gif)

## Measured results

On the versioned synthetic ticket-v2 benchmark, pooled-head training plus dual-head
decoding reaches **96.51% ± 2.98 test joint accuracy** across three seeds. Word
TF-IDF reaches **99.51%**. On the challenge split, the same Transformer variant
reaches **73.15% ± 6.14**, while character TF-IDF reaches **93.89%**.

The comparisons show learning, the effect of auxiliary supervision, and the strength
of classical methods on small synthetic datasets. These scores do not establish
real-ticket performance. The [full case study](pretraining/README.md) includes every
seed, the validation selection, plots, uncertainty and failures.

![Measured model comparison](pretraining/reports/ticket-v2/comparison.png)

## Run locally

Use Linux x86-64 (including WSL2), Python 3.12.12 and uv 0.12.1. From the repository root:

```bash
uv python install 3.12.12
uv sync --locked --extra cpu --group pretraining --group demo
uv run --no-sync pretraining data prepare --config pretraining/configs/ticket-base.toml
uv run --no-sync pretraining train --config pretraining/configs/ticket-base.toml --run-id ticket-v2-base-42
uv run --no-sync pretraining demo --model pretraining/runs/ticket-v2-base-42 --report pretraining/reports/ticket-v2
```

The local demo opens at `http://127.0.0.1:7860`. This trains the **validation-selected**
Transformer; it is intentionally distinct from the best average test-scoring variant.
Training took about one minute per run on the recorded i5-13420H CPU with two Torch
threads. Timings vary by machine. No pretrained weights or dataset download is needed.

For the full three-seed benchmark, artifact verification, notebook and demo recording,
follow the [pretraining reproduction guide](pretraining/README.md#reproduce-the-experiments).

## Environment and checks

The independent track packages share one root `uv.lock` and `.venv`.
Choose exactly one profile: `cpu`, `cu126` (local GPU), or `cu128` (planned Space).
CUDA profiles are lock-resolved; CUDA training and the 3 GiB memory target have not
been verified. From a track directory, add `--project ..` to root sync commands.
Keep `--no-sync` on subsequent commands to preserve the selected profile.

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
uv lock --check
uv build --all-packages
uv run --no-sync python scripts/verify_profiles.py
```

[CPU CI](.github/workflows/cpu.yml) runs software checks and tiny fixtures; it does
not rerun the full experiment matrix. The original [P01 roadmap](plan.md) retains
its issue IDs and separate WikiText/LoRA contracts. Full datasets, checkpoints,
environments and videos remain outside Git. Compact reports and demo previews are included.
