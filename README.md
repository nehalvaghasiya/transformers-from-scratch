# Small LLM Pretraining & Fine-Tuning

A portfolio project demonstrating how to build a language model from scratch and
adapt a pretrained model, with reproducible experiments and honest evaluation.

| Track | Planned work |
| --- | --- |
| [Pretraining](pretraining/README.md) | Integrate the existing scratch-trained ticket Transformer; retain WikiText/BPE language modeling as a separate planned experiment |
| [Fine-tuning](finetuning/README.md) | Adapt SmolLM2-135M-Instruct with LoRA for structured job extraction; compare against rules and untouched-model baselines |

**Current status:** the workspace packages contain environment setup. An existing
trained ticket Transformer, comparisons and saved results are available in
`pretrained/`; its [integration plan](pretraining/integration-plan.md) records the
artifact audit and proposed migration. WikiText training and LoRA extraction remain
planned. The original issue roadmap and contracts are in [plan.md](plan.md).

The showcase will center on readable model and training code, data preparation,
correctness tests, learning curves, ablations, error analysis and reproducible
results. Each track owns its source, tests, configs and reports. Full datasets,
weights and checkpoints stay out of Git.

## Setup

Use Linux x86-64 (including WSL2), Python 3.12 and uv 0.12.1.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```bash
uv python install 3.12.12
uv sync --locked --extra cpu
uv run --no-sync python scripts/smoke_cpu.py
```

The smoke check runs one tiny local LoRA optimizer update without downloading a
model or dataset. It verifies library compatibility; it is not a training result.

Choose exactly one accelerator profile:

| Profile | PyTorch | Use |
| --- | --- | --- |
| `cpu` | 2.8.0+cpu | Local software checks |
| `cu126` | 2.8.0+cu126 | Local RTX 4050 experiments |
| `cu128` | 2.8.0+cu128 | Planned Hugging Face Space GPU |

Replace `cpu` in the sync command to select a CUDA profile. Add `--group demo`
for Gradio. The CUDA profiles are lock-resolved; GPU execution and the planned
3 GiB training target still need measurement.

The two independent packages share one root `uv.lock` and `.venv`. From a track
directory, use `uv sync --project .. --locked --extra cpu`. Keep `--no-sync` on
subsequent `uv run` commands to preserve the selected profile, or repeat the same
profile flags when syncing. Development tools are included by default; use
`--no-dev` for runtime dependencies only.

## Development checks

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv lock --check
uv build --all-packages
uv run --no-sync python scripts/verify_profiles.py
```

Add tests under each track's `tests/` with its ML implementation and run them with
`uv run --no-sync pytest`. There are no functional track tests yet.
Commands for data preparation, training, evaluation and inference will be added
alongside those implementations.

The [environment verification notes](docs/packaging-verification.md) record the
scope of the setup checks. Build outputs go to ignored `dist/`.
