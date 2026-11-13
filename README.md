# Transformers from Scratch and LoRA Fine-Tuning

### From-scratch ticket routing and TinySQL LoRA adaptation with custom PyTorch Transformers

**655,872-parameter ticket model • 32,768-parameter LoRA adapter • classical baselines • live demos**

[![Python](https://img.shields.io/badge/Python-3.12.12-blue)](#shared-cpu-setup) [![Training](https://img.shields.io/badge/training-from%20scratch-informational)](#pretraining-ticket-routing) [![Fine-tuning](https://img.shields.io/badge/fine--tuning-custom%20LoRA-teal)](#fine-tuning-tinysql-lora)

Two independent, synthetic-data experiments built with custom PyTorch Transformers.
They share a locked Python workspace, but **do not share weights, tasks, or benchmarks**.

| Experiment | Implemented work | Main limitation |
| --- | --- | --- |
| [Ticket model](pretraining/README.md) | 655,872-parameter causal Transformer trained from random initialization; three configurations × three seeds | Supervised ticket routing, rather than general language pretraining; classical baselines are stronger |
| [TinySQL LoRA](finetuning/README.md) | 32,768-parameter custom adapter over a frozen, separately trained 439,552-parameter SQL base | Historical synthetic benchmark; no improvement on challenge execution accuracy |

[Pretraining: ticket routing](#pretraining-ticket-routing) ·
[Fine-tuning: TinySQL LoRA](#fine-tuning-tinysql-lora)

## Shared CPU setup

Supported environment: Linux/WSL x86-64, Python **3.12.12**, uv **0.12.1**.
Run from the repository root:

```bash
uv python install 3.12.12
uv sync --locked --extra cpu --group pretraining --group demo
```

The historical dependency-group name `pretraining` supplies scikit-learn and matplotlib
for **both** experiments. Gradio is optional for training; include `demo` for the apps.
Both packages are installed in the root `.venv`. Keep `--no-sync` on subsequent commands.

## Pretraining: ticket routing

This track trains a Transformer **from random initialization** on supervised synthetic
support tickets. The directory name is historical: this is task-specific training,
rather than general language pretraining. It supplies no weights to the SQL experiment.

### Demo

![Ticket routing: live predictions and urgency edits](pretraining/docs/assets/demo-preview.gif)

The ticket model generates constrained intent and priority label tokens, which Python maps
to JSON. Its optional auxiliary objective trains two classifiers; inference optionally
fuses the **intent** classifier into intent-token logits. Historical `pool-dual` result
identifiers refer to that intent-head fusion, rather than two-head inference.

The [recorded walkthrough and live app](pretraining/README.md#demo-and-recorded-walkthrough)
show character-by-character ticket entry, predictions, an urgency edit, a real failure,
learning curves and reproduction commands.

### What the ticket results establish

| Ticket system | Test joint accuracy | Challenge joint accuracy |
| --- | ---: | ---: |
| Validation-selected demo: base, seed 42 | 91.99% | 55.56% |
| Pooled training + intent-head fusion, three-seed mean ± sample SD | 96.51% ± 2.98 | 73.15% ± 6.14 |
| Word TF-IDF | 99.51% | 78.89% |
| Character TF-IDF | 99.18% | 93.89% |

The demo was selected using validation, independently of the larger pooled variant's test
score. [Full ticket evidence](pretraining/reports/ticket-v2/benchmark.json) includes individual
seeds, uncertainty and the selection record. These scores measure synthetic template behavior.

The Transformer learns the ticket task, while the strongest classical baselines remain
more accurate on this benchmark.

### Train and launch the ticket demo

From the repository root, after the shared setup:

```bash
uv run --no-sync pretraining data prepare --config pretraining/configs/ticket-base.toml
uv run --no-sync pretraining train --config pretraining/configs/ticket-base.toml --run-id my-ticket-run
uv run --no-sync pretraining demo --model pretraining/runs/my-ticket-run --report pretraining/reports/ticket-v2
```

See the [ticket guide](pretraining/README.md) for the full experiment matrix and export.

## Fine-tuning: TinySQL LoRA

TinySQL learns SQL continuations conditioned on a question and fixed schema. LoRA freezes
all original weights and trains low-rank updates to attention and feed-forward projections.
Its **separately trained SQL base** was already trained on this task; adaptation adds
training-only synthetic wording. This track does not fine-tune the ticket model.

### Demo

![TinySQL: real frozen-base and LoRA SQL generation and execution](finetuning/docs/assets/demo-preview.gif)

The [SQL demo](finetuning/README.md#demo-and-recorded-walkthrough) runs the same question
through both models and executes their generated SQL against an isolated synthetic database.
The recording shows selected corrections and a real failure. These examples illustrate
behavior; the benchmark below measures accuracy across complete splits.

### What the fine-tuning results establish

| TinySQL system | Validation execution | Historical test execution | Challenge execution |
| --- | ---: | ---: | ---: |
| Frozen base | 69.38% | 73.44% | 22.50% |
| Released rank-8 LoRA | 75.94% | 88.75% | 22.50% |
| TF-IDF + slot copying | 78.44% | 64.06% | 78.12% |

LoRA improves historical test execution accuracy by **15.31 percentage points**, while
challenge execution accuracy stays at **22.50%** and challenge SQL validity
**falls from 41.25% to 25.00%**. Test/challenge were inspected during earlier development.
See the [SQL experiment report](finetuning/docs/EXPERIMENT_REPORT.md) for all baselines
and the [comparison plot](finetuning/plots/test_accuracy.png).

### Train and launch the SQL demo

From the repository root, after the shared setup:

```bash
cd finetuning
uv run --project .. --no-sync python -m finetuning.tinysql.data --out data
uv run --project .. --no-sync python -m finetuning.lora_sql.augment --out runs/augmentation/train.jsonl
uv run --project .. --no-sync python -m finetuning.tinysql.train --data-dir data --out runs/base --epochs 20 --device cpu --num-threads 2
uv run --project .. --no-sync python -m finetuning.lora_sql.train_lora \
  --base runs/base --data runs/augmentation --out runs/adapter \
  --rank 8 --alpha 16 --lr 0.0003 --batch 64 --epochs 1 --steps 90 --seed 42 --device cpu
uv run --project .. --no-sync python spaces/app.py --base runs/base --adapter runs/adapter/adapter.pt
```

These commands create a new experiment. The historical scores above require the exact
separately stored historical checkpoints; they are not promised for a newly trained model.
See the [SQL guide](finetuning/README.md) for evaluation and historical artifact restoration.

The demos run at **http://127.0.0.1:7860**; launch one at a time. Use a fresh run/output
location for repeated commands. The SQL demo executes only against an isolated synthetic DB.

## Reproduction and artifacts

Each track uses different tasks and metrics; their scores should be interpreted separately.
Both use synthetic data. Neither establishes real-world accuracy, general language
understanding, or new-schema SQL generalization.

- [Ticket guide](pretraining/README.md): full nine-run matrix, recovery, inference and export.
- [SQL guide](finetuning/README.md): local training, external-artifact replay, notebook and export.
- [Reproduction summary](docs/packaging-verification.md): recorded replay results and verification commands.

Model weights, adapters, tokenizers, training histories, generated datasets and exports
stay outside Git for both experiments. Existing local files remain available, but fresh
clones regenerate data and train their own models or restore separately managed artifacts.
The generators, dataset documentation, compact checksum/configuration manifests, reference
predictions and plots remain versioned. See `pretraining/reports/`, `finetuning/reports/`
and `finetuning/experiments/` for the recorded evidence.

New runs belong under ignored `runs/`. The artifact directories `finetuning/adapters/`,
`finetuning/base_small/`, `finetuning/data/` and `finetuning/data_lora/` are also ignored.
The root `reports/` path, raw logs, caches and personal editor settings stay local; the
shared VS Code workspace and concise reproduction summary remain versioned.

The Python code uses custom models and custom LoRA. Transformers, PEFT, Datasets and
Accelerate are not required. Gradio may bring its own transitive dependencies.

## Software checks

VS Code automatically formats Python files and notebook cells on save when using
[this workspace](.vscode/transformers-from-scratch.code-workspace) with the Ruff extension.
It normalizes spacing, four-space indentation, quotes and line wrapping, and runs
import sorting and safe lint fixes on explicit saves. The editor uses the locked
`.venv/bin/ruff` and the shared `pyproject.toml` configuration. Run the setup above
first. These settings are also applied to this checkout's local VS Code configuration.

Use Ruff for import sorting and Black-style formatting:

```bash
uv run --no-sync ruff check . --select I --fix
uv run --no-sync ruff format .
```

Verify formatting and run the other checks:

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy
uv run --no-sync pytest
uv run --no-sync python scripts/smoke_cpu.py
uv lock --check
uv build --all-packages
uv run --no-sync python scripts/verify_profiles.py
```

[CPU CI](.github/workflows/cpu.yml) runs lightweight checks rather than the full training
matrix. `cu126` and `cu128` profiles remain optional, mutually exclusive Torch sources;
resolution checks do not verify GPU execution or memory. CPU is the verified runtime.
Public hosting is not claimed. No specific VRAM capacity is promised.
