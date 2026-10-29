# Pretraining from scratch

Build and train a miniature GPT from random initialization, making the tokenizer,
model, optimization and evaluation code easy to inspect.

An existing scratch-trained ticket Transformer, saved checkpoint and comparison
experiments are in `../pretrained/`. The [integration plan](integration-plan.md)
describes how to bring them into this package with verified results, reproduction
commands, documentation and a demo video. The WikiText experiment below remains
separate planned work.

Planned experiment: WikiText-2, a training-only 4,096-token byte-level BPE
tokenizer, and a four-layer GPT with width 192, six attention heads and context
128 (approximately 2.59M parameters). Compare three training seeds, a two-layer
depth ablation, random initialization and simple token baselines. Report
token-weighted held-out NLL/perplexity, learning curves, fixed-prompt generations,
throughput and measured memory.

**Current status:** importable package and shared environment setup. The existing
ticket project has not been migrated; WikiText data, training and evaluation are
not implemented in this package yet.
See the [root README](../README.md) for setup and checks.

Implementation belongs in `src/pretraining/`; add tests in `tests/` with the model
and data code. Keep configs and compact reports here, downloaded data in ignored
`data/`, and checkpoints in ignored `runs/`. This package is independent of
`finetuning`.

| Roadmap issues | Responsibility |
| --- | --- |
| P01-010–014 | WikiText preparation, document splits, training-only tokenizer, packed tokens |
| P01-015–022 | Scratch GPT, correctness tests, optimization, generation, checkpoint recovery |
| P01-023–031 | Overfit diagnostic, pilot, three main seeds, depth ablation, held-out scoring and report |
| P01-048–053 | Scratch evaluation, uncertainty, performance and reporting |
| P01-054–060 | Scratch artifact, model card, release verification and demonstration |

[plan.md](../plan.md) defines the detailed configurations, run IDs and acceptance
criteria. Add its `pretraining` commands as their workflows are implemented.
