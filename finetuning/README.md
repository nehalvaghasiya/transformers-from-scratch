# Fine-tuning a pretrained model

Adapt SmolLM2-135M-Instruct with LoRA to extract job title, skills, minimum
experience, education and work mode from job descriptions.

The planned comparison includes deterministic rules, zero-shot and one-shot base
model prompts, three main LoRA seeds, a rank ablation and a synthetic-only data
ablation. Use reviewed real examples and deterministic synthetic data. Report
native JSON/schema validity, per-field and pooled F1, exact-record accuracy,
robustness, error analysis and measured inference/training resources.

**Current status:** importable package and shared environment setup. Extraction
data, baselines, LoRA training, scoring and the Gradio demo are not implemented
yet. See the [root README](../README.md) for setup and checks.

Implementation belongs in `src/finetuning/`; add tests in `tests/` with the data,
training and scoring code. Keep configs and compact reports here, downloaded
data in ignored `data/`, and checkpoints in ignored `runs/`. This package is
independent of `pretraining`.

| Roadmap issues | Responsibility |
| --- | --- |
| P01-032–047 | Reviewed extraction data, rules and prompting baselines, LoRA pipeline, runs, selection and export |
| P01-048–053 | Extraction evaluation, uncertainty, robustness, failures and performance |
| P01-054–060 | Adapter/merged artifacts, extraction dataset, cards, Gradio Space and release verification |

[plan.md](../plan.md) defines the detailed configurations, run IDs and acceptance
criteria. Add its `finetuning` commands as their workflows are implemented.
