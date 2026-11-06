# Environment and packaging verification

The root is a non-installable uv workspace with two independent packages.
`pretraining` now implements ticket data generation, training/recovery, prediction,
evaluation, a frozen benchmark, artifact verification and a local demo.
`finetuning` remains an importable package with its implementation planned.

Verification covers:

- Ruff lint/format, strict mypy and 12 meaningful CPU tests.
- Separate wheel builds and independent installation/import outside the checkout.
- A full selected-model training replay from the installed pretraining wheel,
  compared with the reference model tensors and held-out predictions.
- Locked CPU/CUDA profile exports, correct Torch sources and conflict rejection.
- The tiny CPU LoRA dependency-compatibility smoke and installed dependency checks.
- Notebook smoke/prediction/validation cells and the actual Gradio browser flow.

Compact command receipts are in [checks.json](../reports/packaging/checks.json);
independent-wheel results are in [wheel-smoke.json](../reports/packaging/wheel-smoke.json).
[CPU dependency smoke](../reports/packaging/cpu-smoke.json) and
[profile resolution](../reports/packaging/profiles.json) remain separate from the
[actual ticket experiments](../pretraining/reports/ticket-v2/benchmark.json).

The [CPU workflow](../.github/workflows/cpu.yml) is added with pinned action revisions.
Local checks and shell syntax were verified; no remote Actions run is claimed.
CUDA execution, a physical 3 GiB GPU target and public hosting have not been measured
or deployed by these CPU checks. Follow the [root README](../README.md) and
[pretraining guide](../pretraining/README.md) to repeat the appropriate checks.
