# P01-002 environment verification

The repository has two independent buildable packages and a non-installable uv
workspace root. Both packages currently contain only a package docstring;
training and evaluation belong to later roadmap issues.

Setup verification covers:

- CPU imports and one optimizer update on a tiny randomly initialized Llama with
  PEFT LoRA, without downloading models or datasets.
- Separate wheel builds and imports from outside the repository, with no
  dependency on the other track.
- Locked CPU/CUDA profile exports, matching PyTorch sources and hashes, and
  rejection of conflicting profiles.
- Ruff lint/format, mypy, lock consistency and installed dependency compatibility.

Results are recorded in [checks.json](../reports/packaging/checks.json),
[cpu-smoke.json](../reports/packaging/cpu-smoke.json),
[wheel-smoke.json](../reports/packaging/wheel-smoke.json) and
[profiles.json](../reports/packaging/profiles.json).

To repeat the environment checks after the [initial setup](../README.md#setup):

```bash
uv run --no-sync python scripts/smoke_cpu.py
uv run --no-sync python scripts/verify_profiles.py
uv build --all-packages
```

Track tests will accompany the data, model, training and scoring implementations.
No training experiment, CUDA execution, memory target, CI run or Space deployment
is established by these environment checks.
