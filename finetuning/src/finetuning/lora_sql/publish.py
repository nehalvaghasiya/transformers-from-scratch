"""Prepare a portable local bundle; no upload or publication side effects."""

from __future__ import annotations

import argparse
import json
import shutil
from importlib.metadata import distribution
from pathlib import Path

from finetuning.tinysql.train import load_model
from finetuning.tinysql.utils import file_hash

from .adapters import load_adapter


def model_card(
    adapter_dir: str | Path = "adapters/aug_r8_fast",
    out: str | Path = "runs/adapter-card.md",
    reports: tuple[str | Path, ...] = (),
) -> Path:
    adapter_dir = Path(adapter_dir)
    meta = json.loads((adapter_dir / "metadata.json").read_text())
    lines = [
        "# TinySQL custom LoRA adapter",
        "",
        "Fixed-schema synthetic SQL experiment. Requires the exact matching base and tokenizer.",
        "This base was trained separately from the ticket model.",
        "",
        f"Adapter parameters: {meta['trainable_parameters']:,}; rank {meta['config']['rank']}; "
        f"alpha {meta['config']['alpha']}. Seed: {meta['seed']}.",
        "Checkpoint selection within this run uses validation SQL-token cross-entropy.",
        (
            f"Validation CE: {meta['base_val_sql_ce']:.5f} base, "
            f"{meta['best_val_sql_ce']:.5f} adapter."
        ),
        meta.get("release_selection", "No comparison across adapter configurations is asserted."),
        "",
        "## Evaluation",
        "",
    ]
    if not reports:
        lines.append("No execution-accuracy report was supplied for this export.")
    for report_path in reports:
        data = json.loads(Path(report_path).read_text())
        if data.get("adapter_sha256") != file_hash(adapter_dir / "adapter.pt"):
            raise ValueError("Evaluation report does not identify this adapter")
        scores = data["model_scores"]
        lines.append(
            f"{data['split']}, n={data['sample_count']}: base "
            f"{scores['frozen_base']['execution_accuracy']:.2%}, adapter "
            f"{scores['lora_raw']['execution_accuracy']:.2%}."
        )
    lines += [
        "",
        "Synthetic data with repeated SQL skeletons; historical test and challenge splits",
        "were inspected during prior development. Results are exploratory. No real-world",
        (
            "accuracy or new-schema generalization is established. Training row "
            "counts include duplicates."
        ),
        "Classical retrieval baselines can be stronger, particularly on challenge wording.",
        "",
        "## Loading",
        "",
        "Install the documented CPU environment. The export includes the `finetuning/` source,",
        "the frozen `base/`, tokenizer, adapter and compatibility metadata. From the bundle:",
        "",
        "```bash",
        'python predict.py "how many orders are completed"',
        "```",
        "",
        "Custom PyTorch architecture; this bundle is not an AutoModel/PEFT checkpoint.",
        "Generated SQL can be invalid or incorrect. Execute only on the included sample database.",
        "GPU and hosted deployment have not been verified.",
        "",
    ]
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines))
    return target


def prepare_hf(
    adapter: str | Path = "adapters/aug_r8_fast",
    out: str | Path = "runs/exports/tinysql",
    base: str | Path = "base_small",
    reports: tuple[str | Path, ...] = (),
) -> Path:
    """Keep the historical function name for local callers; create a fresh complete bundle."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    model, _ = load_model(base)
    load_adapter(model, Path(adapter) / "adapter.pt")
    out.mkdir(parents=True)
    shutil.copytree(base, out / "base", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("adapter.pt", "metadata.json", "compatibility.json"):
        source = Path(adapter) / name
        if source.exists():
            shutil.copy2(source, out / name)
    source_root = Path(__file__).parents[1]
    shutil.copytree(
        source_root, out / "finetuning", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    candidates = [Path(__file__).parents[3] / "LICENSE"]
    dist = distribution("small-llm-finetuning")
    candidates.extend(
        Path(str(dist.locate_file(f)))
        for f in dist.files or []
        if str(f).endswith("/licenses/LICENSE")
    )
    license_file = next((p for p in candidates if p.is_file()), None)
    if license_file is None:
        raise FileNotFoundError("MIT license missing from installed package")
    shutil.copy2(license_file, out / "LICENSE")
    model_card(adapter, out / "README.md", reports)
    (out / "predict.py").write_text(
        "from pathlib import Path\nimport sys\n"
        "from finetuning.tinysql.train import load_model\n"
        "from finetuning.lora_sql.adapters import load_adapter\n"
        "root = Path(__file__).resolve().parent\n"
        'model, tok = load_model(root / "base")\n'
        'load_adapter(model, root / "adapter.pt")\n'
        'question = sys.argv[1] if len(sys.argv)>1 else "how many orders are completed"\n'
        "print(tok.decode(model.generate(tok.format_input(question), tok.eos_id)))\n"
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "format": 1,
                "files": {
                    str(p.relative_to(out)): file_hash(p)
                    for p in sorted(out.rglob("*"))
                    if p.is_file()
                },
            },
            indent=2,
        )
        + "\n"
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="adapters/aug_r8_fast")
    parser.add_argument("--out", default="runs/exports/tinysql")
    parser.add_argument("--base", default="base_small")
    parser.add_argument("--report", action="append", default=[])
    args = parser.parse_args()
    print(prepare_hf(args.adapter, args.out, args.base, tuple(args.report)))
