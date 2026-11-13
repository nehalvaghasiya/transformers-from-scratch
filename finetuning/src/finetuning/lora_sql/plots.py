"""Generate training/evaluation visualizations using measured files only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(
    run_dirs: tuple[str | Path, ...] = (
        "adapters/aug_r4",
        "adapters/aug_r8_fast",
        "adapters/aug_r16",
        "adapters/aug_r8_long",
    ),
    reports: tuple[str | Path, ...] = (
        "reports/val_final.json",
        "reports/test_final.json",
        "reports/challenge_final.json",
    ),
    out: str | Path = "runs/plots",
) -> list[str]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_ = False

    for path in run_dirs:
        h = Path(path) / "history.json"
        meta = Path(path) / "metadata.json"

        if h.exists():
            hist = json.loads(h.read_text())
            m = json.loads(meta.read_text())
            any_ = True
            ax.plot(
                [0] + [z["step"] for z in hist],
                [m["base_val_sql_ce"]] + [z["val_sql_ce"] for z in hist],
                marker="o",
                label=Path(path).name,
            )

    if any_:
        ax.set(
            xlabel="Optimization steps",
            ylabel="Held-out SQL token cross-entropy",
            title="LoRA validation learning curves",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "validation_loss.png", dpi=160)

    plt.close(fig)

    for file in reports:
        p = Path(file)

        if not p.exists():
            continue

        d = json.loads(p.read_text())
        scores = d["model_scores"]
        names = list(scores)
        values = [100 * scores[k]["execution_accuracy"] for k in names]

        fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.5), 4.5))
        bars = ax.bar(range(len(names)), values)
        ax.set_xticks(
            range(len(names)), [x.replace("_", "\n") for x in names], fontsize=8
        )
        ax.set(
            ylim=(0, 100),
            ylabel="Execution accuracy (%)",
            title=f"{d['split']} | n={d['sample_count']} | 2 SQLite DBs",
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=9)

        fig.tight_layout()
        fig.savefig(out / (d["split"] + "_accuracy.png"), dpi=160)
        plt.close(fig)

        if "frozen_base" in scores and "lora_raw" in scores:
            fam = sorted(scores["frozen_base"]["by_family"])
            x = np.arange(len(fam))
            b = [
                100 * scores["frozen_base"]["by_family"][f]["execution_accuracy"]
                for f in fam
            ]
            adapted_values = [
                100 * scores["lora_raw"]["by_family"][f]["execution_accuracy"]
                for f in fam
            ]

            fig, ax = plt.subplots(figsize=(11, 4))
            ax.bar(x - 0.2, b, 0.4, label="Frozen base")
            ax.bar(x + 0.2, adapted_values, 0.4, label="LoRA")
            ax.set_xticks(x, fam, rotation=35, ha="right")
            ax.set(
                ylabel="Accuracy (%)",
                ylim=(0, 105),
                title=f"{d['split']}: accuracy by query family",
            )
            ax.legend()

            fig.tight_layout()
            fig.savefig(out / (d["split"] + "_by_family.png"), dpi=160)
            plt.close(fig)

    return sorted(str(p) for p in out.glob("*.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append")
    parser.add_argument("--report", action="append")
    parser.add_argument("--out", default="runs/plots")
    args = parser.parse_args()
    options = {}

    if args.run_dir is not None:
        options["run_dirs"] = tuple(args.run_dir)

    if args.report is not None:
        options["reports"] = tuple(args.report)

    print("\n".join(plot(out=args.out, **options)))
