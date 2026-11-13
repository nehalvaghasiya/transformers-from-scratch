"""Compare external SQL artifacts and a ticket replay with historical results."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from finetuning.lora_sql.augment import create
from finetuning.lora_sql.evaluate_lora import evaluate
from finetuning.lora_sql.plots import plot
from finetuning.tinysql.data import build_dataset
from finetuning.tinysql.tokenizer import WordSQLTokenizer
from finetuning.tinysql.train import load_model
from finetuning.tinysql.utils import file_hash, provenance, read_jsonl
from pretraining.train import source_hashes

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ticket-report", type=Path, required=True)
    parser.add_argument("--reuse-sql", action="store_true", help="Reuse matching SQL replay files")
    args = parser.parse_args()
    required = [
        ROOT / "finetuning/base_small/model.pt",
        ROOT / "finetuning/base_small/tokenizer.json",
        ROOT / "finetuning/adapters/aug_r8_fast/adapter.pt",
        ROOT / "finetuning/adapters/aug_r8_fast/compatibility.json",
    ]
    required += [
        ROOT / f"finetuning/data/{split}.jsonl" for split in ["train", "val", "test", "challenge"]
    ]
    required.append(ROOT / "finetuning/data_lora/train.jsonl")
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        parser.error(
            "Historical replay needs separately stored weights and generated data: "
            + ", ".join(missing)
            + ". See finetuning/README.md."
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.reuse_sql)
    sql_root = ROOT / "finetuning"
    result = {
        "lock_sha256": file_hash(ROOT / "uv.lock"),
        "verification_script_sha256": file_hash(__file__),
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "data": {},
        "sql": {},
        "ticket": {},
    }
    with tempfile.TemporaryDirectory(prefix="tinysql-regeneration-") as directory:
        data = Path(directory)
        build_dataset(data, 6400, 320, 320, 160, seed=42)
        result["data"]["original_bytes_match"] = {
            split: file_hash(data / f"{split}.jsonl") == file_hash(sql_root / f"data/{split}.jsonl")
            for split in ["train", "val", "test", "challenge"]
        }
        create(sql_root / "data/train.jsonl", data / "aug.jsonl")
        result["data"]["augmentation_bytes_match"] = file_hash(data / "aug.jsonl") == file_hash(
            sql_root / "data_lora/train.jsonl"
        )
    _, tokenizer = load_model(sql_root / "base_small")
    result["data"]["tokenizer_matches_training"] = (
        tokenizer.vocab == WordSQLTokenizer.fit(read_jsonl(sql_root / "data/train.jsonl")).vocab
    )
    train = read_jsonl(sql_root / "data/train.jsonl")
    augmented = read_jsonl(sql_root / "data_lora/train.jsonl")
    result["data"]["training_counts"] = {
        "original_rows": len(train),
        "original_unique_questions": len({r["question"] for r in train}),
        "augmented_rows": len(augmented),
        "augmented_unique_questions": len({r["question"] for r in augmented}),
    }
    result["data"]["exact_question_overlap"] = {
        split: len(
            {r["question"] for r in augmented}
            & {r["question"] for r in read_jsonl(sql_root / f"data/{split}.jsonl")}
        )
        for split in ["val", "test", "challenge"]
    }
    for split in ["val", "test", "challenge"]:
        if args.reuse_sql:
            reproduced = json.loads((output / f"{split}.json").read_text())
            identity = provenance(sql_root / "data")
            if any(reproduced[k] != identity[k] for k in ["source_hashes", "data_hashes"]):
                raise ValueError("SQL replay source or data changed; use a fresh output")
            for key, path in [
                ("adapter_sha256", sql_root / "adapters/aug_r8_fast/adapter.pt"),
                ("base_model_sha256", sql_root / "base_small/model.pt"),
                ("tokenizer_sha256", sql_root / "base_small/tokenizer.json"),
            ]:
                if reproduced[key] != file_hash(path):
                    raise ValueError("SQL replay artifact changed")
        else:
            reproduced = evaluate(
                adapter=sql_root / "adapters/aug_r8_fast/adapter.pt",
                base=sql_root / "base_small",
                data=sql_root / "data",
                split=split,
                count=0,
                output=output / f"{split}.json",
            )
        historical = json.loads((sql_root / f"reports/{split}_final.json").read_text())
        before = read_jsonl(sql_root / f"reports/{split}_final_predictions.jsonl")
        after = read_jsonl(output / f"{split}_predictions.jsonl")
        aligned = [r["id"] for r in before] == [r["id"] for r in after]
        comparison = {}
        for name, metrics in reproduced["model_scores"].items():
            comparison[name] = {
                "historical_execution_accuracy": historical["model_scores"][name][
                    "execution_accuracy"
                ],
                "reproduced_execution_accuracy": metrics["execution_accuracy"],
                "sql_strings_match": aligned
                and all(
                    a[name]["sql"] == b[name]["sql"] for a, b in zip(before, after, strict=True)
                ),
                "all_metrics_match": metrics == historical["model_scores"][name],
            }
        result["sql"][split] = {
            "ids_match": aligned,
            "models": comparison,
            "bootstrap_interval_matches": reproduced["paired_bootstrap_95pct_ci_pp"]
            == historical["paired_bootstrap_95pct_ci_pp"],
        }
    plot(
        run_dirs=(sql_root / "adapters/aug_r8_fast",),
        reports=tuple(output / f"{s}.json" for s in ["val", "test", "challenge"]),
        out=output / "plots",
    )
    reference_root = ROOT / "pretraining/reports/ticket-v2"
    old = json.loads((reference_root / "benchmark.json").read_text())
    new = json.loads((args.ticket_report / "benchmark.json").read_text())
    metrics = [
        "joint_accuracy",
        "intent_accuracy",
        "intent_macro_f1",
        "priority_accuracy",
        "priority_macro_f1",
    ]

    def index(row):
        return row["split"], row["system"], row["seed"]

    expected = {index(r): r for r in old["results"]}
    actual = {index(r): r for r in new["results"]}
    differences = []
    if expected.keys() != actual.keys():
        differences.append({"problem": "Missing or extra systems/splits/seeds"})
    for key in expected.keys() & actual.keys():
        for metric in metrics:
            if abs(expected[key][metric] - actual[key][metric]) > 1e-12:
                differences.append(
                    {
                        "system_split_seed": key,
                        "metric": metric,
                        "historical": expected[key][metric],
                        "reproduced": actual[key][metric],
                    }
                )
    prediction_matches = {}
    for path in sorted(reference_root.glob("*.csv")):
        if path.name == "summary.csv":
            continue
        destination = args.ticket_report / path.name
        # Rows are compared semantically to avoid differences in newline conventions.
        prediction_matches[path.name] = destination.exists() and (
            list(csv.DictReader(path.open())) == list(csv.DictReader(destination.open()))
        )
    selection_keys = ["variant", "mode", "seed", "comparison_baseline", "test_used_for_selection"]
    result["ticket"] = {
        "run_count": len(new["runs"]),
        "metric_differences": differences,
        "prediction_files_match": prediction_matches,
        "selection_matches": all(
            old["selection"][k] == new["selection"][k] for k in selection_keys
        ),
        "source_hashes_match_current": {
            run: info["source_hashes"] == source_hashes() for run, info in new["runs"].items()
        },
        "paired_intervals_match": old["paired_intervals"] == new["paired_intervals"],
    }
    result["status"] = (
        "passed"
        if (
            all(result["data"]["original_bytes_match"].values())
            and result["data"]["augmentation_bytes_match"]
            and result["data"]["tokenizer_matches_training"]
            and not any(result["data"]["exact_question_overlap"].values())
            and all(
                s["ids_match"]
                and s["bootstrap_interval_matches"]
                and all(
                    m["all_metrics_match"] and m["sql_strings_match"] for m in s["models"].values()
                )
                for s in result["sql"].values()
            )
            and not differences
            and all(prediction_matches.values())
            and result["ticket"]["selection_matches"]
            and result["ticket"]["paired_intervals_match"]
            and all(result["ticket"]["source_hashes_match_current"].values())
        )
        else "differences_found"
    )
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
