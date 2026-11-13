"""Local Gradio comparison using explicitly selected, separately stored model artifacts."""

import argparse
from pathlib import Path

import gradio as gr
import torch

from finetuning.lora_sql.adapters import load_adapter
from finetuning.tinysql.data import create_db, execute_sql
from finetuning.tinysql.train import load_model

ROOT = Path(__file__).resolve().parents[1]


def build_demo(base_dir: Path, adapter: Path) -> gr.Blocks:
    for path in (base_dir / "model.pt", base_dir / "tokenizer.json", adapter):
        if not path.is_file():
            raise FileNotFoundError(
                f"Local model artifact missing: {path}. "
                "Train a base and adapter or restore an external bundle; see finetuning/README.md."
            )
    torch.set_num_threads(2)
    base, tok = load_model(base_dir, device="cpu")
    adapted, _ = load_model(base_dir, device="cpu")
    load_adapter(adapted, adapter)
    adapted.eval()

    def run(question):
        if not question.strip():
            return "Enter a question", "", [], []
        if len(question) > 280:
            return "Input too long", "", [], []
        ids = tok.format_input(question)
        if len(ids) >= base.config.context:
            return "Input exceeds context", "", [], []
        original = tok.decode(base.generate(ids, tok.eos_id, max_new_tokens=85))
        final = tok.decode(adapted.generate(ids, tok.eos_id, max_new_tokens=85))

        def safe_run(sql):
            db = create_db(2026)
            try:
                return execute_sql(sql, db)[:20]
            except Exception as error:
                return [f"Cannot execute: {str(error)[:160]}"]
            finally:
                db.close()

        return original, final, safe_run(original), safe_run(final)

    with gr.Blocks(title="TinySQL LoRA: Base vs Adapted") as demo:
        gr.Markdown(
            "# TinySQL LoRA — frozen base vs adapter\n"
            "Both SQL strings are generated autoregressively; the small model has serious "
            "limitations. Demo uses a synthetic, isolated sample database."
        )
        question = gr.Textbox(label="Question", value="how many completed orders are there")
        button = gr.Button("Generate and execute")
        with gr.Row():
            base_sql = gr.Textbox(label="Frozen base SQL", lines=3, interactive=False)
            lora_sql = gr.Textbox(label="LoRA SQL", lines=3, interactive=False)
        with gr.Row():
            base_rows = gr.JSON(label="Frozen base results", height=160, elem_id="base-results")
            lora_rows = gr.JSON(label="LoRA results", height=160, elem_id="lora-results")
        button.click(run, inputs=question, outputs=[base_sql, lora_sql, base_rows, lora_rows])
        gr.Markdown(
            "Results show at most 20 rows from a fixed sample database. "
            "SQL that executes can still answer the question incorrectly. "
            "Benchmark accuracy is measured separately on two populated databases."
        )
        gr.Examples(
            [
                "retrieve names for the customers in oh",
                "i need number of completed orders for ny customers",
                "i want 5 expensive orders in pending status of at least 40",
                "how many orders are completed",
                "i am interested in names of people from wa sort by name",
            ],
            inputs=question,
        )
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "base_small")
    parser.add_argument("--adapter", type=Path, default=ROOT / "adapters/aug_r8_fast/adapter.pt")
    args = parser.parse_args()
    build_demo(args.base, args.adapter).launch()
