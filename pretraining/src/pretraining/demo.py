"""Local demonstration using actual evaluated weights and explicit decoding modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inference import load_artifact, predict


def build_demo(artifact: Path, report: Path | None = None) -> Any:
    import gradio as gr

    model, tokenizer, head, legacy = load_artifact(artifact)
    metadata = json.loads((artifact / "run_info.json").read_text())
    modes = ["Generative"] + (["Intent-head fusion"] if head is not None else [])
    selected_mode = "Generative"
    if report is not None and (report / "selection.json").exists():
        selection = json.loads((report / "selection.json").read_text())
        if selection.get("mode") == "dual" and head is not None:
            selected_mode = "Intent-head fusion"

    def classify(text: str, mode: str) -> dict[str, Any]:
        result: dict[str, Any] = dict(
            predict(
                model,
                tokenizer,
                [text],
                pool_head=head if mode == "Intent-head fusion" else None,
                legacy_pooling=legacy,
            )[0]
        )
        tokens = tokenizer.encode_text(text)
        result["known_vocabulary_fraction"] = round(
            sum(token != tokenizer.unk_id for token in tokens) / max(1, len(tokens)), 3
        )
        return result

    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
    with gr.Blocks(title="Ticket Transformer · trained from scratch") as demo:
        gr.Markdown("# Ticket Transformer\n### A small causal model, trained from scratch")
        gr.Markdown(
            f"**{model.count_parameters():,} Transformer parameters** · "
            f"{model.config.n_layer} blocks · {model.config.n_head} attention heads · "
            f"{len(tokenizer.vocab)} vocabulary entries\n\n"
            "Type a short English support ticket to generate its intent and priority. "
            "This research demo uses synthetic training data and constrained label generation."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    lines=5,
                    label="Support ticket",
                    placeholder="My invoice is wrong. There is no rush; next week is fine.",
                )
                mode = gr.Radio(modes, value=selected_mode, label="Prediction method")
                button = gr.Button("Route ticket", variant="primary")
            with gr.Column():
                output = gr.JSON(label="Actual model prediction")
        button.click(classify, [text, mode], output, api_name="predict")
        gr.Examples(
            [
                ["I was charged twice for AtlasDesk. There is no rush; next week is fine."],
                [
                    "The export button in NovaBoard crashes when I click it. "
                    "Our team is blocked from an essential task today."
                ],
                [
                    "Could you add a dark mode option to ZenFlow? "
                    "This is only for future reference, not an active blocker."
                ],
            ],
            inputs=[text],
        )
        with gr.Accordion("Measured comparisons and limitations", open=True):
            if report is not None and (report / "results.md").exists():
                gr.Markdown((report / "results.md").read_text())
            gr.Markdown(
                "Six intents and three priorities; English synthetic tickets only. "
                "Unknown vocabulary, multiple intents and implicit urgency can fail. "
                "Valid labels are guaranteed by decoding, not by correct understanding. "
                "Vocabulary coverage is not a confidence score."
            )
        if report is not None:
            with gr.Accordion("Learning and evaluation figures", open=False):
                for name, label in (
                    ("comparison.png", "Three-seed comparisons"),
                    ("learning-curves.png", "Training and validation answer loss"),
                ):
                    if (report / name).exists():
                        gr.Image(str(report / name), label=label, interactive=False)
            with gr.Accordion("Reproduce this model", open=False):
                gr.Markdown(
                    "Run from the repository root with the committed lockfile:\n\n"
                    "```bash\n"
                    "uv sync --locked --extra cpu --group pretraining --group demo\n"
                    "uv run --no-sync pretraining data prepare "
                    "--config pretraining/configs/ticket-base.toml\n"
                    "uv run --no-sync pretraining train "
                    "--config pretraining/configs/ticket-base.toml --run-id ticket-v2-base-42\n"
                    "```\n\n"
                    "The source, frozen experiment contract, individual seeds and predictions "
                    "are documented in the pretraining case study."
                )
        gr.Markdown(
            f"Artifact: `{metadata.get('run_id', 'legacy-official')}` · "
            f"seed `{metadata.get('seed', 'unknown')}` · actual loaded weights"
        )
    demo._ticket_theme = theme
    return demo


def launch(artifact: Path, report: Path | None = None, port: int = 7860) -> None:
    demo = build_demo(artifact, report)
    demo.launch(server_name="127.0.0.1", server_port=port, theme=demo._ticket_theme)
