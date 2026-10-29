"""Verify the isolated CPU ML stack with a tiny local model and one LoRA update."""

import argparse
import importlib
import json
import math
import sys
from importlib.metadata import distributions, version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["pretraining", "finetuning", "both"], default="both")
    parser.add_argument("--demo", action="store_true", help="Also verify Gradio import.")
    parser.add_argument("--output", type=Path, help="Write the JSON evidence to this path.")
    args = parser.parse_args()

    if sys.prefix == sys.base_prefix or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run this smoke check inside a clean Python 3.12 virtual environment.")
    names = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "datasets": "datasets",
        "accelerate": "accelerate",
        "tokenizers": "tokenizers",
        "huggingface_hub": "huggingface-hub",
        "numpy": "numpy",
        "safetensors": "safetensors",
    }
    tracks = ["pretraining", "finetuning"] if args.track == "both" else [args.track]
    names.update({track: f"small-llm-{track}" for track in tracks})
    if args.demo:
        names["gradio"] = "gradio"
    modules = {name: importlib.import_module(name) for name in names}
    environment = Path(sys.prefix).resolve()
    for name, module in modules.items():
        # Track packages may be editable; third-party dependencies must be isolated.
        if name not in {"pretraining", "finetuning"} and not Path(
            module.__file__
        ).resolve().is_relative_to(environment):
            raise RuntimeError(f"{name} was imported from outside the virtual environment.")

    torch = modules["torch"]
    if torch.__version__ != "2.8.0+cpu" or torch.version.cuda is not None:
        raise RuntimeError(f"Expected PyTorch 2.8.0 CPU, got {torch.__version__}.")
    cuda_packages = sorted(
        dist.metadata["Name"]
        for dist in distributions()
        if dist.metadata["Name"].lower().startswith("nvidia-")
        or dist.metadata["Name"].lower() == "triton"
    )
    if cuda_packages:
        raise RuntimeError(f"CPU environment contains CUDA packages: {cuda_packages}")

    # No remote model, dataset, tokenizer, or accelerator is needed for this check.
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=0.01)
    tokens = torch.tensor([[1, 3, 4, 5, 6, 7, 8, 2]])
    loss = model(input_ids=tokens, labels=tokens).loss
    if not math.isfinite(loss.item()):
        raise RuntimeError("Tiny model produced a non-finite loss.")
    loss.backward()
    if not all(parameter.grad is not None for parameter in trainable):
        raise RuntimeError("An adapter parameter received no gradient.")
    if not all(torch.isfinite(parameter.grad).all() for parameter in trainable):
        raise RuntimeError("Adapter gradients are non-finite.")
    optimizer.step()
    changed = sum(not torch.equal(old, new) for old, new in zip(before, trainable, strict=True))
    if not changed:
        raise RuntimeError("The optimizer did not update the LoRA adapter.")

    result = {
        "status": "passed",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "environment": sys.prefix,
        "package_paths": {name: modules[name].__file__ for name in tracks},
        "versions": {distribution: version(distribution) for distribution in names.values()},
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_packages": cuda_packages,
        "tiny_lora_loss": loss.item(),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "updated_parameter_tensors": changed,
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
