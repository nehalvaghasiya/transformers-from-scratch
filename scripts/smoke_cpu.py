"""Exercise the actual ticket model and custom SQL LoRA in the isolated CPU environment."""

import argparse
import importlib
import json
import sys
from importlib.metadata import distributions, version
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["pretraining", "finetuning", "both"], default="both")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if sys.prefix == sys.base_prefix or sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the documented Python 3.12 virtual environment")
    if torch.__version__ != "2.8.0+cpu" or torch.version.cuda is not None:
        raise RuntimeError("Expected the locked PyTorch 2.8.0 CPU profile")
    environment = Path(sys.prefix).resolve()
    if not Path(torch.__file__).resolve().is_relative_to(environment):
        raise RuntimeError("Torch was imported from outside the virtual environment")
    cuda = [
        dist.metadata["Name"]
        for dist in distributions()
        if dist.metadata["Name"].lower().startswith("nvidia-")
        or dist.metadata["Name"].lower() == "triton"
    ]
    if cuda:
        raise RuntimeError(f"Unexpected GPU dependencies: {cuda}")
    if args.demo:
        importlib.import_module("gradio")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    results = {}
    if args.track in {"pretraining", "both"}:
        from pretraining.model import ModelConfig, TinyTicketLM

        model = TinyTicketLM(ModelConfig(32, 16, 1, 2, 16, 0.0))
        before = model.token_emb.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        tokens = torch.tensor([[1, 3, 4, 5, 6]])
        targets = torch.tensor([[-100, -100, 5, 6, 2]])
        _, loss = model(tokens, targets)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        assert not torch.equal(before, model.token_emb.weight)
        results["pretraining"] = {"loss": loss.item(), "parameters": model.count_parameters()}
    if args.track in {"finetuning", "both"}:
        from finetuning.lora_sql.adapters import (
            LoRAConfig,
            adapter_parameters,
            attach_lora,
            base_fingerprint,
        )
        from finetuning.tinysql.model import ModelConfig as SQLConfig
        from finetuning.tinysql.model import TinySQLLM

        sql = TinySQLLM(SQLConfig(32, 16, 16, 1, 2))
        frozen = base_fingerprint(sql)
        attach_lora(sql, LoRAConfig(rank=2, alpha=4))
        parameters = adapter_parameters(sql)
        before_adapter = [p.detach().clone() for p in parameters]
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        tokens = torch.tensor([[1, 3, 4, 5, 6]])
        _, loss = sql(tokens, tokens)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before_adapter, parameters, strict=True))
        assert frozen == base_fingerprint(sql)
        results["finetuning"] = {
            "loss": loss.item(),
            "adapter_parameters": sum(p.numel() for p in parameters),
            "frozen_base_unchanged": True,
        }
    tracks = ["pretraining", "finetuning"] if args.track == "both" else [args.track]
    result = {
        "status": "passed",
        "python": sys.version.split()[0],
        "environment": sys.prefix,
        "torch": str(torch.__version__),
        "cuda_runtime_tested": False,
        "package_paths": {name: importlib.import_module(name).__file__ for name in tracks},
        "versions": {
            name: version(name) for name in ["torch", "numpy", *[f"small-llm-{x}" for x in tracks]]
        },
        "checks": results,
    }
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
