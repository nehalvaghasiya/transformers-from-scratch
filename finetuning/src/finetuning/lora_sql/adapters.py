"""Custom LoRA with frozen-base identity checks and portable adapter-only weights."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from finetuning.tinysql.model import TinySQLLM
from finetuning.tinysql.utils import file_hash


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    targets: tuple[str, ...] = ("qkv", "proj", "ff.0", "ff.2")

    def __post_init__(self) -> None:
        if self.rank < 1 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("Rank and alpha must be positive")

        if not 0 <= self.dropout < 1 or not self.targets:
            raise ValueError("Invalid adapter dropout or targets")


class LoRALinear(nn.Module):
    def __init__(
        self, original: nn.Linear, rank: int, alpha: float, dropout: float
    ) -> None:
        super().__init__()
        LoRAConfig(rank, alpha, dropout)
        self.original = original

        for parameter in original.parameters():
            parameter.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_a = nn.Parameter(
            original.weight.new_empty((rank, original.in_features))
        )
        self.lora_b = nn.Parameter(
            original.weight.new_zeros((original.out_features, rank))
        )
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        result: Tensor = (
            self.original(x)
            + F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b)
            * self.scaling
        )

        return result

    def merged_weight(self) -> Tensor:
        return (
            self.original.weight.detach()
            + (self.lora_b @ self.lora_a).detach() * self.scaling
        )


def _parent(root: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    module = root

    for name in parts[:-1]:
        module = module.get_submodule(name)

    return module, parts[-1]


def _targets(model: nn.Module, config: LoRAConfig) -> dict[str, nn.Linear]:
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("Model already has an adapter")

    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and any(name.endswith("." + target) for target in config.targets)
    }

    if not layers or any(
        not any(name.endswith("." + target) for name in layers)
        for target in config.targets
    ):
        raise ValueError("No matching linear layers for one or more targets")

    return layers


def attach_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    layers = _targets(model, config)

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for name, original in layers.items():
        parent, leaf = _parent(model, name)
        setattr(
            parent,
            leaf,
            LoRALinear(original, config.rank, config.alpha, config.dropout),
        )

    return list(layers)


def adapter_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def adapter_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name.endswith((".lora_a", ".lora_b"))
    }


def base_fingerprint(model: TinySQLLM) -> str:
    """Hash architecture and every frozen tensor, independent of serialization paths."""
    digest = hashlib.sha256(json.dumps(asdict(model.config), sort_keys=True).encode())
    tensors = {
        name.replace(".original.", "."): value
        for name, value in model.state_dict().items()
        if ".lora_" not in name
    }

    for name, tensor in sorted(tensors.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())

    return digest.hexdigest()


def save_adapter(
    model: TinySQLLM,
    path: str | Path,
    config: LoRAConfig,
    base_ref: str | Path = "base_small/model.pt",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": 2,
            "state_dict": adapter_state(model),
            "config": asdict(config),
            "base_ref": str(base_ref),
            "base_fingerprint": base_fingerprint(model),
            "tokenizer_sha256": model.tokenizer_sha256,
        },
        path,
    )


def load_adapter(model: TinySQLLM, path: str | Path) -> LoRAConfig:
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if checkpoint.get("format") == 2:
        identity = checkpoint
    elif "format" not in checkpoint:
        manifest = path.with_name("compatibility.json")

        if not manifest.exists():
            raise ValueError(
                "Legacy adapter requires an explicit compatibility manifest"
            )

        identity = json.loads(manifest.read_text())

        if identity.get("format") != 1 or identity.get("adapter_sha256") != file_hash(
            path
        ):
            raise ValueError("Adapter compatibility manifest checksum mismatch")
    else:
        raise ValueError("Unsupported adapter format")

    if identity.get("base_fingerprint") != base_fingerprint(model):
        raise ValueError("Adapter base fingerprint mismatch")

    if identity.get("tokenizer_sha256") != model.tokenizer_sha256:
        raise ValueError("Adapter tokenizer checksum mismatch")

    config = LoRAConfig(**checkpoint["config"])
    layers = _targets(model, config)
    shapes = {}

    for name, layer in layers.items():
        shapes[name + ".lora_a"] = (config.rank, layer.in_features)
        shapes[name + ".lora_b"] = (layer.out_features, config.rank)

    state = checkpoint["state_dict"]

    if set(shapes) != set(state) or any(
        tuple(state[k].shape) != s for k, s in shapes.items()
    ):
        raise ValueError("LoRA tensor names or shapes do not match targets")

    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Adapter contains non-finite weights")

    attach_lora(model, config)
    parameters = dict(model.named_parameters())

    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))

    return config


def merge_lora(model: TinySQLLM) -> TinySQLLM:
    """Replace low-rank wrappers with equivalent frozen linear layers."""
    layers = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }

    for name, old in layers.items():
        parent, leaf = _parent(model, name)
        new = nn.Linear(
            old.original.in_features,
            old.original.out_features,
            bias=old.original.bias is not None,
            device=old.original.weight.device,
            dtype=old.original.weight.dtype,
        )

        with torch.no_grad():
            new.weight.copy_(old.merged_weight())

            if new.bias is not None:
                new.bias.copy_(old.original.bias)

        new.requires_grad_(False)
        setattr(parent, leaf, new)

    return model
