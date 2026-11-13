"""Small causal Transformer; state names remain compatible with the imported model."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 112
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if (
            min(
                self.vocab_size,
                self.max_seq_len,
                self.n_layer,
                self.n_head,
                self.n_embd,
            )
            < 1
        ):
            raise ValueError("Model dimensions must be positive")

        if self.n_embd % self.n_head or not 0 <= self.dropout < 1:
            raise ValueError("Width must divide into heads; dropout must be in [0, 1)")


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.dim = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.drop = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q = q.view(batch, length, self.n_head, self.dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, self.dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, self.dim).transpose(1, 2)
        attention = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.drop if self.training else 0.0, is_causal=True
        )
        projected = self.proj(
            attention.transpose(1, 2).contiguous().view(batch, length, width)
        )

        return projected  # type: ignore[no-any-return]


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.ff = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))

        return x + self.ff(self.norm2(x))  # type: ignore[no-any-return]


class TinyTicketLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, idx: Tensor) -> Tensor:
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.config.max_seq_len:
            raise ValueError(
                f"Expected nonempty sequences of at most {self.config.max_seq_len} tokens"
            )

        positions = torch.arange(idx.shape[1], device=idx.device)
        x = self.drop(self.token_emb(idx) + self.pos_emb(positions)[None, :, :])

        for block in self.blocks:
            x = block(x)

        return self.norm(x)  # type: ignore[no-any-return]

    def forward(
        self, idx: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        logits = self.lm_head(self.encode(idx))
        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def save_model(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), folder / "model.pt")
        (folder / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n"
        )

    @classmethod
    def load_model(cls, folder: str | Path, device: str = "cpu") -> TinyTicketLM:
        folder = Path(folder)
        config = ModelConfig(**json.loads((folder / "config.json").read_text()))

        # Loading an artifact should not alter a future training run's random stream.
        with torch.random.fork_rng(devices=[]):
            model = cls(config)

        model.load_state_dict(
            torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
        )

        return model.to(device).eval()


def make_pool_head(width: int) -> nn.ModuleDict:
    from .data import INTENTS, PRIORITIES

    return nn.ModuleDict(
        {
            "intent": nn.Linear(width, len(INTENTS)),
            "priority": nn.Linear(width, len(PRIORITIES)),
        }
    )


def pool_prompt(hidden: Tensor, tokens: Tensor, pad_id: int, sep_id: int) -> Tensor:
    """Pool BOS and prompt words, excluding SEP, answer tokens and padding."""
    before_sep = (tokens == sep_id).cumsum(dim=1) == 0
    mask = before_sep & (tokens != pad_id)

    return (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
