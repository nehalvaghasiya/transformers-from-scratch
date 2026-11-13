"""Causal SQL decoder; state names remain compatible with the supplied checkpoint."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    context: int = 160
    dim: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.context, self.dim, self.layers, self.heads) < 1:
            raise ValueError("Model dimensions must be positive")

        if self.dim % self.heads or not 0 <= self.dropout < 1:
            raise ValueError("Invalid attention dimensions or dropout")


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.dropout = cfg.dropout
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).split(width, dim=-1)
        q = q.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        k = k.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        v = v.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )

        return self.resid_drop(  # type: ignore[no-any-return]
            self.proj(y.transpose(1, 2).contiguous().view(batch, length, width))
        )


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(cfg.dim)
        self.attn = Attention(cfg)
        self.n2 = nn.LayerNorm(cfg.dim)
        self.ff = nn.Sequential(
            nn.Linear(cfg.dim, cfg.dim * 4),
            nn.GELU(),
            nn.Linear(cfg.dim * 4, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x))

        return x + self.ff(self.n2(x))  # type: ignore[no-any-return]


class TinySQLLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.config = cfg
        self.tokenizer_sha256: str | None = None

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.context, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, 0, 0.02)

        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(
        self, ids: Tensor, labels: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.context:
            raise ValueError(
                f"Expected nonempty sequences of at most {self.config.context} tokens"
            )

        x = self.tok_emb(ids) + self.pos_emb(
            torch.arange(ids.shape[1], device=ids.device)
        )

        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.ln(x))
        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[list[int]],
        eos_id: int,
        max_new_tokens: int = 85,
        batch: int = 16,
    ) -> list[list[int]]:
        """Greedy decoding at original learned positions, shared by every inference path."""
        if max_new_tokens < 0 or batch < 1:
            raise ValueError("Token budget must be nonnegative and batch size positive")

        if any(not p or len(p) >= self.config.context for p in prompts):
            raise ValueError("Prompt must be nonempty and leave room in the context")

        self.eval()
        device = next(self.parameters()).device
        buckets: dict[int, list[int]] = defaultdict(list)

        for index, prompt in enumerate(prompts):
            buckets[len(prompt)].append(index)

        outputs: list[list[int]] = [[] for _ in prompts]

        for length, indices in buckets.items():
            for start in range(0, len(indices), batch):
                selected = indices[start : start + batch]
                inputs = torch.tensor([prompts[i] for i in selected], device=device)
                done = [False] * len(selected)

                for _ in range(min(max_new_tokens, self.config.context - length)):
                    logits, _ = self(inputs)
                    next_ids = logits[:, -1].argmax(dim=-1)

                    for j, token in enumerate(next_ids.tolist()):
                        if not done[j] and token != eos_id:
                            outputs[selected[j]].append(token)

                        done[j] = done[j] or token == eos_id

                    if all(done):
                        break

                    inputs = torch.cat((inputs, next_ids[:, None]), dim=1)

        return outputs

    def generate(
        self, prompt_ids: list[int], eos_id: int, max_new_tokens: int = 85
    ) -> list[int]:
        return self.generate_batch([prompt_ids], eos_id, max_new_tokens)[0]

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
