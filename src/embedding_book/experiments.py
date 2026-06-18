from __future__ import annotations

import torch
from torch import nn


def embedding_parameter_count(vocab_size: int, dim: int) -> int:
    return vocab_size * dim


def factorized_embedding_parameter_count(vocab_size: int, rank: int, dim: int) -> int:
    return vocab_size * rank + rank * dim


class FactorizedEmbedding(nn.Module):
    """
    Low-rank embedding layer: E = A @ B.

    A token owns a small rank-dimensional code, then a shared projection maps
    that code into the model embedding space.
    """

    def __init__(self, vocab_size: int, rank: int, dim: int):
        super().__init__()
        self.codes = nn.Embedding(vocab_size, rank)
        self.proj = nn.Linear(rank, dim, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes(ids))
