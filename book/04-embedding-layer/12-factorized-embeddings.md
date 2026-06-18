# Factorized Embeddings

A normal embedding table is:

```math
E \in \mathbb{R}^{V \times d}
```

A factorized embedding table writes:

```math
E = AB
```

where:

```math
A \in \mathbb{R}^{V \times r}
```

and:

```math
B \in \mathbb{R}^{r \times d}
```

with `r << d`.

## Intuition

A token no longer owns a full independent vector. It owns a smaller latent code of size `r`.

The matrix `B` maps that latent code into the model embedding space.

So each token vector becomes a combination of shared basis directions.

## What this really means

Full embedding:

> Every token can independently choose any point in `d`-dimensional space.

Factorized embedding:

> Every token must be built from `r` shared components.

This is a capacity constraint.

## PyTorch sketch

```python
import torch
from torch import nn

class FactorizedEmbedding(nn.Module):
    def __init__(self, vocab_size, rank, dim):
        super().__init__()
        self.codes = nn.Embedding(vocab_size, rank)
        self.proj = nn.Linear(rank, dim, bias=False)

    def forward(self, ids):
        return self.proj(self.codes(ids))
```

## Parameter count

Full:

```math
Vd
```

Factorized:

```math
Vr + rd
```

Factorization helps when:

```math
Vr + rd < Vd
```
