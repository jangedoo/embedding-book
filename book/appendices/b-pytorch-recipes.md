# PyTorch Recipes

This appendix collects small PyTorch snippets used throughout the book. They are intentionally minimal so the shapes and modeling assumptions stay visible.

## Embedding lookup

```python
import torch
from torch import nn

embedding = nn.Embedding(num_embeddings=50_000, embedding_dim=768)

ids = torch.tensor([[101, 2023, 2003], [101, 2009, 2515]])
x = embedding(ids)

print(x.shape)  # torch.Size([2, 3, 768])
```

## Factorized embedding layer

```python
import torch
from torch import nn

class FactorizedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, rank: int):
        super().__init__()
        self.codes = nn.Embedding(num_embeddings, rank)
        self.proj = nn.Linear(rank, embedding_dim, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.codes(ids))

layer = FactorizedEmbedding(50_000, 768, rank=128)
out = layer(torch.tensor([1, 2, 3]))
```

This implements:

```{math}
E = AB
```

where the lookup selects rows of `A` and the linear layer applies `B`.

## Pairwise cosine similarity

```python
import torch
import torch.nn.functional as F

x = torch.randn(32, 384)
y = torch.randn(1000, 384)

x = F.normalize(x, dim=-1)
y = F.normalize(y, dim=-1)

scores = x @ y.T
print(scores.shape)  # torch.Size([32, 1000])
```

## Normalized retrieval

```python
import torch
import torch.nn.functional as F

query = torch.randn(384)
docs = torch.randn(20_000, 384)

query = F.normalize(query, dim=0)
docs = F.normalize(docs, dim=-1)

scores = docs @ query
values, indices = scores.topk(k=10)
```

Normalize both sides before indexing if the production system serves cosine search through inner product.

## Dot-product recommendation score

```python
import torch
from torch import nn

users = nn.Embedding(100_000, 64)
items = nn.Embedding(50_000, 64)

user_ids = torch.tensor([4, 8, 15])
item_ids = torch.tensor([16, 23, 42])

score = (users(user_ids) * items(item_ids)).sum(dim=-1)
```

This score uses both vector direction and vector length.

## Top-k retrieval from a score matrix

```python
import torch

queries = torch.randn(8, 384)
docs = torch.randn(1000, 384)

scores = queries @ docs.T
values, indices = scores.topk(k=5, dim=1)

print(indices.shape)  # torch.Size([8, 5])
```

Use normalized `queries` and `docs` first if these scores are meant to represent cosine similarity.

## In-batch contrastive loss

```python
import torch
import torch.nn.functional as F

queries = F.normalize(torch.randn(32, 256), dim=-1)
docs = F.normalize(torch.randn(32, 256), dim=-1)

temperature = 0.05
logits = queries @ docs.T / temperature
labels = torch.arange(queries.shape[0])

loss = F.cross_entropy(logits, labels)
```

The diagonal pairs are positives. Every off-diagonal document is treated as a negative for that query.

## Mask false negatives in contrastive logits

```python
import torch

logits = torch.randn(4, 4)
labels = torch.arange(4)

false_negative_mask = torch.zeros(4, 4, dtype=torch.bool)
false_negative_mask[0, 2] = True

logits = logits.masked_fill(false_negative_mask, -1e9)
```

Do not mask the diagonal positive unless the label is changed too.

## SVD reconstruction

```python
import torch

X = torch.randn(1000, 256)
U, S, Vh = torch.linalg.svd(X, full_matrices=False)

rank = 32
X_hat = (U[:, :rank] * S[:rank]) @ Vh[:rank]
reconstruction_mse = (X - X_hat).pow(2).mean()
```

Use this to study low-rank structure before choosing a factorization rank.

## Remove top principal components

```python
import torch
import torch.nn.functional as F

X = torch.randn(5000, 384)
Xc = X - X.mean(dim=0, keepdim=True)

_, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
top = Vh[:2]

X_clean = Xc - (Xc @ top.T) @ top
X_clean = F.normalize(X_clean, dim=-1)
```

Fit the mean and principal components on a calibration set, then apply the same transform to documents and queries.

## Apply stored component removal

```python
import torch
import torch.nn.functional as F

docs = torch.randn(20_000, 384)
query = torch.randn(384)
mean = torch.zeros(1, 384)
components = F.normalize(torch.randn(2, 384), dim=-1)

def remove_components(X: torch.Tensor, mean: torch.Tensor, components: torch.Tensor):
    Xc = X - mean
    Xr = Xc - (Xc @ components.T) @ components
    return F.normalize(Xr, dim=-1)

docs_clean = remove_components(docs, mean, components)
query_clean = remove_components(query[None, :], mean, components)[0]
```

`components` should have shape `k x d` and contain unit principal directions.

## Recall@k

```python
import torch

scores = torch.randn(100, 1000)
relevant = torch.zeros(100, 1000, dtype=torch.bool)
relevant[torch.arange(100), torch.randint(0, 1000, (100,))] = True

topk = scores.topk(k=10, dim=1).indices
hit = relevant.gather(1, topk).any(dim=1)

recall_at_10 = hit.float().mean()
```

This assumes at least one relevant item per query. For real evaluations, store relevance labels explicitly and handle multiple relevant items.

## Top-k overlap

```python
import torch

old_topk = torch.tensor([[1, 2, 3], [4, 5, 6]])
new_topk = torch.tensor([[2, 3, 9], [4, 8, 9]])

overlap = []
for old, new in zip(old_topk, new_topk):
    overlap.append(len(set(old.tolist()) & set(new.tolist())) / old.numel())

print(sum(overlap) / len(overlap))
```

Use overlap to measure behavior drift after compression, whitening, quantization, or index changes.
