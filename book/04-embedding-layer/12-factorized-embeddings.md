# Factorized Embeddings

Factorized embeddings replace one large embedding table with two smaller matrices. The idea is simple: instead of giving every token a full independent vector in model space, give each token a smaller latent code and learn a shared projection from that code into the model dimension.

This saves parameters and memory, but it also creates a low-rank bottleneck. The savings are not free.

## Intuition

In a full embedding table, every token owns its own `d`-dimensional vector.

In a factorized embedding table, every token owns a smaller `r`-dimensional code. A shared matrix expands that code into the `d`-dimensional model space.

The token vector becomes a mixture of shared basis directions. Tokens can still differ, but they can only differ through the smaller latent code and the shared projection.

## The full embedding table

A normal embedding table is:

```math
E \in \mathbb{R}^{V \times d}
```

where:

- `V` is the number of rows
- `d` is the model embedding dimension

Looking up token `i` returns:

```math
E_i \in \mathbb{R}^{d}
```

The parameter count is:

```math
Vd
```

## The factorized form

A factorized embedding table writes:

```math
E = AB
```

where:

```math
A \in \mathbb{R}^{V \times r}, \quad B \in \mathbb{R}^{r \times d}, \quad r \ll d
```

Here:

- `A` stores one small latent code per token
- `B` maps the latent code into model space
- `r` is the rank or bottleneck width

For token `i`:

```math
E_i = A_i B
```

So the token does not directly own a full independent vector. It owns a smaller latent code `A_i`, and the projection matrix `B` maps that code into model space.

## Parameter count

The full table has:

```math
Vd
```

parameters.

The factorized table has:

```math
Vr + rd
```

parameters.

Factorization saves parameters when:

```math
Vr + rd < Vd
```

For a large vocabulary, the dominant term changes from `Vd` to `Vr`. If `r` is much smaller than `d`, this can be a major memory reduction.

## Low-rank bottleneck

Because:

```math
E = AB
```

the rank of `E` is limited by:

```math
\text{rank}(E) \leq r
```

This is the core tradeoff.

The full embedding table can have up to `min(V, d)` independent directions. The factorized table can have at most `r` independent directions. Every token vector is built from the same `r` basis directions stored in `B`.

This often works because embedding tables contain redundancy. But if `r` is too small, unrelated tokens are forced to share capacity and the model loses distinctions it needs.

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

emb = FactorizedEmbedding(vocab_size=250_000, rank=128, dim=1024)
ids = torch.tensor([1, 5, 42])
x = emb(ids)
print(x.shape)
```

This implements:

```math
A_i B
```

for the selected rows `A_i`.

## Numpy equivalent

```python
import numpy as np

V = 250_000
r = 128
d = 1024

A = np.random.randn(V, r).astype("float32")
B = np.random.randn(r, d).astype("float32")

ids = np.array([1, 5, 42])
x = A[ids] @ B

print(x.shape)
```

The matrix `E` does not need to be materialized. In practice, computing `A[ids] @ B` is usually cheaper than building the full `V x d` matrix.

## Why this helps

For `V = 250000`, `d = 1024`, and `r = 128`:

Full table:

```math
250000 \times 1024 = 256000000
```

Factorized table:

```math
250000 \times 128 + 128 \times 1024 = 32131072
```

That is about 32 million parameters instead of 256 million.

The table is roughly 8x smaller before optimizer state.

## What this means in real ML systems

Factorization is useful when the embedding table is large and redundant.

In language models, it can reduce input embedding memory for large vocabularies. Some systems also tie input and output embeddings, which changes the tradeoff because the output softmax also depends on the embedding representation.

In recommender systems, factorization is natural: user-item matrix factorization already represents users and items through lower-dimensional latent factors. The same idea appears when categorical feature embeddings are projected into a common model dimension.

In retrieval systems, factorization can compress learned entity or item embeddings, but the serving path must still produce vectors compatible with the similarity metric and index.

## Capacity interpretation

Full embedding:

> Every token can independently choose any point in `d`-dimensional space.

Factorized embedding:

> Every token must be built from `r` shared components.

The factorized version saves memory because it reduces independent capacity. This can improve generalization for rare tokens by forcing sharing, or hurt performance by merging distinctions that the task needs.

## Choosing the rank

The rank `r` should be treated as a compression knob.

Small `r`:

- lower memory
- lower optimizer state
- stronger sharing
- higher risk of underfitting

Large `r`:

- more capacity
- more memory
- closer to the full embedding table
- less compression benefit

A practical workflow is to train or fine-tune with several ranks and measure task quality, retrieval quality, and latency. The best rank is usually empirical.

## Visual idea

Draw a large `V x d` table. Then draw it as the product of a tall thin `V x r` matrix and a short wide `r x d` matrix. Highlight one token row in `A`, then show arrows from its `r` code values to shared basis directions in `B`.

## Small experiment

Create a random full embedding table `E`, compute low-rank approximations with SVD, and measure reconstruction error as `r` increases.

```python
import torch

V, d = 1000, 64
E = torch.randn(V, d)

U, S, Vh = torch.linalg.svd(E, full_matrices=False)

for r in [4, 8, 16, 32, 64]:
    E_hat = (U[:, :r] * S[:r]) @ Vh[:r]
    rel_error = (E - E_hat).norm() / E.norm()
    print(r, float(rel_error))
```

Then replace `E` with a structured table, such as clustered vectors, and compare the errors. Structured embeddings should compress better than random independent vectors.

## Common failure modes

- Choosing `r` only from memory budget and not validating quality.
- Forgetting that `rank(E) <= r`.
- Assuming expansion from `r` to `d` restores information that was not present in the code.
- Materializing `E = AB` during serving when row-wise computation would be cheaper.
- Using a factorized input embedding while leaving an expensive output softmax unchanged.
- Over-compressing rare but important tokens.
- Ignoring optimizer-state savings, which may be as important as weight savings during training.

## Practical takeaways

- A factorized embedding writes `E = AB`.
- `A` gives each token a small latent code.
- `B` maps that code into model space.
- The parameter count changes from `Vd` to `Vr + rd`.
- The rank is limited by `r`, creating a low-rank bottleneck.
- Factorization saves memory by reducing independent token capacity.
- Evaluate rank choices with downstream quality, not only parameter count.
