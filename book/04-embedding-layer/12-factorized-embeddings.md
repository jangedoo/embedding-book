# Factorized Embeddings

## Summary

Factorized embeddings replace one large embedding table with two smaller matrices. The idea is simple: instead of giving every token a full independent vector in model space, give each token a smaller latent code and learn a shared projection from that code into the model dimension.

This saves parameters and memory, but it also creates a low-rank bottleneck. The savings are not free.

## Intuition

In a full embedding table, every token owns its own `d`-dimensional vector.

In a factorized embedding table, every token owns a smaller `r`-dimensional code. A shared matrix expands that code into the `d`-dimensional model space.

The token vector becomes a mixture of shared basis directions. Tokens can still differ, but they can only differ through the smaller latent code and the shared projection.

The practical question is whether the original table really needed `d` independent degrees of freedom per token. If many rows share structure, a smaller code can preserve quality. If the task needs many fine-grained token distinctions, the bottleneck can become visible.

## The full embedding table

A normal embedding table is:

```{math}
E \in \mathbb{R}^{V \times d}
```

where:

- `V` is the number of rows
- `d` is the model embedding dimension

Looking up token `i` returns:

```{math}
E_i \in \mathbb{R}^{d}
```

The parameter count is:

```{math}
Vd
```

## The factorized form

A factorized embedding table writes:

```{math}
E = AB
```

where:

```{math}
A \in \mathbb{R}^{V \times r}, \quad B \in \mathbb{R}^{r \times d}, \quad r \ll d
```

Here:

- `A` stores one small latent code per token
- `B` maps the latent code into model space
- `r` is the rank or bottleneck width

For token `i`:

```{math}
E_i = A_i B
```

So the token does not directly own a full independent vector. It owns a smaller latent code `A_i`, and the projection matrix `B` maps that code into model space.

## Parameter count

The full table has:

```{math}
Vd
```

parameters.

The factorized table has:

```{math}
Vr + rd
```

parameters.

Factorization saves parameters when:

```{math}
Vr + rd < Vd
```

Equivalently:

```{math}
r(V + d) < Vd
```

so the rank must satisfy:

```{math}
r < \frac{Vd}{V + d}
```

For a large vocabulary, the dominant term changes from `Vd` to `Vr`. If `r` is much smaller than `d`, this can be a major memory reduction.

When `V` is much larger than `d`, the break-even point is close to `r < d`. When `V` is small, the extra projection matrix `B` can erase much of the benefit.

## Low-rank bottleneck

Because:

```{math}
E = AB
```

the rank of `E` is limited by:

```{math}
\text{rank}(E) \leq r
```

This is the core tradeoff.

The full embedding table can have up to `min(V, d)` independent directions. The factorized table can have at most `r` independent directions. Every token vector is built from the same `r` basis directions stored in `B`.

This often works because embedding tables contain redundancy. But if `r` is too small, unrelated tokens are forced to share capacity and the model loses distinctions it needs.

The important point is that the projection from `r` to `d` does not create new independent information. It expresses each token code in a larger coordinate system. The output vector has `d` numbers, but those vectors still live in an at-most-`r` dimensional subspace before later nonlinear layers or contextual layers transform them.

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

```{math}
A_i B
```

for the selected rows `A_i`.

The lookup output has shape `(..., d)`, so downstream layers can remain unchanged. The compression is hidden inside the embedding layer.

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

Materializing `E` can still be useful for analysis or for exporting a static retrieval table, but it gives up some memory savings at serving time. Whether to materialize depends on whether lookup latency or resident memory is the bottleneck.

## Why this helps

For `V = 250000`, `d = 1024`, and `r = 128`:

Full table:

```{math}
250000 \times 1024 = 256000000
```

Factorized table:

```{math}
250000 \times 128 + 128 \times 1024 = 32131072
```

That is about 32 million parameters instead of 256 million.

The table is roughly 8x smaller before optimizer state.

A small helper makes the tradeoff easier to scan:

```python
def params_full(vocab_size, dim):
    return vocab_size * dim

def params_factorized(vocab_size, rank, dim):
    return vocab_size * rank + rank * dim

for r in [32, 64, 128, 256, 512]:
    ratio = params_factorized(250_000, r, 1024) / params_full(250_000, 1024)
    print(r, round(ratio, 3))
```

This prints the fraction of the full table retained at each rank. The same ratio approximately applies to weight memory and dense optimizer state.

## What this means in real ML systems

Factorization is useful when the embedding table is large and redundant.

In language models, it can reduce input embedding memory for large vocabularies. Some systems also tie input and output embeddings, which changes the tradeoff because the output softmax also depends on the embedding representation.

In recommender systems, factorization is natural: user-item matrix factorization already represents users and items through lower-dimensional latent factors. The same idea appears when categorical feature embeddings are projected into a common model dimension.

In retrieval systems, factorization can compress learned entity or item embeddings, but the serving path must still produce vectors compatible with the similarity metric and index.

In deployed systems there are two common serving modes:

- compute `A_i B` at lookup time, which saves memory but adds a small matrix multiply
- precompute the full rows `E_i`, which speeds lookup but stores the larger table

The first mode is attractive when memory is the bottleneck. The second can be attractive when latency is tight, the table is small enough after export, or an ANN index expects full vectors.

## Capacity interpretation

Full embedding:

> Every token can independently choose any point in `d`-dimensional space.

Factorized embedding:

> Every token must be built from `r` shared components.

The factorized version saves memory because it reduces independent capacity. This can improve generalization for rare tokens by forcing sharing, or hurt performance by merging distinctions that the task needs.

A useful mental model is dictionary learning. The rows of `B` are shared basis directions, and each row of `A` says how strongly a token uses those directions. A rare token no longer has to estimate all `d` coordinates independently, but it also cannot choose a direction outside the shared basis.

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

Good rank sweeps report more than validation loss. For retrieval, measure recall@k, MRR, nDCG, and top-k churn. For recommendation, measure ranking metrics and score calibration. For language models, measure perplexity and downstream task quality, and check rare-token behavior separately.

## Initialization and training details

A factorized embedding layer is usually trained end to end. Both `A` and `B` receive gradients from the same downstream loss.

Initialization matters because the product `A_i B` should have a reasonable scale. If both matrices are initialized too large, embedding norms can explode. If both are initialized too small, early signals can be weak. A practical approach is to use standard embedding initialization for `A`, standard linear-layer initialization for `B`, and then check the norm distribution of `A_i B` against the full embedding baseline.

Weight decay may also need attention. Applying the same regularization to `A` and `B` does not always behave like regularizing the product `E`. Scaling one matrix up and the other down can leave the product similar while changing regularization cost. In high-stakes compression work, monitor row norms and validation quality rather than assuming the original optimizer settings transfer unchanged.

## Visual idea

```{image} ../../assets/figures/factorized-embedding-table.svg
:alt: Full embedding table represented as the product of a latent code matrix A and a shared projection matrix B.
:align: center
:width: 100%
```

The figure contrasts a full `V x d` embedding table with the factorized form `A B`. In the full table, the highlighted token owns an independent row with `d` learned coordinates. In the factorized table, the token owns only a short row in `A`, and the shared projection `B` expands that latent code into model space.

This makes the capacity tradeoff concrete. The output vector still has width `d`, so downstream layers see the expected shape, but every row is assembled from the same `r` shared basis directions. Lowering `r` saves memory and optimizer state, while also limiting how many independent directions the table can represent.

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

As a task-level version, train a tiny classifier or retrieval encoder with a full embedding table, then repeat with ranks such as `16`, `32`, and `64`. Plot memory against validation quality. The useful rank is where the quality curve flattens before the memory curve becomes expensive.

## Common failure modes

- Choosing `r` only from memory budget and not validating quality.
- Forgetting that `rank(E) <= r`.
- Assuming expansion from `r` to `d` restores information that was not present in the code.
- Materializing `E = AB` during serving when row-wise computation would be cheaper.
- Using a factorized input embedding while leaving an expensive output softmax unchanged.
- Over-compressing rare but important tokens.
- Ignoring optimizer-state savings, which may be as important as weight savings during training.
- Comparing parameter counts but not lookup latency when `A_i B` is computed online.
- Assuming the same rank is appropriate for every vocabulary, language, domain, or feature table.
- Forgetting that tied input-output embeddings constrain both the input representation and the output classifier.

## Practical takeaways

Companion notebook: [Factorized embeddings demo](../../notebooks/02_factorized_embeddings.ipynb).

- A factorized embedding writes `E = AB`.
- `A` gives each token a small latent code.
- `B` maps that code into model space.
- The parameter count changes from `Vd` to `Vr + rd`.
- The rank is limited by `r`, creating a low-rank bottleneck.
- Factorization saves memory by reducing independent token capacity.
- Evaluate rank choices with downstream quality, not only parameter count.
- Computing `A_i B` online saves memory; precomputing full rows can save latency.
- Dimension expansion from `r` to `d` changes coordinates but does not restore information removed by the bottleneck.
