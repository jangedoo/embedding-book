# Embeddings as Matrix Factorization

Many embedding systems can be viewed as factorizing a relationship matrix. This perspective connects word embeddings, recommender systems, retrieval models, and low-rank compression under one mental model: learn smaller matrices whose products explain observed relationships.

Factorization is useful because real systems often contain huge sparse matrices. We rarely store or model every user-item, word-context, or query-document relationship directly.

## Intuition

Suppose you have a matrix where rows are users and columns are movies. Most entries are missing, but observed entries say whether a user watched or liked a movie. Matrix factorization says: instead of learning one independent score per user-movie pair, learn a vector for each user and a vector for each movie. Their dot product reconstructs the score.

The same idea appears in language. A word can be represented by how often it appears near context words. Embeddings learn low-dimensional coordinates that approximate this large word-context relationship table.

## Mathematical object

Let:

```math
R \in \mathbb{R}^{m \times n}
```

be a relationship matrix. A rank-`d` factorization approximates it as:

```math
R \approx UV^\top
```

where:

```math
U \in \mathbb{R}^{m \times d}, \quad V \in \mathbb{R}^{n \times d}
```

The predicted relationship between row `i` and column `j` is:

```math
\hat{R}_{ij} = u_i^\top v_j
```

Low-rank factorization assumes the observed relationships can be explained by fewer latent factors than there are rows or columns.

Examples:

- word-context matrix
- user-item interaction matrix
- query-document relevance matrix
- entity-relation matrix
- token embedding table compression

## PyTorch equivalent

```python
import torch
from torch import nn

num_users, num_items, d = 10_000, 5_000, 64
U = nn.Embedding(num_users, d)
V = nn.Embedding(num_items, d)

user_id = torch.tensor([1, 2, 3])
item_id = torch.tensor([10, 20, 30])

pred = (U(user_id) * V(item_id)).sum(dim=-1)
```

For a dense matrix, truncated SVD gives the best rank-`d` approximation under squared reconstruction error:

```python
R = torch.randn(1000, 500)
U_svd, S, Vh = torch.linalg.svd(R, full_matrices=False)

d = 32
R_hat = (U_svd[:, :d] * S[:d]) @ Vh[:d]
error = (R - R_hat).pow(2).mean()
```

In recommender systems, matrices are sparse and missing entries are not simply zeros. Training uses observed examples, negative sampling, weighting, and regularization.

Bias terms are often essential:

```math
\hat{R}_{ij} = \mu + b_i + c_j + u_i^\top v_j
```

The dot product should explain interaction-specific structure, not spend all its capacity modeling global popularity or row activity.

## What this means in ML systems

The factorization view helps explain:

- why embeddings can compress huge relationship tables
- why dimension `d` controls capacity
- why dot product is a natural scoring function
- why popularity and frequency can dominate
- why low-rank bottlenecks save memory but lose independent detail

It also clarifies factorized embedding tables. A full table:

```math
E \in \mathbb{R}^{V \times d}
```

can be replaced by:

```math
E = AB
```

where:

```math
A \in \mathbb{R}^{V \times r}, \quad B \in \mathbb{R}^{r \times d}, \quad r \ll d
```

Each token owns a small latent code in `A`; `B` maps that code into model space. This saves memory but restricts the table to rank at most `r`.

This restriction is useful when many rows share structure. It is harmful when rare rows need independent directions that cannot be reconstructed from the shared basis.

## Common failure modes

- Assuming missing entries are negative examples without checking data collection.
- Choosing a dimension so small that important relationships collapse.
- Choosing a dimension so large that frequent artifacts are memorized.
- Ignoring bias terms for row and column popularity.
- Evaluating reconstruction error when the real task is ranking.
- Treating a low-rank approximation as neutral compression when it changes rare items most.

## Visual idea

Draw a large sparse user-item matrix decomposed into a tall user matrix and a tall item matrix. Highlight one user row and one item row whose dot product reconstructs a single cell. Next to it, draw a full token embedding table factorized into `A` and `B`.

Add a rank slider to the visual: as rank increases, reconstruction improves but memory grows. Show rare rows separately so average reconstruction error does not hide their degradation.

## Small experiment

Create a synthetic low-rank matrix plus noise. Fit rank `r` approximations for several values of `r`. Plot reconstruction error and top-k neighbor stability. Then repeat with a matrix containing rare rows. This shows that average error can look good while rare entities degrade.

For a token-table experiment, factorize a trained embedding matrix with several ranks and evaluate nearest-neighbor overlap for frequent, medium-frequency, and rare tokens separately.

## Practical takeaways

Embeddings often work because large relationship tables have useful low-rank structure.

The rank or dimension is a capacity knob. Tune it against the downstream task, not only reconstruction error or memory savings.
