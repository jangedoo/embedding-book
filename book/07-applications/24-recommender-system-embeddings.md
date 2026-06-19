# Recommender-System Embeddings

Recommender systems often learn user and item embeddings so they can score matches between people and items. The vectors are not just semantic descriptions of products. They are coordinates shaped by exposure, clicks, purchases, ratings, dwell time, skips, and the objective used for ranking.

## Intuition

A user vector can be read as a compact summary of preferences. An item vector can be read as a compact summary of who tends to interact with it. If the vectors point in compatible directions, the recommender gives the item a high score.

This is useful because the model can generalize. A user who has never seen an item can still receive it if the item is close to other items the user likes.

## Mathematical object

With `M` users, `N` items, and dimension `d`:

```{math}
U \in \mathbb{R}^{M \times d}, \quad V \in \mathbb{R}^{N \times d}
```

A simple score is:

```{math}
score(user, item) = u \cdot v
```

Often we add bias terms:

```{math}
s_{ui} = u_u^\top v_i + b_u + b_i
```

For implicit feedback, training may compare observed interactions against sampled negatives:

```{math}
L = -\log \sigma(u^\top v^+) - \sum_j \log \sigma(-u^\top v_j^-)
```

The score depends on both angle and norm. Item norm can encode popularity or confidence, while angle can encode preference match.

## PyTorch equivalent

```python
import torch
from torch import nn

num_users, num_items, d = 100_000, 50_000, 64
users = nn.Embedding(num_users, d)
items = nn.Embedding(num_items, d)
item_bias = nn.Embedding(num_items, 1)

user_ids = torch.tensor([12, 99, 42])
item_ids = torch.tensor([5, 7, 5])

u = users(user_ids)
v = items(item_ids)
score = (u * v).sum(dim=-1) + item_bias(item_ids).squeeze(-1)
```

For candidate retrieval, item vectors can be stored in an ANN index and searched with maximum inner product or cosine similarity, depending on the model design.

When serving maximum inner product search, confirm that the ANN backend is configured for inner product. Accidentally switching to cosine can erase useful norm information, while accidentally switching from cosine to dot product can over-promote high-norm items.

## What this means in ML systems

Recommendation embeddings face different pressures than sentence embeddings:

- observations are biased by what users were shown
- feedback can be implicit and noisy
- popularity can dominate gradients
- new users and new items need cold-start handling
- ranking metrics may conflict with diversity or business constraints

Dot product is common because vector length can carry useful confidence. But high-norm popular items can swamp personalized matches. Bias terms, regularization, sampling strategy, and reranking constraints all shape the final system.

Most recommenders are two-stage systems. Embeddings retrieve a few hundred or thousand candidates quickly; a heavier ranker then uses features, constraints, freshness, and business rules. The embedding stage should be judged by candidate recall and candidate quality, not by final ranking metrics alone.

## Common failure modes

- Treating missing interactions as true dislikes.
- Sampling negatives from the wrong distribution.
- Letting popular items win mostly through large norms.
- Evaluating with random splits that leak future behavior.
- Ignoring cold-start users and items.
- Optimizing clicks while harming long-term satisfaction or diversity.
- Serving approximate retrieval with a metric different from training.

## Visual idea

```{image} ../../assets/figures/recommender-norm-bias-funnel.svg
:alt: User and item vectors showing norm-driven popularity bias, angular niche alignment, and a candidate-to-rerank funnel.
:align: center
:width: 100%
```

This figure separates two ingredients in recommender scoring: direction and length. A high-norm popular item can achieve a large dot product for many users, while a lower-norm niche item may be more angularly aligned with one specific user's taste. Cosine similarity, raw dot product, and norm clipping each make a different choice about how much popularity-like magnitude should influence ranking.

The funnel shows why retrieval mistakes are hard to repair later. If the embedding index fails to include a relevant niche item in the ANN candidate set, the reranker never sees it. Measuring candidate recall, norm distributions, coverage, and popularity bias is therefore as important as measuring the final ranked list.

## Small experiment

Train matrix factorization on a small implicit-feedback dataset. Compare recommendations using raw dot product, cosine similarity, and dot product with item-norm clipping. Report recall@10, catalog coverage, and average popularity of recommended items. This makes popularity bias visible.

Use a time-based split rather than a random split. Random splits can let the model train on behavior that happened after the evaluation interaction.

## Practical takeaways

Recommender embeddings are preference and exposure models, not neutral item maps.

Always evaluate both relevance and distributional behavior: popularity, diversity, coverage, novelty, and cold-start performance.
