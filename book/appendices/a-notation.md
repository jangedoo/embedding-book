# Notation

This appendix collects the symbols used throughout the book. Shapes are included because many embedding bugs are shape bugs.

## Common symbols

| Symbol | Meaning |
|---|---|
| `V` | vocabulary size or number of IDs |
| `d` | embedding dimension |
| `r` | factorization rank |
| `E` | embedding matrix |
| `x`, `y` | vectors |
| `W` | linear layer weight matrix |
| `b` | bias vector |
| `sim(x, y)` | similarity function |
| `q` | query vector |
| `d_i` | document or candidate vector |
| `k` | retrieval cutoff or number of neighbors |
| `tau` | softmax or contrastive temperature |

## Shapes

| Object | Shape | Meaning |
|---|---:|---|
| `E` | `V x d` | embedding table |
| `E_i` | `d` | row vector for ID `i` |
| `ids` | `B` | batch of IDs |
| `X = E[ids]` | `B x d` | batch of looked-up embeddings |
| `tokens` | `B x L` | batch of token sequences |
| `H` | `B x L x d` | sequence hidden states |
| `U` | `M x d` | user embedding table |
| `V_items` | `N x d` | item embedding table |
| `A` | `V x r` | low-rank token codes |
| `B` | `r x d` | low-rank projection into model space |
| `Q` | `n_q x d` | query embeddings |
| `D` | `n_d x d` | document embeddings |
| `S = QD^T` | `n_q x n_d` | score matrix |
| `topk(S)` | `n_q x k` | candidate IDs for each query |
| `C` | `d x d` | covariance matrix |
| `U_k` | `d x k` | selected principal directions |

## Geometry

Dot product:

```math
x^\top y = \sum_i x_i y_i
```

L2 norm:

```math
\|x\|_2 = \sqrt{\sum_i x_i^2}
```

Cosine similarity:

```math
\cos(x, y) = \frac{x^\top y}{\|x\|_2\|y\|_2}
```

Squared Euclidean distance:

```math
\|x-y\|_2^2 = \sum_i (x_i-y_i)^2
```

Normalized vector:

```math
\hat{x} = \frac{x}{\|x\|_2}
```

When `x` and `y` are normalized:

```math
\|x-y\|_2^2 = 2 - 2x^\top y
```

So cosine similarity, dot product, and Euclidean distance induce the same ranking over candidates.

Projection onto a unit direction `v`:

```math
score(x) = x^\top v
```

Remove a unit direction `v`:

```math
x' = x - (x^\top v)v
```

## Learning and scoring

Linear layer:

```math
y = Wx + b
```

Softmax:

```math
p_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
```

Temperature-scaled logits:

```math
z_i = \frac{x^\top w_i}{\tau}
```

Matrix factorization:

```math
R \approx UV^\top
```

Factorized embedding table:

```math
E = AB
```

with:

```math
A \in \mathbb{R}^{V \times r}, \quad B \in \mathbb{R}^{r \times d}
```

Contrastive loss with one positive and sampled negatives:

```math
L = -\log \frac{\exp(q^\top x^+ / \tau)}
{\exp(q^\top x^+ / \tau) + \sum_j \exp(q^\top x_j^- / \tau)}
```

## Retrieval metrics

Recall@k:

```math
Recall@k = \frac{\text{queries with a relevant item in top k}}{\text{queries}}
```

Mean reciprocal rank:

```math
MRR = \frac{1}{Q}\sum_{q=1}^{Q}\frac{1}{rank_q}
```

Top-k overlap between two systems:

```math
overlap@k = \frac{|N_k^{old}(q) \cap N_k^{new}(q)|}{k}
```

Mean reciprocal rank assumes `rank_q` is the position of the first relevant result. If a query has no relevant result in the evaluated candidate list, define its reciprocal rank as zero for that list.

## Convention notes

Vectors are treated as row or column vectors depending on context. PyTorch batches usually store vectors as rows, so scores are often written as:

```python
scores = queries @ documents.T
```

Mathematical notation often writes a single vector product as `x^T y`. The meaning is the same: multiply matching coordinates and sum.

Unless stated otherwise, retrieval examples assume higher scores are better. Distance metrics such as Euclidean distance are usually sorted in ascending order, so code should convert carefully when mixing distances and similarities.
