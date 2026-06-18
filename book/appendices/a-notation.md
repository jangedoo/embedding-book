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

## Convention notes

Vectors are treated as row or column vectors depending on context. PyTorch batches usually store vectors as rows, so scores are often written as:

```python
scores = queries @ documents.T
```

Mathematical notation often writes a single vector product as `x^T y`. The meaning is the same: multiply matching coordinates and sum.
