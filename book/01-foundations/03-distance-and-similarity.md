# Distance and Similarity

Similarity is not one thing. It is a modeling choice.

The same embeddings can produce different nearest neighbors depending on whether we use cosine similarity, dot product, or Euclidean distance.

## Dot product

```math
x \cdot y = \|x\| \|y\| \cos(\theta)
```

Dot product rewards both alignment and length.

Practical interpretation:

- useful when vector norm carries signal
- common in recommender systems
- can mix relevance with popularity or confidence

## Cosine similarity

```math
\cos(x, y) = \frac{x \cdot y}{\|x\|\|y\|}
```

Cosine compares direction and ignores length.

## Euclidean distance

```math
\|x-y\|_2
```

Euclidean distance uses both direction and length.

If vectors are normalized:

```math
\|x-y\|^2 = 2 - 2\cos(x, y)
```

So for normalized vectors, cosine similarity and Euclidean distance give the same ranking, just reversed and transformed.

Without normalization, Euclidean distance can strongly care about norm differences.

## Practical takeaway

Before choosing a metric, ask:

1. Does vector length contain useful signal?
2. Was the model trained with this metric?
3. Does the ANN index support this metric?
4. Do I normalize before storage, before query, both, or neither?
