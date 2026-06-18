# Vectors, Norms, Angles, and Projections

Embeddings live in vector spaces. Before we can reason about semantic meaning, we need to reason about vector geometry.

## Vector

A vector is an ordered list of numbers:

```math
x = [x_1, x_2, \ldots, x_d]
```

For embeddings, `d` is the embedding dimension.

## Norm

```math
\|x\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_d^2}
```

It measures vector length.

## Dot product

```math
x \cdot y = \sum_i x_i y_i
```

The dot product combines direction and length:

```math
x \cdot y = \|x\| \|y\| \cos(\theta)
```

This equation is one of the most useful equations in embedding work.
