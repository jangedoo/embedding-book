# Vectors, Norms, Angles, and Projections

An embedding vector is just a list of numbers, but the operations on that list create the behavior we care about: similarity, distance, projection, compression, and retrieval.

Embeddings live in vector spaces. Before we can reason about semantic meaning, we need to reason about vector geometry.

## Summary

Vectors give embeddings coordinates. Norms measure length, angles measure direction, dot products combine both, and projections measure how much of a vector lies along a direction. These are not abstract operations: they become attention scores, retrieval scores, classifier logits, bottlenecks, and diagnostics in real systems.

## Intuition

Think of an embedding as a point or arrow in `d` dimensions. A model rarely reads one coordinate in isolation. It mostly asks geometric questions: how long is this vector, which direction does it point, how aligned is it with another vector, and how much of it lies along a learned direction?

The previous chapter defined the embedding table. This chapter defines the basic operations that make its rows useful.

## Vector

A vector is an ordered list of numbers:

```math
x = [x_1, x_2, \ldots, x_d]
```

For embeddings, `d` is the embedding dimension.

In PyTorch:

```python
import torch

x = torch.tensor([0.2, -1.0, 0.5])
print(x.shape)  # torch.Size([3])
```

For a batch of `B` embeddings:

```math
X \in \mathbb{R}^{B \times d}
```

The coordinates depend on the learned space. Coordinate 17 is not automatically "sentiment" or "price sensitivity." Meaning usually appears in directions, subspaces, neighborhoods, and downstream scores, not in isolated dimensions.

## Norm

```math
\|x\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_d^2}
```

It measures vector length.

```python
length = torch.linalg.vector_norm(x, ord=2)
```

Length can matter. In some systems it represents confidence, popularity, frequency, or activation strength. In others it is mostly an optimization artifact.

For a batch:

```python
X = torch.randn(32, 768)
lengths = torch.linalg.vector_norm(X, dim=-1)
```

The next chapters return to length because it changes ranking under dot product and Euclidean distance.

## Dot product

```math
x \cdot y = \sum_i x_i y_i
```

The dot product combines direction and length:

```math
x \cdot y = \|x\| \|y\| \cos(\theta)
```

This equation is one of the most useful equations in embedding work.

It says the dot product increases when vectors point in similar directions and when either vector gets longer.

```python
y = torch.tensor([0.1, -0.8, 0.3])
score = torch.dot(x, y)
```

For a batch of candidate embeddings:

```python
query = torch.randn(768)
items = torch.randn(1_000, 768)
scores = items @ query
```

This is the core operation behind many recommendation scores, attention scores, and exact nearest-neighbor baselines.

## Angle

The angle between two vectors is defined through cosine similarity:

```math
\cos(\theta) = \frac{x \cdot y}{\|x\|_2 \|y\|_2}
```

Cosine similarity cares about direction. If two vectors are multiplied by positive constants, their cosine similarity does not change.

```python
cos = torch.nn.functional.cosine_similarity(x[None, :], y[None, :])
```

This is why cosine is common for sentence embedding retrieval: the system often wants directional semantic match rather than raw magnitude.

Cosine is undefined for the all-zero vector, so practical implementations add a small epsilon during normalization.

## Projection

The scalar projection of `x` onto a unit direction `u` is:

```math
x \cdot u
```

The vector projection is:

```math
(x \cdot u)u
```

Projection answers a practical question:

> How much of this embedding lies in a particular direction?

If a direction represents "medical topic" or "high purchase intent", projection gives a simple way to score how strongly an embedding expresses that direction. Real learned directions are rarely this clean, but the operation is the same.

```python
u = torch.nn.functional.normalize(torch.tensor([1.0, 1.0, 0.0]), dim=0)
amount = torch.dot(x, u)
projection = amount * u
```

If `u` is not unit length, use:

```math
\operatorname{proj}_u(x) = \frac{x^\top u}{u^\top u}u
```

Normalizing the direction first is often simpler and less error-prone.

## PyTorch and NumPy equivalents

```python
import numpy as np

x_np = np.array([0.2, -1.0, 0.5])
y_np = np.array([0.1, -0.8, 0.3])

dot_np = x_np @ y_np
norm_np = np.linalg.norm(x_np)
cos_np = dot_np / (np.linalg.norm(x_np) * np.linalg.norm(y_np))
```

The PyTorch versions are the same operations with tensors, autograd, batching, and GPU support. The shape discipline is the important part: a query of shape `[d]` scored against candidates of shape `[N, d]` produces `N` scores.

## Practical interpretation

Vector operations become model operations:

- dot products score compatibility
- norms measure magnitude
- angles compare direction
- projections measure components along learned directions
- matrix multiplication applies many projections at once

A linear layer is a bank of learned projections. A nearest-neighbor search is many distance or similarity computations. A classifier head often compares an embedding with one vector per class.

This also explains why metric choices matter. If the downstream system ranks by dot product, it uses both angle and length. If it normalizes vectors and ranks by cosine, it mostly uses direction. [Distance and Similarity](03-distance-and-similarity.md) makes that distinction explicit.

## Common failure modes

- Treating each coordinate as independently interpretable without evidence.
- Forgetting that dot product changes when vector norms change.
- Comparing vectors with cosine even though the model was trained with dot product.
- Normalizing vectors before a downstream component that expects magnitude information.
- Explaining 768-dimensional geometry only from a 2D plot.
- Forgetting shape conventions and accidentally scoring across a batch dimension instead of the embedding dimension.

## Visual idea

Draw two arrows from the origin, mark their angle, show their lengths, and drop a perpendicular line from one vector onto the other. Label the projection as "component of x along y".

## Small experiment

Generate three vectors: `x`, `2x`, and a random vector `z`. Compare dot product, cosine similarity, and Euclidean distance between `x` and the other two vectors. The result shows why "similar" depends on the metric.

```python
import torch
import torch.nn.functional as F

x = torch.tensor([1.0, 2.0, 0.0])
candidates = torch.stack([2 * x, torch.tensor([1.0, 0.0, 2.0])])

dot = candidates @ x
cos = F.cosine_similarity(candidates, x[None, :], dim=-1)
dist = torch.linalg.vector_norm(candidates - x, dim=-1)

print(dot)
print(cos)
print(dist)
```

`2x` has perfect cosine similarity with `x`, a larger dot product than `x` itself would have, and nonzero Euclidean distance because it is longer.

## Practical takeaways

For embeddings, geometry is not decoration. Norms, angles, and projections are the primitives that become retrieval scores, model logits, attention scores, and clustering behavior.

When debugging an embedding system, inspect the actual operations and tensor shapes before interpreting the model's semantics.
