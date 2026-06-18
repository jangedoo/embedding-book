# Vectors, Norms, Angles, and Projections

An embedding vector is just a list of numbers, but the operations on that list create the behavior we care about: similarity, distance, projection, compression, and retrieval.

Embeddings live in vector spaces. Before we can reason about semantic meaning, we need to reason about vector geometry.

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

## Norm

```math
\|x\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_d^2}
```

It measures vector length.

```python
length = torch.linalg.vector_norm(x, ord=2)
```

Length can matter. In some systems it represents confidence, popularity, frequency, or activation strength. In others it is mostly an optimization artifact.

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

## Practical interpretation

Vector operations become model operations:

- dot products score compatibility
- norms measure magnitude
- angles compare direction
- projections measure components along learned directions
- matrix multiplication applies many projections at once

A linear layer is a bank of learned projections. A nearest-neighbor search is many distance or similarity computations. A classifier head often compares an embedding with one vector per class.

## Common failure modes

- Treating each coordinate as independently interpretable without evidence.
- Forgetting that dot product changes when vector norms change.
- Comparing vectors with cosine even though the model was trained with dot product.
- Normalizing vectors before a downstream component that expects magnitude information.
- Explaining 768-dimensional geometry only from a 2D plot.

## Visual idea

Draw two arrows from the origin, mark their angle, show their lengths, and drop a perpendicular line from one vector onto the other. Label the projection as "component of x along y".

## Small experiment

Generate three vectors: `x`, `2x`, and a random vector `z`. Compare dot product, cosine similarity, and Euclidean distance between `x` and the other two vectors. The result shows why "similar" depends on the metric.

## Practical takeaways

For embeddings, geometry is not decoration. Norms, angles, and projections are the primitives that become retrieval scores, model logits, attention scores, and clustering behavior.
