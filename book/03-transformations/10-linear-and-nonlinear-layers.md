# How Linear and Nonlinear Layers Reshape Embeddings

Embeddings rarely stay as raw lookup vectors. They pass through projections, MLPs, activations, normalization layers, residual connections, and attention blocks.

This chapter builds intuition for what those transformations do to the geometry.

## Linear layer

A linear layer computes:

```math
y = Wx + b
```

If:

```math
x \in \mathbb{R}^{d_{in}}, \quad W \in \mathbb{R}^{d_{out} \times d_{in}}
```

then:

```math
y \in \mathbb{R}^{d_{out}}
```

A linear layer changes coordinates. It can rotate, scale, shear, reflect, project, or translate the embedding space.

## Fewer dimensions: compression and bottlenecks

If `d_out < d_in`, the layer maps vectors into a smaller space.

Practical interpretation:

- some information must be discarded unless the original vectors lie on a lower-dimensional structure
- the model learns which directions are useful for the next task
- different original vectors can collapse to the same output
- this can regularize or damage performance depending on the task

A 768 → 128 projection is not just "making vectors smaller." It is asking the model to preserve only the information that is useful downstream.

## More dimensions: expansion and feature construction

If `d_out > d_in`, the layer maps vectors into a larger space.

Important intuition:

> Increasing dimension does not magically create new information by itself.

A purely linear expansion from 2D to 8D still has at most 2 intrinsic degrees of freedom. The points live on a transformed 2D plane inside 8D.

So why expand?

- to give later nonlinear layers more room
- to create many learned views of the same input
- to make gating and feature selection easier
- to support overcomplete representations

## Rank controls information

The rank of `W` controls how many independent directions can pass through.

If `W` is low rank, it creates a bottleneck even if `d_out` is large.

## Bias translates the space

The bias term shifts all points by the same learned vector.

This matters before nonlinearities. A bias can move points across ReLU thresholds, changing which neurons activate.

## ReLU

ReLU computes:

```math
ReLU(x) = max(0, x)
```

For a vector, it applies this coordinate by coordinate.

## What ReLU does geometrically

ReLU is a gate.

For each coordinate:

- positive values pass through
- negative values are set to zero

This creates piecewise-linear regions. Inside one region, the network behaves like a linear function. Across regions, the active set changes, so the function changes.

Practical interpretation:

- ReLU can discard negative evidence
- it creates sparse activations
- it partitions space into regions
- it allows an MLP to separate patterns that one linear layer cannot
- it makes "which side of a learned boundary am I on?" matter

## Linear + ReLU

A layer like:

```math
h = ReLU(Wx + b)
```

does two things:

1. It projects the input onto learned directions.
2. It keeps only directions whose activation is positive.

Each neuron asks a question:

> Does this embedding have enough of this learned feature direction?

If yes, the feature is passed forward. If no, it is shut off.

## MLP as space reshaping

An MLP repeatedly creates learned feature directions, gates them, recombines them, and changes the neighborhood structure.

Points that were close before the MLP may become far apart. Points that were far apart may become close if the model learns to ignore irrelevant directions.

## Residual connections

A residual block adds transformed features back to the original:

```math
y = x + f(x)
```

Interpretation:

- preserve the old representation
- add a learned correction
- make optimization easier
- allow gradual reshaping instead of total replacement

## Normalization

LayerNorm changes scale and centering per example. It makes representations more stable, but it also changes what vector length means.

After normalization, raw norm is less meaningful inside the model than direction and relative coordinate pattern.

## Practical takeaways

- A linear layer learns which directions matter.
- A smaller output dimension is a bottleneck.
- A larger output dimension is an expansion of views, not automatic new information.
- ReLU gates learned features and creates piecewise-linear geometry.
- MLPs reshape neighborhoods, not just individual vectors.
- Residuals preserve old information while adding corrections.
- Normalization changes the interpretation of magnitude.

## Experiment idea

Create 2D points that cannot be separated by one line. Pass them through a small MLP with ReLU. Plot the hidden activations and show how the MLP makes the classes separable.
