# How Linear and Nonlinear Layers Reshape Embeddings

Linear and nonlinear layers do not merely "process" embeddings. They change the coordinate system, compress or expand directions, gate regions of space, and reshape which points become neighbors.

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

In PyTorch:

```python
import torch
from torch import nn

layer = nn.Linear(in_features=3, out_features=2)
x = torch.randn(8, 3)
y = layer(x)

print(y.shape)  # torch.Size([8, 2])
```

The matrix `W` contains one learned row per output coordinate. Each output coordinate is a learned projection of the input plus a bias.

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

More precisely:

```math
\operatorname{rank}(W) \le \min(d_{out}, d_{in})
```

If:

```math
\operatorname{rank}(W) = k
```

then at most `k` independent input directions can affect the output. A 2D to 8D linear expansion can produce eight coordinates, but those coordinates still lie in at most a 2D linear subspace before a nonlinearity changes the geometry.

You can test this with singular values:

```python
W = torch.randn(8, 2)
print(torch.linalg.matrix_rank(W))  # at most 2
```

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

For a single hidden unit:

```math
h_j = \max(0, w_j^\top x + b_j)
```

The boundary:

```math
w_j^\top x + b_j = 0
```

is a hyperplane. One side passes through with a positive value; the other side is clamped to zero. Many hidden units create many such half-space gates.

```python
points = torch.tensor([
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0],
])

gate = nn.Sequential(nn.Linear(2, 4), nn.ReLU())
hidden = gate(points)
```

## MLP as space reshaping

An MLP repeatedly creates learned feature directions, gates them, recombines them, and changes the neighborhood structure.

Points that were close before the MLP may become far apart. Points that were far apart may become close if the model learns to ignore irrelevant directions.

An MLP with one hidden layer can be written as:

```math
f(x) = W_2 \sigma(W_1x + b_1) + b_2
```

where `sigma` is a nonlinearity such as ReLU. Without `sigma`, the two linear maps collapse into one:

```math
W_2(W_1x + b_1) + b_2 = (W_2W_1)x + (W_2b_1 + b_2)
```

So depth without nonlinearity is still linear. Nonlinearity is what lets the model use different linear behavior in different regions of the input space.

## Practical examples

### Projecting 2D points to 1D

A projection from 2D to 1D computes:

```math
y = w^\top x + b
```

Many different 2D points can share the same 1D value. If two points differ only in a direction orthogonal to `w`, the projection cannot distinguish them.

```python
w = torch.tensor([1.0, 0.0])
points = torch.tensor([[2.0, -3.0], [2.0, 4.0]])
print(points @ w)  # tensor([2., 2.])
```

This is useful when the discarded direction is irrelevant and harmful when it carries task information.

### Expanding 2D to 8D

An expansion can create eight learned views of the same 2D input:

```python
expand = nn.Linear(2, 8)
z = expand(torch.randn(100, 2))
print(torch.linalg.matrix_rank(z - z.mean(dim=0)))  # at most 2 for generic linear data
```

Before a nonlinearity, the intrinsic dimension is still limited by the original data and the rank of the map. Expansion gives later gates more coordinates to work with; it does not create new independent information by itself.

### ReLU turning half-spaces off

For `h = ReLU(w^T x + b)`, one side of the line is zero. This lets a model ignore a learned feature for some inputs while preserving it for others.

### Separating patterns a linear layer cannot

XOR-like points are not separable by one straight line in the original 2D space. A small ReLU MLP can map them into hidden activations where the classes become separable.

```python
mlp = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)
logits = mlp(points)
```

The first layer creates learned gates; ReLU chooses active regions; the final layer recombines those region-specific features.

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

In transformer blocks, residuals and normalization interact:

```math
y = \operatorname{LayerNorm}(x + f(x))
```

The residual path preserves access to the original representation, while `f(x)` proposes an update. LayerNorm then rescales the combined representation so later layers see a more stable distribution.

## Common failure modes

- Treating a dimension increase as extra information rather than extra coordinates.
- Compressing embeddings before checking whether the discarded directions carry task signal.
- Forgetting that low-rank matrices bottleneck information even with large output dimension.
- Using ReLU without noticing that negative evidence is clamped to zero.
- Interpreting hidden coordinates directly without accounting for rotations and basis changes.
- Comparing vector norms across layers that use normalization differently.

## Visual idea

Draw four panels: original 2D points, a 1D projection that collapses two points, an 8D expansion shown as multiple learned projection axes, and a ReLU gate that turns one half-plane off. For an MLP, draw several half-plane gates whose active regions are recombined by the final layer.

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

Also try these controlled variations:

1. Project 2D points to 1D and find pairs that collide.
2. Expand 2D points to 8D with a linear layer and verify that rank is still at most 2 before nonlinearities.
3. Plot `ReLU(w^T x + b)` over a 2D grid to see one half-space turn off.
4. Train a linear classifier and a one-hidden-layer ReLU MLP on XOR points and compare decision boundaries.
