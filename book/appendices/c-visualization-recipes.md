# Visualization Recipes

This appendix gives small visualization recipes for embedding work. The snippets are starting points for debugging, not evidence by themselves. Always pair plots with nearest-neighbor inspection and quantitative metrics.

## Vector arrows in 2D

```python
import matplotlib.pyplot as plt
import numpy as np

vectors = np.array([[1.0, 0.2], [0.4, 0.9], [-0.6, 0.5]])
labels = ["query", "doc A", "doc B"]

fig, ax = plt.subplots(figsize=(4, 4))
for v, label in zip(vectors, labels):
    ax.arrow(0, 0, v[0], v[1], head_width=0.04, length_includes_head=True)
    ax.text(v[0] * 1.08, v[1] * 1.08, label)

ax.axhline(0, color="0.8", linewidth=1)
ax.axvline(0, color="0.8", linewidth=1)
ax.set_aspect("equal")
ax.set_title("Vectors as directions and lengths")
plt.show()
```

Use this to show why dot product depends on both angle and norm.

## Unit circle for cosine similarity

```python
import matplotlib.pyplot as plt
import numpy as np

theta = np.linspace(0, 2 * np.pi, 300)
circle = np.stack([np.cos(theta), np.sin(theta)], axis=1)

points = np.array([[2.0, 0.5], [0.7, 1.4], [-1.0, 0.3]])
points_n = points / np.linalg.norm(points, axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(circle[:, 0], circle[:, 1], color="0.7")
ax.scatter(points[:, 0], points[:, 1], label="raw")
ax.scatter(points_n[:, 0], points_n[:, 1], label="normalized")
for raw, normed in zip(points, points_n):
    ax.plot([raw[0], normed[0]], [raw[1], normed[1]], color="0.8", linestyle="--")

ax.set_aspect("equal")
ax.legend()
plt.show()
```

This makes normalization concrete: all nonzero vectors move to the unit circle.

## PCA embedding cloud

```python
import matplotlib.pyplot as plt
import torch

X = torch.randn(1000, 128)
labels = torch.randint(0, 4, (1000,))

Xc = X - X.mean(dim=0, keepdim=True)
_, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
Z = Xc @ Vh[:2].T

plt.figure(figsize=(5, 4))
plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=8, cmap="tab10", alpha=0.7)
plt.title("PCA projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
```

PCA is deterministic and good for seeing dominant linear variation. It may ignore lower-variance directions that matter for retrieval.

## Nearest-neighbor edges

```python
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

X = F.normalize(torch.randn(200, 64), dim=-1)
Z = torch.randn(200, 2)

scores = X @ X.T
scores.fill_diagonal_(-float("inf"))
nn = scores.argmax(dim=1)

plt.figure(figsize=(5, 5))
plt.scatter(Z[:, 0], Z[:, 1], s=12)
for i in range(40):
    j = nn[i].item()
    plt.plot([Z[i, 0], Z[j, 0]], [Z[i, 1], Z[j, 1]], color="0.8", linewidth=0.8)
plt.title("Projected points with original-space nearest-neighbor edges")
plt.show()
```

Edges remind readers that the projection is not the original geometry.

## Singular value spectrum

```python
import matplotlib.pyplot as plt
import torch

X = torch.randn(5000, 384)
Xc = X - X.mean(dim=0, keepdim=True)
S = torch.linalg.svdvals(Xc)

plt.figure(figsize=(5, 3))
plt.plot(S.numpy())
plt.yscale("log")
plt.title("Singular value spectrum")
plt.xlabel("component")
plt.ylabel("singular value")
plt.show()
```

A steep spectrum suggests a few directions dominate variance. That can motivate centering or component-removal experiments.

## Retrieval ranking diagram

```python
import matplotlib.pyplot as plt

docs = ["A", "B", "C", "D", "E"]
scores = [0.82, 0.77, 0.65, 0.61, 0.54]
relevant = [False, True, False, True, False]

colors = ["tab:green" if r else "0.7" for r in relevant]

fig, ax = plt.subplots(figsize=(5, 2.5))
ax.barh(docs[::-1], scores[::-1], color=colors[::-1])
ax.set_xlim(0, 1)
ax.set_xlabel("similarity score")
ax.set_title("Retrieved list")
plt.show()
```

Use this for explaining recall@k, MRR, reranking cutoffs, and near-miss retrieval errors.

## Plot checklist

Before using a visualization in analysis, record:

- embedding model and version
- preprocessing and normalization
- distance metric used in production
- sample size and sampling method
- projection method and random seed
- labels or metadata used only for coloring
- nearest-neighbor examples from the original space

The checklist prevents the plot from becoming detached from the system it is supposed to explain.
