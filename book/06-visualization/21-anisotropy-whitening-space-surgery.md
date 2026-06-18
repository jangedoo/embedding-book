# Anisotropy, Whitening, and Space Surgery

Embedding spaces are often anisotropic: vectors occupy a narrow cone, have a large shared mean, or are dominated by a few directions. In such spaces, many vectors look more similar to each other than expected, and nearest-neighbor search may return generic or frequent examples.

Centering, component removal, and whitening are ways to modify the geometry after training. They can help retrieval and clustering, but they are space surgery, not free improvements.

## Intuition

If every sentence embedding contains a strong "common sentence" direction, cosine similarity spends part of its budget comparing that shared direction. Removing the common direction can make topic-specific differences easier to see.

But not every dominant direction is bad. A top component might encode language, domain, quality, or another signal the application needs. Removing it may improve one benchmark while hurting the product.

## Mathematical object

For embeddings:

```math
X \in \mathbb{R}^{n \times d}
```

the mean vector is:

```math
\mu = \frac{1}{n}\sum_{i=1}^{n} x_i
```

Centering subtracts it:

```math
x_i' = x_i - \mu
```

If `u_1, ..., u_k` are top principal directions, component removal projects them out:

```math
x_i' = x_i - \sum_{j=1}^{k}(x_i^\top u_j)u_j
```

Whitening transforms centered vectors so covariance is closer to identity:

```math
z = \Lambda^{-1/2} U^\top (x - \mu)
```

where `U` contains covariance eigenvectors and `Lambda` contains eigenvalues.

## PyTorch equivalent

```python
import torch
import torch.nn.functional as F

X = torch.randn(5000, 384)

mu = X.mean(dim=0, keepdim=True)
Xc = X - mu

U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
top = Vh[:3]

X_removed = Xc - (Xc @ top.T) @ top
X_removed = F.normalize(X_removed, dim=-1)
```

A simple diagonal whitening version rescales each centered coordinate:

```python
std = Xc.std(dim=0, keepdim=True).clamp_min(1e-6)
X_diag_white = Xc / std
```

Full whitening can be unstable when the covariance estimate is noisy or dimensions are nearly redundant.

## What this means in ML systems

Space surgery is usually applied as a post-processing step:

1. Fit statistics on a representative calibration set.
2. Transform all stored document vectors.
3. Transform every query vector with the same statistics.
4. Re-normalize if serving with cosine similarity.
5. Rebuild or refresh the ANN index.

This must be treated like a model change. It can affect recall, latency, cache keys, vector database contents, and backward compatibility with older embeddings.

## Common failure modes

- Fitting mean and principal components on evaluation data.
- Transforming documents but not queries, or queries but not documents.
- Removing a component that encodes a useful business signal.
- Over-whitening noisy small datasets and amplifying unstable directions.
- Comparing pre-surgery and post-surgery vectors in the same index.
- Improving average retrieval while damaging rare languages or minority domains.

## Visual idea

Draw an elongated point cloud in 2D. Show three panels: raw cloud with a large mean offset, centered cloud around the origin, and whitened cloud with roughly circular covariance. Add a warning label that real high-dimensional whitening can amplify noise.

## Small experiment

Take a sentence embedding dataset with labels. Measure top-10 neighbor label purity before and after centering, after removing the top one component, and after removing the top five components. Plot both quality and nearest-neighbor overlap. The overlap shows how much the retrieval behavior changed, even when metrics improve.

## Practical takeaways

Centering and whitening can make useful differences more visible, especially for cosine search over anisotropic sentence embeddings.

They should be validated as deployment changes: fit on calibration data, apply symmetrically, rebuild indexes, and check subgroup metrics before deciding they helped.
