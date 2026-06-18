# High-Dimensional Geometry

Embeddings usually live in hundreds or thousands of dimensions. In those spaces, intuition from 2D scatter plots is useful but incomplete. Random directions become nearly orthogonal, distances concentrate, and nearest-neighbor behavior can become surprising.

High-dimensional spaces behave differently from 2D intuition.

## Random vectors become nearly orthogonal

As dimension grows, random vectors tend to have cosine similarity near zero.

For independent random unit vectors `x` and `y` in `d` dimensions:

```math
\mathbb{E}[x^\top y] = 0
```

and the typical size of the dot product shrinks roughly like:

```math
O\left(\frac{1}{\sqrt{d}}\right)
```

This means a cosine similarity of `0.2` may be meaningful in 768 dimensions even though it sounds small.

```python
import torch
import torch.nn.functional as F

x = F.normalize(torch.randn(10_000, 768), dim=-1)
y = F.normalize(torch.randn(10_000, 768), dim=-1)
cos = (x * y).sum(dim=-1)
print(cos.mean(), cos.std())
```

## Distance concentration

Distances can become less distinguishable in high dimensions. This makes clustering and nearest-neighbor search harder.

If all points are random and similarly distributed, nearest and farthest distances may be closer than expected. A retrieval system then needs learned structure, normalization, filtering, or reranking to make neighborhoods meaningful.

## Hubness

Some points become nearest neighbors of many other points. These are hubs.

Hubness is common in high-dimensional nearest-neighbor search. A hub can be a genuinely central concept, but it can also be an artifact of anisotropy, frequency, or vector norms.

In retrieval, hubs show up as documents that appear in many unrelated top-k lists. In recommendation, they show up as generic items that get recommended too broadly.

## Anisotropy

An embedding space is anisotropic when vectors occupy a narrow cone or share dominant directions instead of spreading evenly around the sphere.

One symptom is that unrelated vectors still have high cosine similarity because many vectors share a common component. Centering, removing top principal components, or whitening can sometimes help. These operations can also remove useful signal, so they should be evaluated on the downstream task.

## Practical interpretation

High-dimensional geometry affects:

- how meaningful a similarity threshold is
- whether clustering produces stable groups
- how many neighbors need reranking
- whether approximate nearest-neighbor indexes preserve recall
- whether visualization tells a faithful story

The practical response is measurement. Inspect similarity distributions, norm distributions, nearest-neighbor overlap across metrics, and repeated appearances in top-k results.

## Common failure modes

- Assuming a cosine threshold like `0.8` has the same meaning across models.
- Picking DBSCAN `eps` from a 2D projection.
- Trusting t-SNE or UMAP clusters without nearest-neighbor checks in the original space.
- Ignoring hubs because individual examples look reasonable.
- Removing principal components without checking retrieval quality afterward.

## Visual idea

Show a histogram of cosine similarities for random vectors in 32, 128, and 768 dimensions. The histogram narrows around zero as dimension increases.

## Small experiment

Generate random normalized vectors at several dimensions. Plot the distribution of pairwise cosine similarities and the gap between each point's nearest and farthest neighbor. Then add a shared bias direction to all vectors and observe how anisotropy changes the distribution.

## Practical implications

- DBSCAN `eps` is difficult to choose in high dimensions.
- UMAP/t-SNE plots can be visually persuasive but misleading.
- Nearest-neighbor inspection is often more useful than only plotting.
- Centering and whitening can sometimes help, but may remove useful signal.

## Practical takeaways

High-dimensional embedding spaces need empirical calibration. Look at distributions, hubs, metric agreement, and task metrics before trusting thresholds or visual clusters.
