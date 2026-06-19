# High-Dimensional Geometry

Embeddings usually live in hundreds or thousands of dimensions. In those spaces, intuition from 2D scatter plots is useful but incomplete. Random directions become nearly orthogonal, distances concentrate, and nearest-neighbor behavior can become surprising.

High-dimensional spaces behave differently from 2D intuition.

## Summary

In high dimensions, random vectors are almost orthogonal, distance distributions can concentrate, some points become repeated nearest-neighbor hubs, and embedding spaces often occupy anisotropic cones rather than the whole sphere. These effects change how thresholds, clusters, visualizations, and ANN recall should be interpreted.

## Intuition

A 2D plot lets us see all directions. A 768-dimensional embedding space has far more room. Most random directions are neither clearly similar nor clearly opposite; they are close to perpendicular. This makes apparently small cosine differences meaningful, but it also means that thresholds and clusters need calibration from real distributions.

The metric choices from [Distance and Similarity](03-distance-and-similarity.md) still apply, but high dimension changes the scale and reliability of those metrics.

## Random vectors become nearly orthogonal

As dimension grows, random vectors tend to have cosine similarity near zero.

For independent random unit vectors `x` and `y` in `d` dimensions:

```{math}
\mathbb{E}[x^\top y] = 0
```

and the typical size of the dot product shrinks roughly like:

```{math}
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

For unit random vectors, the standard deviation of `x^\top y` is approximately:

```{math}
\frac{1}{\sqrt{d}}
```

So in 768 dimensions, many random cosine similarities fall near zero with a scale of about `0.036`. A cosine score of `0.2` can therefore be far from random even though it is not visually "close" in 2D intuition.

## Distance concentration

Distances can become less distinguishable in high dimensions. This makes clustering and nearest-neighbor search harder.

If all points are random and similarly distributed, nearest and farthest distances may be closer than expected. A retrieval system then needs learned structure, normalization, filtering, or reranking to make neighborhoods meaningful.

For normalized vectors, squared Euclidean distance is tied to cosine:

```{math}
\|x-y\|_2^2 = 2 - 2x^\top y
```

As random cosines concentrate near zero, random distances concentrate near `\sqrt{2}`. Useful embedding spaces must create non-random local structure on top of that background.

## Hubness

Some points become nearest neighbors of many other points. These are hubs.

Hubness is common in high-dimensional nearest-neighbor search. A hub can be a genuinely central concept, but it can also be an artifact of anisotropy, frequency, or vector norms.

In retrieval, hubs show up as documents that appear in many unrelated top-k lists. In recommendation, they show up as generic items that get recommended too broadly.

Hubs can come from useful centrality, but they can also come from high norms, shared boilerplate, duplicated content, anisotropy, or training frequency. Measure how often each candidate appears in top-k lists across a query sample.

## Anisotropy

An embedding space is anisotropic when vectors occupy a narrow cone or share dominant directions instead of spreading evenly around the sphere.

One symptom is that unrelated vectors still have high cosine similarity because many vectors share a common component. Centering, removing top principal components, or whitening can sometimes help. These operations can also remove useful signal, so they should be evaluated on the downstream task.

Mathematically, a simple diagnostic is the mean vector:

```{math}
\mu = \frac{1}{N}\sum_{i=1}^N x_i
```

If many normalized embeddings have a large projection onto `\mu`, the space has a shared direction. Principal component analysis gives a stronger diagnostic by showing whether a few directions explain a large fraction of variance.

## PyTorch and NumPy equivalent

```python
import torch
import torch.nn.functional as F

X = F.normalize(torch.randn(2_000, 768), dim=-1)
scores = X @ X.T
scores.fill_diagonal_(-float("inf"))

nearest = scores.max(dim=1).values
hub_counts = scores.topk(k=10, dim=1).indices.flatten().bincount(minlength=X.size(0))

print(nearest.mean(), nearest.std())
print(hub_counts.float().quantile(torch.tensor([0.5, 0.9, 0.99])))
```

This does not prove your production embeddings are healthy, but it gives a baseline for what random normalized vectors look like.

## Practical interpretation

High-dimensional geometry affects:

- how meaningful a similarity threshold is
- whether clustering produces stable groups
- how many neighbors need reranking
- whether approximate nearest-neighbor indexes preserve recall
- whether visualization tells a faithful story

The practical response is measurement. Inspect similarity distributions, norm distributions, nearest-neighbor overlap across metrics, and repeated appearances in top-k results.

For RAG systems, this means a cosine threshold should be tuned per embedding model and corpus. For clustering, it means `eps`, `k`, and linkage choices should come from original-space distances rather than a 2D projection. For ANN indexes, it means recall should be measured against exact top-k under the intended metric.

## Common failure modes

- Assuming a cosine threshold like `0.8` has the same meaning across models.
- Picking DBSCAN `eps` from a 2D projection.
- Trusting t-SNE or UMAP clusters without nearest-neighbor checks in the original space.
- Ignoring hubs because individual examples look reasonable.
- Removing principal components without checking retrieval quality afterward.
- Choosing `k` or `eps` from a small demo corpus and reusing it after the corpus changes.
- Trusting average similarity alone while missing long-tail hub behavior.

## Visual idea

Show a histogram of cosine similarities for random vectors in 32, 128, and 768 dimensions. The histogram narrows around zero as dimension increases. Next to it, show a bar chart of top-k neighbor counts where a few points become hubs after adding a shared bias direction.

## Small experiment

Generate random normalized vectors at several dimensions. Plot the distribution of pairwise cosine similarities and the gap between each point's nearest and farthest neighbor. Then add a shared bias direction to all vectors and observe how anisotropy changes the distribution.

```python
import torch
import torch.nn.functional as F

def cosine_sample(n=1_000, d=128):
    X = F.normalize(torch.randn(n, d), dim=-1)
    Y = F.normalize(torch.randn(n, d), dim=-1)
    return (X * Y).sum(dim=-1)

for d in [32, 128, 768]:
    cos = cosine_sample(d=d)
    print(d, cos.mean().item(), cos.std().item())

X = F.normalize(torch.randn(2_000, 128), dim=-1)
bias = F.normalize(torch.randn(128), dim=0)
X_biased = F.normalize(X + 0.6 * bias, dim=-1)

print("random mean cosine", (X[:500] @ X[500:1000].T).mean().item())
print("biased mean cosine", (X_biased[:500] @ X_biased[500:1000].T).mean().item())
```

The biased vectors share a common component, so unrelated pairs can look more similar. This is the intuition behind anisotropy diagnostics and some whitening or centering methods.

## Practical implications

- DBSCAN `eps` is difficult to choose in high dimensions.
- UMAP/t-SNE plots can be visually persuasive but misleading.
- Nearest-neighbor inspection is often more useful than only plotting.
- Centering and whitening can sometimes help, but may remove useful signal.
- Similarity thresholds should be calibrated on labeled pairs or retrieval metrics.
- Hub documents and generic recommendations should be monitored as first-class quality issues.

## Practical takeaways

High-dimensional embedding spaces need empirical calibration. Look at distributions, hubs, metric agreement, and task metrics before trusting thresholds or visual clusters.

Use visualizations to generate hypotheses, then verify them in the original embedding space.
