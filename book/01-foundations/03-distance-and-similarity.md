# Distance and Similarity

Distance and similarity functions decide which vectors count as neighbors, matches, recommendations, duplicates, or outliers. The metric is not a detail added after training. It is part of the model's behavior.

Similarity is not one thing. It is a modeling choice.

The same embeddings can produce different nearest neighbors depending on whether we use cosine similarity, dot product, or Euclidean distance.

## Summary

Dot product rewards alignment and length. Cosine similarity keeps direction and removes length. Euclidean distance measures physical separation in the coordinate space. Manhattan distance sums coordinate differences. Mahalanobis distance rescales the space by covariance or an equivalent learned metric. These choices are interchangeable only under specific conditions, especially normalization.

## Intuition

Imagine a query vector and two document vectors:

- one document points in almost the same direction but has a small norm
- another points in a less aligned direction but has a very large norm

Cosine similarity prefers the first document because direction dominates. Dot product may prefer the second because length contributes to the score. Euclidean distance may prefer whichever vector is physically closer in the full coordinate space.

This is why retrieval bugs often look like "the vectors are good, but the ranking is strange." The ranking is the geometry plus the metric.

The previous chapter defined norms and angles. This chapter turns those primitives into ranking rules.

## Dot product

```math
x \cdot y = \|x\| \|y\| \cos(\theta)
```

Dot product rewards both alignment and length.

Practical interpretation:

- useful when vector norm carries signal
- common in recommender systems
- can mix relevance with popularity or confidence

In PyTorch:

```python
import torch

scores = query @ items.T
ranking = scores.argsort(descending=True)
```

In item recommendation, a user vector `u` and item vector `v_i` are often scored by:

```math
s_i = u^\top v_i
```

Large item norms can make popular or broadly appealing items rank highly. That can be desirable when norm reflects confidence or exposure, but harmful when it creates popularity bias.

Dot product is also common because it is fast. A query matrix `Q \in \mathbb{R}^{B \times d}` and candidate matrix `X \in \mathbb{R}^{N \times d}` produce all exact scores as:

```math
S = QX^\top \in \mathbb{R}^{B \times N}
```

That same matrix multiply is easy to batch on GPUs and easy for vector databases to approximate with maximum inner product search.

## Cosine similarity

```math
\cos(x, y) = \frac{x \cdot y}{\|x\|\|y\|}
```

Cosine compares direction and ignores length.

```python
query_n = torch.nn.functional.normalize(query, dim=-1)
items_n = torch.nn.functional.normalize(items, dim=-1)
scores = query_n @ items_n.T
```

Cosine is common for sentence embedding retrieval because sentence vector length is often less interpretable than direction. If the embedding model was trained with contrastive cosine-like objectives, cosine is usually the first metric to try.

For cosine distance, many libraries use:

```math
d_{\cos}(x, y) = 1 - \cos(x, y)
```

This is useful for APIs that expect a distance, where smaller means closer.

## Euclidean distance

```math
\|x-y\|_2
```

Euclidean distance uses both direction and length.

For ranking, squared Euclidean distance is often equivalent and cheaper because it avoids the square root:

```math
\|x-y\|_2^2 = \sum_i (x_i - y_i)^2
```

```python
dist = torch.cdist(query[None, :], items, p=2)
ranking = dist.squeeze(0).argsort()
```

If you only need ranking, square after `cdist` or compute squared distances directly:

```python
dist2 = ((items - query) ** 2).sum(dim=-1)
ranking = dist2.argsort()
```

For a fixed query `q`, squared Euclidean distance expands to:

```math
\|q-x\|_2^2 = \|q\|_2^2 + \|x\|_2^2 - 2q^\top x
```

The query norm is constant across candidates, so Euclidean ranking depends on candidate norm and dot product. This is the key difference from pure dot-product ranking.

If vectors are normalized:

```math
\|x-y\|^2 = 2 - 2\cos(x, y)
```

So for normalized vectors, cosine similarity and Euclidean distance give the same ranking, just reversed and transformed.

Without normalization, Euclidean distance can strongly care about norm differences.

The equivalence depends on both sides being normalized. Normalizing only queries or only candidates does not make Euclidean and cosine rankings equivalent.

## Manhattan distance

Manhattan distance uses absolute coordinate differences:

```math
\|x-y\|_1 = \sum_i |x_i-y_i|
```

It is less dominated by one very large coordinate difference than squared Euclidean distance, but it still depends on the coordinate system. In dense neural embeddings it is less common than dot product, cosine, or Euclidean distance, but it can be useful for sparse features or systems where additive coordinate differences have a meaningful interpretation.

```python
dist1 = torch.cdist(query[None, :], items, p=1)
```

## Mahalanobis distance

Mahalanobis distance rescales distance by a covariance matrix:

```math
d_M(x, y) = \sqrt{(x-y)^\top \Sigma^{-1}(x-y)}
```

If some directions naturally vary more than others, Mahalanobis distance can avoid over-penalizing variation along high-variance directions and can emphasize low-variance directions.

In practice, using a full covariance matrix is expensive and statistically fragile in high dimensions. Common approximations include diagonal scaling, whitening, or learning a projection before nearest-neighbor search.

A diagonal version is often enough to express the idea:

```math
d_D(x, y)^2 = \sum_i \frac{(x_i - y_i)^2}{\sigma_i^2}
```

where coordinates with large natural variance receive less weight. In neural systems, a learned linear projection followed by Euclidean or cosine search often plays a similar role.

```python
delta = items - query
inv_var = 1.0 / (items.var(dim=0, unbiased=False) + 1e-6)
dist2_diag = (delta.square() * inv_var).sum(dim=-1)
ranking = dist2_diag.argsort()
```

## Normalized versus unnormalized vectors

Normalization maps every nonzero vector onto the unit sphere:

```math
\hat{x} = \frac{x}{\|x\|_2}
```

After normalization:

- all vectors have length 1
- dot product equals cosine similarity
- Euclidean distance and cosine give the same ranking
- vector length can no longer influence retrieval

Before normalization:

- dot product rewards alignment and length
- Euclidean distance penalizes both angular mismatch and norm mismatch
- candidate norm can change rankings even if angle is similar

This choice should match the training objective. If the model learned useful confidence in the norm, normalizing may throw away signal. If norm mostly reflects frequency or artifacts, normalization may improve retrieval.

The fastest sanity check is to compare top-k overlap across metrics. If cosine and dot product disagree sharply, vector length is affecting ranking. [What Does Vector Length Mean?](04-vector-length.md) gives a more detailed diagnostic.

## Implementation sketch

```python
import torch
import torch.nn.functional as F

query = torch.randn(768)
items = torch.randn(50_000, 768)

dot_scores = items @ query
cos_scores = F.normalize(items, dim=-1) @ F.normalize(query, dim=0)
euclidean_dist2 = ((items - query) ** 2).sum(dim=-1)
manhattan_dist = (items - query).abs().sum(dim=-1)

dot_top = dot_scores.topk(10).indices
cos_top = cos_scores.topk(10).indices
euclidean_top = euclidean_dist2.topk(10, largest=False).indices
manhattan_top = manhattan_dist.topk(10, largest=False).indices
```

Use this exact-search version as a baseline before configuring an ANN index. Approximate search should be measured against the metric you actually intend to serve.

## DBSCAN with cosine versus Euclidean

DBSCAN groups points using a radius `eps` and a minimum-neighbor count. The metric changes what "inside the radius" means.

With Euclidean distance on raw embeddings, dense regions can form around vectors with similar magnitude. Long vectors may be far from short vectors even when they point in the same semantic direction.

With cosine distance:

```math
d_{\cos}(x, y) = 1 - \cos(x, y)
```

clusters form by direction. This is often better for sentence embeddings where semantic topic is mostly angular.

The practical trap is `eps`: values are not comparable across metrics. An `eps` that works for Euclidean distance says nothing about a good `eps` for cosine distance.

A practical workflow is to sample pairwise distances, plot their distribution, and choose candidate `eps` values from the scale of that metric. Then validate clusters with original examples, not just cluster counts.

## Sentence embedding retrieval

For many sentence embedding models, the standard recipe is:

```python
docs = torch.nn.functional.normalize(docs, dim=-1)
query = torch.nn.functional.normalize(query, dim=-1)
scores = query @ docs.T
topk = scores.topk(k=10)
```

This is cosine search implemented as an inner product over normalized vectors. Many vector databases use exactly this trick because maximum inner product search can serve cosine search after normalization.

Failure mode: if queries are normalized but stored documents are not, ranking becomes a hybrid of cosine and document norm. That bug can be hard to notice because results may still look plausible.

In a production RAG system, document vectors are usually normalized before being written to the index, and query vectors are normalized with the same code path before search. Store this convention next to the index metadata so a later rebuild does not silently change the metric.

## Item recommendation with dot product

Recommendation models often use:

```math
score(user, item) = u^\top v_i
```

Here, vector length can be useful. A high-norm user vector may represent strong preferences. A high-norm item vector may represent broad appeal or high confidence from many interactions.

But dot product can also amplify exposure bias: frequently observed items receive more training updates and may develop larger norms. Ranking then mixes preference with popularity. Evaluation should check both relevance metrics and catalog diversity.

If popularity is useful but too strong, common fixes include norm clipping, regularization, calibrated popularity features outside the embedding score, or reranking constraints that preserve relevance while improving coverage.

## Common failure modes

- Training with one metric and serving with another.
- Normalizing only queries or only documents.
- Using Euclidean distance on raw vectors when norm is an artifact.
- Using cosine similarity when norm encodes confidence.
- Comparing DBSCAN `eps` values across metrics.
- Forgetting that squared Euclidean distance and Euclidean distance produce the same ranking, but not the same numeric threshold.
- Building an ANN index with one metric and querying it as if it used another.
- Treating a single similarity threshold as portable across embedding models or dimensions.

## Visual idea

Draw one query vector and three candidates. Show that a same-direction short vector wins under cosine, a longer moderately aligned vector can win under dot product, and a nearby vector can win under Euclidean distance.

## Small experiment

Create five 2D vectors with controlled directions and lengths. Rank them against the same query using dot product, cosine similarity, Euclidean distance, squared Euclidean distance, and Manhattan distance. Then normalize all vectors and show that cosine, dot product, and Euclidean rankings become tied by a monotonic transformation.

```python
import torch
import torch.nn.functional as F

q = torch.tensor([1.0, 0.0])
X = torch.tensor([
    [0.9, 0.0],   # aligned, short
    [3.0, 0.7],   # longer, less aligned
    [0.7, 0.7],   # 45 degrees
    [-1.0, 0.0],  # opposite
    [0.2, -0.1],  # nearby but small
])

def order(values, largest=True):
    return values.argsort(descending=largest).tolist()

dot = X @ q
cos = F.cosine_similarity(X, q[None, :], dim=-1)
dist2 = ((X - q) ** 2).sum(dim=-1)
l1 = (X - q).abs().sum(dim=-1)

print("dot", order(dot))
print("cos", order(cos))
print("euclidean", order(dist2, largest=False))
print("manhattan", order(l1, largest=False))

Xn = F.normalize(X, dim=-1)
qn = F.normalize(q, dim=0)
print("normalized dot", order(Xn @ qn))
print("normalized euclidean", order(((Xn - qn) ** 2).sum(dim=-1), largest=False))
```

The normalized dot and normalized Euclidean rankings match because both are monotonic functions of cosine.

## Practical takeaways

Before choosing a metric, ask:

1. Does vector length contain useful signal?
2. Was the model trained with this metric?
3. Does the ANN index support this metric?
4. Do I normalize before storage, before query, both, or neither?

The metric is part of the model. Treat it as a design decision, test it directly, and document it in the retrieval pipeline.
