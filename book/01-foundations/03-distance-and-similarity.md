# Distance and Similarity

Distance and similarity functions decide which vectors count as neighbors, matches, recommendations, duplicates, or outliers. The metric is not a detail added after training. It is part of the model's behavior.

Similarity is not one thing. It is a modeling choice.

The same embeddings can produce different nearest neighbors depending on whether we use cosine similarity, dot product, or Euclidean distance.

## Intuition

Imagine a query vector and two document vectors:

- one document points in almost the same direction but has a small norm
- another points in a less aligned direction but has a very large norm

Cosine similarity prefers the first document because direction dominates. Dot product may prefer the second because length contributes to the score. Euclidean distance may prefer whichever vector is physically closer in the full coordinate space.

This is why retrieval bugs often look like "the vectors are good, but the ranking is strange." The ranking is the geometry plus the metric.

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
dist2 = torch.cdist(query[None, :], items, p=2).square()
ranking = dist2.squeeze(0).argsort()
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

## DBSCAN with cosine versus Euclidean

DBSCAN groups points using a radius `eps` and a minimum-neighbor count. The metric changes what "inside the radius" means.

With Euclidean distance on raw embeddings, dense regions can form around vectors with similar magnitude. Long vectors may be far from short vectors even when they point in the same semantic direction.

With cosine distance:

```math
d_{\cos}(x, y) = 1 - \cos(x, y)
```

clusters form by direction. This is often better for sentence embeddings where semantic topic is mostly angular.

The practical trap is `eps`: values are not comparable across metrics. An `eps` that works for Euclidean distance says nothing about a good `eps` for cosine distance.

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

## Item recommendation with dot product

Recommendation models often use:

```math
score(user, item) = u^\top v_i
```

Here, vector length can be useful. A high-norm user vector may represent strong preferences. A high-norm item vector may represent broad appeal or high confidence from many interactions.

But dot product can also amplify exposure bias: frequently observed items receive more training updates and may develop larger norms. Ranking then mixes preference with popularity. Evaluation should check both relevance metrics and catalog diversity.

## Common failure modes

- Training with one metric and serving with another.
- Normalizing only queries or only documents.
- Using Euclidean distance on raw vectors when norm is an artifact.
- Using cosine similarity when norm encodes confidence.
- Comparing DBSCAN `eps` values across metrics.
- Forgetting that squared Euclidean distance and Euclidean distance produce the same ranking, but not the same numeric threshold.

## Visual idea

Draw one query vector and three candidates. Show that a same-direction short vector wins under cosine, a longer moderately aligned vector can win under dot product, and a nearby vector can win under Euclidean distance.

## Small experiment

Create five 2D vectors with controlled directions and lengths. Rank them against the same query using dot product, cosine similarity, Euclidean distance, squared Euclidean distance, and Manhattan distance. Then normalize all vectors and show that cosine, dot product, and Euclidean rankings become tied by a monotonic transformation.

## Practical takeaways

Before choosing a metric, ask:

1. Does vector length contain useful signal?
2. Was the model trained with this metric?
3. Does the ANN index support this metric?
4. Do I normalize before storage, before query, both, or neither?

The metric is part of the model. Treat it as a design decision, test it directly, and document it in the retrieval pipeline.
