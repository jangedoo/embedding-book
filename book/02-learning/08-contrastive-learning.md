# Contrastive Learning and Metric Learning

Contrastive learning trains an embedding space by comparison. Instead of asking only "what class is this?" it asks "which example should be closer to this anchor than the others?" This makes it especially important for retrieval, recommendation, deduplication, clustering, and multimodal matching, where ranking neighbors matters more than predicting a fixed label.

## Summary

Contrastive learning shapes an embedding space by making positives outrank negatives under a chosen similarity function. The loss is simple, but the behavior depends heavily on pair construction, normalization, temperature, batch composition, and the serving metric. This chapter introduces triplet and InfoNCE-style objectives, shows the PyTorch score-matrix pattern used by dual encoders, and explains the failure modes that matter in retrieval and recommendation systems.

## Intuition

Given a query and several documents, a retrieval model should assign the highest score to the relevant document. Given an image and several captions, a multimodal model should align the matching caption and separate mismatched captions. Given two augmented views of the same input, a representation model should keep them close while pushing away other examples.

Contrastive learning creates geometry directly:

- positives are pulled together
- negatives are pushed apart
- the scoring function defines what "together" means
- the sampling strategy defines what the model learns to distinguish

The hard part is not writing the loss. The hard part is choosing positives, negatives, temperature, normalization, and evaluation so the learned neighborhoods match the deployed system.

This is the ranking-oriented counterpart to the previous chapter. Prediction asks the embedding to help choose a label; contrastive learning asks it to win a competition against other candidates.

## Mathematical object

Let an encoder map an input to a vector:

```{math}
z = f_\theta(x) \in \mathbb{R}^{d}
```

For an anchor `a`, a positive `p`, and a negative `n`, we want:

```{math}
sim(f_\theta(a), f_\theta(p)) >
sim(f_\theta(a), f_\theta(n))
```

With normalized vectors, dot product equals cosine similarity:

```{math}
\hat{z} = \frac{z}{\|z\|_2}, \quad sim(\hat{x}, \hat{y}) = \hat{x}^\top \hat{y}
```

A margin-based triplet loss is:

```{math}
L = \max(0, m + sim(a, n) - sim(a, p))
```

where `m` is the required margin. The loss is zero when the positive is already ahead of the negative by at least `m`.

A common batch contrastive loss is InfoNCE. For each anchor `i`, assume its positive is `i` in another batch of views. With similarity scores:

```{math}
s_{ij} = \frac{sim(q_i, k_j)}{\tau}
```

the loss is:

```{math}
L_i = -\log
\frac{\exp(s_{ii})}
{\sum_{j=1}^{B} \exp(s_{ij})}
```

The temperature `\tau` controls how sharp the competition is. Smaller values make the model focus more strongly on the highest-scoring examples and produce larger gradients for close competitors.

The denominator is not a harmless implementation detail. It defines the candidate set that each positive must beat. With in-batch negatives, every other example in the batch becomes part of the training task.

## PyTorch equivalent

The basic in-batch contrastive pattern is concise:

```python
import torch
import torch.nn.functional as F

def contrastive_loss(query, doc, temperature=0.05):
    query = F.normalize(query, dim=-1)
    doc = F.normalize(doc, dim=-1)

    logits = query @ doc.T
    logits = logits / temperature

    labels = torch.arange(query.size(0), device=query.device)
    return F.cross_entropy(logits, labels)

B, d = 32, 128
query = torch.randn(B, d)
doc = torch.randn(B, d)
loss = contrastive_loss(query, doc)
```

This assumes pair `query[i]` and `doc[i]` is positive, while every `doc[j]` for `j != i` is a negative for `query[i]`.

A symmetric variant trains both directions:

```python
def symmetric_contrastive_loss(a, b, temperature=0.05):
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) +
        F.cross_entropy(logits.T, labels)
    )
```

This is the core shape behind many dual-encoder retrieval and image-text alignment systems.

For a margin-based triplet loss, the code makes the ranking constraint explicit:

```python
def triplet_cosine_loss(anchor, positive, negative, margin=0.2):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negative = F.normalize(negative, dim=-1)

    pos = (anchor * positive).sum(dim=-1)
    neg = (anchor * negative).sum(dim=-1)
    return F.relu(margin + neg - pos).mean()
```

Triplet losses are easy to reason about, but they depend strongly on which negatives are chosen. If most negatives are already far away, the loss is often zero and training stalls.

## What this means in ML systems

In dense retrieval, a query encoder and document encoder produce vectors:

```{math}
q = f_q(query), \quad d = f_d(document)
```

Serving ranks documents by:

```{math}
score(q, d) = q^\top d
```

or by cosine similarity if vectors are normalized. Contrastive training should match that serving score. If the model is trained with normalized cosine but served with unnormalized dot product, vector length can unexpectedly change ranking.

In RAG systems, contrastive learning affects which passages enter the context window. A small ranking error can become a generation error if the answer-bearing passage is not retrieved. This is why retrieval evaluation should measure recall at `k`, mean reciprocal rank, and downstream answer quality, not just training loss.

The basic retrieval check is small enough to write directly:

```python
def recall_at_k(query, doc, positive_doc_index, k):
    query = F.normalize(query, dim=-1)
    doc = F.normalize(doc, dim=-1)
    scores = query @ doc.T
    topk = scores.topk(k, dim=-1).indices
    hits = topk.eq(positive_doc_index[:, None]).any(dim=-1)
    return hits.float().mean()
```

This metric should be computed with the same normalization, candidate pool, and approximate index behavior used at serving time whenever possible.

In recommendation, contrastive objectives can learn user-item or session-item spaces where observed interactions are positives and sampled unobserved items are negatives. The negative sampler becomes part of the model. Sampling only random popular negatives teaches a different space than sampling hard near-miss items.

In multimodal systems, contrastive learning aligns different encoders into one comparison space. The geometry is useful because text and image vectors are trained to compete in the same score matrix.

## Positives, negatives, and hard examples

Positive pairs should represent the invariance you want. If two augmented images are positives, the model learns to ignore those augmentations. If a query and clicked document are positives, the model learns from click behavior, including its biases. If paraphrases are positives, the model learns semantic equivalence more directly.

Negatives define the distinctions the model must learn. Easy negatives produce low loss but weak retrieval behavior. Hard negatives, such as documents that look relevant but do not answer the query, teach finer boundaries.

There is a practical tradeoff:

- random negatives are cheap but often too easy
- hard negatives are informative but can include false negatives
- in-batch negatives are efficient but depend heavily on batch composition
- mined negatives improve ranking but can reinforce errors from the miner

Hard negatives are especially valuable when they represent realistic confusion. For a question answering retriever, "mentions the same entity but does not answer the question" is usually more useful than a random passage about an unrelated topic. But if the label says only one passage is positive when several passages answer the query, hard-negative mining can accidentally create false negatives.

## Common failure modes

- False negatives. A "negative" in the batch may actually be relevant, so the model is trained to push away a valid neighbor.
- Collapse. Poor objective design can make many embeddings identical or nearly identical, especially without enough negative pressure or variance constraints.
- Temperature instability. Too small a temperature can make training brittle; too large can make distinctions weak.
- Easy-negative saturation. Loss decreases while retrieval quality barely improves because negatives are trivial.
- Metric mismatch. Training uses cosine, serving uses raw dot product, or training uses cross-encoder labels while serving uses dual-encoder nearest neighbors.
- Batch artifacts. In-batch negatives are only as diverse and representative as the batch.
- Data leakage. Near-duplicate positives across train and test can inflate retrieval metrics.
- Over-mining. Extremely hard negatives may be mislabeled or ambiguous, causing the model to learn unstable boundaries.
- Poor distributed-batch handling. In multi-GPU training, using only local-batch negatives can make the effective negative set much smaller than expected.
- Ignoring index-time quantization. A model can look good with exact float32 search and lose recall after ANN compression or product quantization.

## Visual idea

```{image} ../../assets/figures/contrastive-ranking-forces.svg
:alt: A query embedding pulled toward a positive example and pushed away from negatives, with a ranked retrieval list changing order.
:align: center
:width: 100%
```

The figure turns the contrastive loss into forces on a retrieval ranking. The query is pulled toward its positive example and pushed away from negatives, especially negatives that currently score too highly. The geometry is not optimized for a pretty scatter plot; it is optimized so the positive item rises above competing candidates.

This framing also explains why hard negatives matter. Easy negatives already sit far from the query and contribute little useful signal. Hard negatives are close enough to confuse the model, so they create stronger gradients and teach the embedding space the fine distinctions needed by real retrieval systems.

## Small experiment

Create two clusters in 2D, sample positive pairs from the same cluster, and train a tiny linear encoder with in-batch contrastive loss. Compare nearest neighbors before and after training:

```python
import torch
import torch.nn.functional as F
from torch import nn

torch.manual_seed(0)

points = torch.cat([
    torch.randn(64, 2) * 0.3 + torch.tensor([1.0, 0.0]),
    torch.randn(64, 2) * 0.3 + torch.tensor([-1.0, 0.0]),
])

encoder = nn.Linear(2, 2)
opt = torch.optim.Adam(encoder.parameters(), lr=0.05)

for _ in range(200):
    idx = torch.randint(0, len(points), (32,))
    a = points[idx] + 0.05 * torch.randn(32, 2)
    b = points[idx] + 0.05 * torch.randn(32, 2)

    za = F.normalize(encoder(a), dim=-1)
    zb = F.normalize(encoder(b), dim=-1)
    logits = za @ zb.T / 0.1
    labels = torch.arange(32)
    loss = F.cross_entropy(logits, labels)

    opt.zero_grad()
    loss.backward()
    opt.step()
```

Plot the encoded points before and after training. Then repeat with deliberately wrong positives sampled from different clusters and observe how the space degrades.

For a retrieval-flavored version, keep a held-out set of anchors and positives, compute `recall_at_k` before and after training, and compare random negatives with hard negatives near the decision boundary.

## Practical takeaways

Contrastive learning is ranking pressure applied to vectors.

For a production retrieval system, document these choices:

1. What counts as a positive pair?
2. How are negatives sampled or mined?
3. Is the model trained with the same similarity metric used at serving time?
4. Are embeddings normalized before indexing?
5. Which retrieval metrics validate the learned space?

The loss function is only one part of the method. The pair construction and serving metric shape the embedding space just as strongly.
