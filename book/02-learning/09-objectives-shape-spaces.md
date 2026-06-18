# Objectives Shape Embedding Spaces

The same objects can form very different embedding spaces under different objectives. A classifier may organize vectors by class boundaries. A contrastive retriever may organize them by pairwise relevance. A language model may organize token representations around next-token prediction. An autoencoder may preserve reconstructable details that a classifier would ignore. The geometry is not an intrinsic property of the data alone; it is the result of the loss, data distribution, architecture, and metric.

## Summary

Embedding geometry is not an objective fact about the inputs. It is shaped by the loss, labels, negatives, model architecture, similarity metric, and evaluation loop. This chapter compares classification, retrieval, reconstruction, regression, language modeling, and multi-task objectives, then gives practical checks for deciding whether a learned space is appropriate for classification, clustering, recommendation, or RAG.

## Intuition

Imagine the same set of product descriptions. If the objective is category classification, running shoes should be near other running shoes. If the objective is price prediction, expensive running shoes may move closer to expensive jackets than to cheap shoes. If the objective is search relevance, products that answer the same query should become neighbors even if their categories differ.

The objects did not change. The question changed.

This is the central habit for working with embeddings: never ask only "are these embeddings good?" Ask "good for what objective, metric, and downstream decision?"

The previous chapters introduced lookup, prediction, and contrastive training. This chapter connects them: each training signal writes a different geometry into the same kind of vector space.

## Mathematical object

Let an encoder produce an embedding:

```math
z_i = f_\theta(x_i) \in \mathbb{R}^{d}
```

Different objectives apply different pressures to the same `z_i`.

For classification:

```math
p(y \mid x_i) = softmax(Wz_i + b)
```

and:

```math
L_{cls} = -\log p(y_i \mid x_i)
```

This encourages embeddings to be useful for linear separation by the classifier head. Points with the same label may cluster, but only if clustering helps the decision boundary.

For contrastive retrieval:

```math
L_{ret,i} = -\log
\frac{\exp(sim(q_i, d_i) / \tau)}
{\sum_j \exp(sim(q_i, d_j) / \tau)}
```

This encourages the matching document to outrank competing documents. It directly shapes neighborhoods under the chosen similarity function.

For reconstruction:

```math
\hat{x}_i = g_\phi(z_i), \quad L_{rec} = \|x_i - \hat{x}_i\|_2^2
```

This encourages `z_i` to preserve information needed to rebuild the input, including details irrelevant to labels or retrieval.

For language modeling:

```math
L_{lm} = -\sum_t \log p(x_{t+1} \mid x_{\le t})
```

This encourages representations that help predict token continuations. That can produce rich linguistic structure, but the geometry is optimized for prediction inside the model, not necessarily for sentence-level cosine retrieval.

## The gradient is the shaping force

An objective changes an embedding through its gradient:

```math
z_i \leftarrow z_i - \eta \frac{\partial L}{\partial z_i}
```

Two losses can see the same example and push its embedding in different directions. A class loss may pull two examples together because they share a label. A retrieval loss may push them apart because they answer different queries. A reconstruction loss may preserve a nuisance feature that the class loss would discard.

This is why embeddings trained on the same dataset can have different nearest neighbors.

## PyTorch equivalent

The same encoder output can be trained with different heads and losses:

```python
import torch
from torch import nn
import torch.nn.functional as F

encoder = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 16),
)

x = torch.randn(32, 20)
z = encoder(x)

# Classification objective.
labels = torch.randint(0, 4, (32,))
classifier = nn.Linear(16, 4)
loss_cls = F.cross_entropy(classifier(z), labels)

# Reconstruction objective.
decoder = nn.Linear(16, 20)
loss_rec = F.mse_loss(decoder(z), x)

# Contrastive objective over two augmented views.
z1 = F.normalize(encoder(x + 0.01 * torch.randn_like(x)), dim=-1)
z2 = F.normalize(encoder(x + 0.01 * torch.randn_like(x)), dim=-1)
logits = z1 @ z2.T / 0.1
targets = torch.arange(x.size(0))
loss_ret = F.cross_entropy(logits, targets)
```

Each loss is valid, but each asks the embedding to preserve different information.

## How common objectives shape geometry

Classification tends to create decision-oriented geometry. If the final classifier is linear, embeddings only need to make classes separable by hyperplanes. Variation inside a class can be compressed or ignored.

Contrastive learning tends to create neighborhood-oriented geometry. It cares about relative scores between positives and negatives. The same class can split into many neighborhoods if the positives demand fine-grained distinctions.

Regression tends to organize vectors along directions that predict a continuous target. If price is the target, one dominant direction may encode price even when other semantic properties are present.

Reconstruction tends to preserve many details. This can help representation quality when the input structure matters, but it can also waste capacity on irrelevant variation.

Language modeling creates context-sensitive representations. A token's hidden state is useful because it supports next-token prediction after attention and MLP layers. A raw token embedding or hidden state from a language model is not automatically a good standalone retrieval embedding unless it is pooled, normalized, and trained or adapted for that use.

Multi-task learning combines pressures:

```math
L = \lambda_1 L_{cls} + \lambda_2 L_{ret} + \lambda_3 L_{reg}
```

The weights `\lambda` are not cosmetic. They decide which geometry wins when objectives disagree.

A practical way to see conflict is to compare gradients from two objectives on the same representation. If their cosine similarity is negative, the losses are trying to move the embedding in opposing directions:

```python
z = encoder(x)
loss_a = F.cross_entropy(classifier(z), labels)
loss_b = F.mse_loss(decoder(z), x)

grad_a = torch.autograd.grad(loss_a, z, retain_graph=True)[0]
grad_b = torch.autograd.grad(loss_b, z, retain_graph=True)[0]
alignment = F.cosine_similarity(
    grad_a.flatten(),
    grad_b.flatten(),
    dim=0,
)
print(alignment.item())
```

This is not a full multi-task optimization strategy, but it is a useful diagnostic. If two objectives consistently disagree, changing only the loss weights may not be enough.

## What this means in ML systems

A production embedding model should be judged by the decision it supports. For RAG, the important question is whether the answer-bearing passages appear in the top `k` retrieved results. For recommendations, the question may involve ranking quality, diversity, calibration, and business constraints. For clustering, the question is whether the space preserves the structure the analyst wants to discover.

Metric choice is part of the objective. A model trained with cosine-normalized contrastive loss is usually safest to serve with normalized vectors and inner product search. A recommender trained with raw dot products may rely on vector norms. A classifier may perform well even if nearest neighbors are not semantically satisfying.

Evaluation should therefore match use:

- classification embeddings: accuracy, calibration, confusion patterns, robustness
- retrieval embeddings: recall at `k`, MRR, nDCG, hard-query slices, latency under ANN search
- recommender embeddings: ranking metrics, novelty, diversity, exposure bias, cold-start behavior
- clustering embeddings: cluster stability, human coherence, downstream usefulness

The same rule applies to offline visualization. A 2D projection can be useful for debugging, but the production question is usually about decisions: which class was assigned, which item was ranked, which passage was retrieved, or which cluster was acted on.

## Common failure modes

- Reusing embeddings outside their objective. A next-token model representation may not behave like a sentence embedding.
- Evaluating with pretty nearest neighbors only. Anecdotes hide failure on rare, ambiguous, or adversarial cases.
- Mixing metrics. Training with one similarity function and serving with another changes the geometry.
- Ignoring objective conflict. Multi-task losses can fight, and the larger gradient may dominate.
- Treating labels as truth without checking collection bias. Clicks, purchases, and ratings reflect exposure and interface effects.
- Over-compressing variation. A classifier can discard distinctions later needed for retrieval or personalization.
- Over-preserving variation. A reconstruction objective can keep noise that hurts semantic ranking.
- Assuming 2D plots reveal the full space. Projection methods show a distorted view of high-dimensional geometry.
- Letting a dominant auxiliary objective take over. In multi-task systems, the easiest or largest-scale loss can shape the shared embedding while the intended downstream task receives little useful capacity.
- Optimizing a proxy metric after the deployment metric has changed. For example, a retriever tuned for exact cosine search may behave differently after ANN indexing, reranking, or context-window constraints are added.

## Visual idea

Use the same set of points in three panels. Color them by category, price bucket, and retrieval relevance. Show arrows from each objective: classification pulls by class, regression orders by target value, and contrastive retrieval pulls query-positive pairs together while pushing negatives away.

## Small experiment

Train two 2D embedding tables over the same 12 IDs. One predicts category; the other predicts a synthetic continuous score. Compare nearest neighbors:

```python
import torch
from torch import nn
import torch.nn.functional as F

def train_classifier(labels):
    emb = nn.Embedding(12, 2)
    head = nn.Linear(2, 3)
    opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=0.05)
    ids = torch.arange(12)

    for _ in range(300):
        loss = F.cross_entropy(head(emb(ids)), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return emb.weight.detach()

def train_regressor(targets):
    emb = nn.Embedding(12, 2)
    head = nn.Linear(2, 1)
    opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=0.05)
    ids = torch.arange(12)

    for _ in range(300):
        pred = head(emb(ids)).squeeze(-1)
        loss = F.mse_loss(pred, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return emb.weight.detach()

category = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2])
score = torch.linspace(0, 1, 12)

z_category = train_classifier(category)
z_score = train_regressor(score)
```

For each embedding table, compute cosine nearest neighbors for every ID. The category-trained table should prefer same-category IDs. The regression-trained table should prefer nearby score values. Same IDs, different objective, different space.

Then evaluate both tables on both tasks. The category table should classify well but may be poor for score ordering. The score table should order nearby scores but may mix categories. This makes objective mismatch visible without relying on a plot.

## Practical takeaways

Objectives create geometry.

Before trusting an embedding space, write down:

1. the loss that trained it
2. the data and labels that supplied the signal
3. the similarity or distance metric used during training
4. the metric and index used during serving
5. the downstream decision the vectors support

Good embeddings are not universally good. They are good when the learned geometry matches the problem the system actually needs to solve.
