# Probabilistic Views of Embeddings

Dot products often become logits. This gives embeddings a probabilistic interpretation: vectors do not only define geometry; they parameterize distributions over labels, tokens, items, contexts, or documents.

This view is useful because many embedding models are trained by asking one vector to assign high probability to the right partner and low probability to alternatives.

## Intuition

If a hidden state `x` points strongly toward class vector `w_i`, the model assigns class `i` a high logit. Softmax turns all logits into probabilities. Training then moves vectors so correct pairs become more probable.

The same mechanism appears in language modeling, contrastive learning, and recommendation. The objects change, but the pattern remains: score candidates with dot products, normalize or compare against negatives, and update the vectors.

## Mathematical object

For class weights `w_i`, the probability of label `i` given vector `x` is:

```{math}
p(y=i|x) = \frac{\exp(x \cdot w_i)}{\sum_j \exp(x \cdot w_j)}
```

Temperature controls how sharp or flat the distribution is:

```{math}
p(y=i|x) = \frac{\exp((x \cdot w_i)/\tau)}{\sum_j \exp((x \cdot w_j)/\tau)}
```

Small `tau` makes the distribution sharper. Large `tau` makes it flatter.

For contrastive learning with one positive `x^+` and negatives `x_j^-`:

```{math}
L = -\log \frac{\exp(q^\top x^+ / \tau)}
{\exp(q^\top x^+ / \tau) + \sum_j \exp(q^\top x_j^- / \tau)}
```

This objective trains `q` to put probability mass on the positive among the candidate set.

## PyTorch equivalent

```python
import torch
import torch.nn.functional as F

batch, d, classes = 32, 128, 1000
x = torch.randn(batch, d)
W = torch.randn(classes, d)
target = torch.randint(0, classes, (batch,))

logits = x @ W.T
loss = F.cross_entropy(logits, target)
```

Contrastive in-batch negatives:

```python
q = F.normalize(torch.randn(32, 128), dim=-1)
doc = F.normalize(torch.randn(32, 128), dim=-1)

temperature = 0.05
logits = q @ doc.T / temperature
labels = torch.arange(q.shape[0])
loss = F.cross_entropy(logits, labels)
```

The diagonal entries are treated as positives. Off-diagonal entries are treated as negatives, which is efficient but can create false negatives.

If a batch contains multiple valid documents for the same query, mask those entries instead of treating them as negatives:

```python
logits = logits.masked_fill(false_negative_mask, -1e9)
```

The mask should not hide the diagonal positive.

## What this means in ML systems

Probabilistic thinking explains several practical choices:

- larger candidate sets make softmax normalization harder but more informative
- sampled negatives approximate full normalization
- temperature affects calibration and gradient sharpness
- vector norms can affect confidence when embeddings are not normalized
- softmax probabilities are conditional on the candidate set

In retrieval, a high score is not automatically an absolute probability of relevance. It is often a relative score among candidates produced by a particular model and index.

If probabilities are used for decisions, evaluate calibration. Bucket examples by predicted probability and compare predicted confidence with empirical accuracy or relevance rate.

## Common failure modes

- Interpreting softmax scores as calibrated probabilities without calibration tests.
- Using in-batch negatives when many examples in the batch are actually related.
- Setting temperature too low and making training unstable or overconfident.
- Normalizing embeddings at inference when training relied on vector norms.
- Comparing probabilities across different candidate sets.
- Evaluating only top-1 accuracy when probability mass and ranking quality matter.

## Visual idea

```{image} ../../assets/figures/probabilistic-softmax-candidates.svg
:alt: Query, positive, and negative candidate vectors scored by dot products, temperature scaling, and softmax probabilities.
:align: center
:width: 100%
```

This figure turns embedding similarity into a probabilistic prediction pipeline. Dot products between the query and candidates become logits, temperature rescales their sharpness, and the softmax converts the candidate set into probabilities. A positive example receives high probability only when it is not merely close to the query, but sufficiently better than the competing negatives.

The candidate-set comparison is the key warning. The same positive score can produce a high probability among easy negatives and a much lower probability among hard negatives. Contrastive and sampled-softmax objectives therefore learn geometry relative to the negatives they see, which makes batch construction, false-negative masking, and evaluation candidate pools part of the model design.

## Small experiment

Train a tiny contrastive model with different temperatures. Plot training loss, retrieval recall@1, mean embedding norm, and entropy of the softmax distribution. This shows how temperature affects both learning dynamics and confidence.

Add a run with deliberately duplicated positives inside the batch. Compare the loss with and without masking false negatives to see how batch construction changes the objective.

## Practical takeaways

Embedding geometry and probability are connected through scoring functions.

When dot products become logits, normalization, temperature, negative sampling, and candidate set construction become part of the statistical model.
