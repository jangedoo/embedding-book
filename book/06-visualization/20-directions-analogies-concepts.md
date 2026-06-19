# Directions, Analogies, and Concepts

A direction in embedding space can sometimes represent a concept. The word "sometimes" matters. Linear directions are powerful debugging tools, but they are not guaranteed to be clean semantic switches.

A concept direction is useful when moving along it changes examples in a consistent, measurable way: more formal, more plural, more expensive, more medical, more action-oriented, or more associated with a product category.

## Intuition

If two groups differ mostly along one axis, the difference between their average vectors can become a concept direction. For example, the average of "question-like" queries minus the average of "statement-like" queries may point toward a question style. Projecting new vectors onto that direction gives a rough score.

Analogy arithmetic is the same idea in a more famous form:

```{math}
king - man + woman \approx queen
```

This works only when the embedding space has learned a locally linear representation of the relationship. Many concepts are curved, entangled, context-dependent, or split across several directions.

## Mathematical object

Given positive examples `P` and negative examples `N`, a simple concept direction is:

```{math}
v = \frac{1}{|P|}\sum_{p \in P} p - \frac{1}{|N|}\sum_{n \in N} n
```

Often we normalize it:

```{math}
\hat{v} = \frac{v}{\|v\|}
```

For a concept direction `v`, projection asks how much another embedding points along that concept:

```{math}
score(x) = x \cdot v
```

If both `x` and `v` are normalized, this is cosine similarity to the direction. If `x` is not normalized, vector length also affects the score.

For analogy search:

```{math}
q = x_b - x_a + x_c
```

Then retrieve nearest neighbors to `q`, usually excluding the source terms.

## PyTorch equivalent

```python
import torch
import torch.nn.functional as F

positive = torch.randn(20, 384)
negative = torch.randn(20, 384)
items = torch.randn(1000, 384)

v = positive.mean(dim=0) - negative.mean(dim=0)
v = F.normalize(v, dim=0)

items_n = F.normalize(items, dim=-1)
scores = items_n @ v
top = scores.topk(k=10)
```

For stronger baselines, train a linear probe and use its weight vector as the direction. The probe tests whether the concept is linearly available, not whether the model truly reasons about the concept.

```python
labels = torch.cat([torch.ones(len(positive)), torch.zeros(len(negative))])
train = torch.cat([positive, negative])
probe = torch.nn.Linear(train.shape[1], 1)
```

After training the probe, inspect false positives and false negatives. Those examples usually reveal whether the direction captures the intended concept or a shortcut such as length, source, or vocabulary.

## What this means in ML systems

Concept directions are useful for:

- auditing whether sensitive attributes are linearly recoverable
- filtering or reranking retrieved examples by style or domain
- discovering axes such as popularity, length, language, or source
- editing embedding spaces by subtracting a nuisance direction
- building diagnostic dashboards for drift

They are risky when promoted from diagnostic signals to product logic without evaluation. A direction found from a small hand-picked set may work on examples but fail on rare terms, polysemous words, or multilingual data.

Concept directions are most useful when they are narrow. "Legal contract language" is easier to validate than "trustworthiness." If the concept requires context, world knowledge, or multiple independent traits, expect one direction to be an incomplete summary.

## Common failure modes

- Confusing correlation with a stable semantic direction.
- Building a concept vector from examples that differ in multiple ways.
- Forgetting that projection scores depend on normalization.
- Assuming analogies are symmetric and globally valid.
- Using nearest neighbors from a transformed query without checking source-term leakage.
- Treating a linear probe as proof of causal use by the downstream model.

## Visual idea

```{image} ../../assets/figures/concept-direction-analogy.svg
:alt: Embedding cloud with a concept direction between class centroids, projection scores, and an analogy parallelogram.
:align: center
:width: 100%
```

This figure makes a concept direction concrete: the arrow from a negative centroid to a positive centroid defines a scoring axis, and each example receives a scalar score by projection onto that axis. The perpendicular drop lines are important because they show that a direction is a hypothesis about one linear factor, not a complete explanation of each point's position.

The analogy panel shows the parallelogram implied by `b - a + c`, then compares the intended retrieved point with a plausible wrong neighbor. That wrong neighbor is not noise in the diagram; it teaches the main lesson. Vector analogies work only when the relation is represented consistently enough for a linear offset to survive local density, normalization, and unrelated semantic variation.

## Small experiment

Use sentence embeddings for short text snippets labeled as questions and statements. Build a concept direction from a small training split, score a held-out split, and plot score histograms. Then intentionally contaminate positives with longer text and test whether the direction learned "question-ness" or just length.

Add a counterfactual check: rewrite some questions as statements without changing topic, and rewrite some statements as questions. A robust direction should move with the form change more than with the topic.

## Practical takeaways

Directions are hypotheses about linear structure.

A good concept direction should be validated with held-out examples, counterexamples, nearest-neighbor inspection, and perturbations that isolate the intended concept from confounders.
