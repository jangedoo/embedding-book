# Designing Embedding Experiments

Embedding work needs careful experiments because small implementation choices can change retrieval, ranking, memory, latency, and downstream model behavior. Good experiments isolate one change, measure the right behavior, and preserve enough examples for inspection.

The goal is not to produce one impressive metric. The goal is to learn which design choices are reliable enough to ship.

## Intuition

Embedding systems are pipelines. A change in chunk size can look like a model improvement. A metric change can look like better embeddings. ANN settings can hide a recall regression. Without careful experiment design, you may optimize the wrong component.

A useful experiment starts with a specific question:

- Does normalization improve retrieval for this model?
- How much rank can a factorized table lose before quality drops?
- Is quantization hurting rare entities more than common ones?
- Which chunk size maximizes answer recall per token?

## Mathematical object

Most embedding experiments compare a baseline system `A` and candidate system `B` on a metric:

```math
\Delta = metric(B) - metric(A)
```

For retrieval, common metrics include:

```math
Recall@k = \frac{\text{queries with a relevant item in top k}}{\text{queries}}
```

```math
MRR = \frac{1}{Q}\sum_{q=1}^{Q}\frac{1}{rank_q}
```

For compression or transformations, also track behavior drift:

```math
overlap@k = \frac{|N_k^{old}(q) \cap N_k^{new}(q)|}{k}
```

where `N_k(q)` is the top-`k` neighbor set for query `q`.

## PyTorch and NumPy equivalent

```python
import torch

scores = torch.randn(100, 1000)
relevant = torch.zeros(100, 1000, dtype=torch.bool)
relevant[torch.arange(100), torch.randint(0, 1000, (100,))] = True

topk = scores.topk(k=10, dim=1).indices
hits = relevant.gather(1, topk).any(dim=1)
recall_at_10 = hits.float().mean()
```

Nearest-neighbor overlap:

```python
old_top = torch.randint(0, 10_000, (100, 10))
new_top = torch.randint(0, 10_000, (100, 10))

overlap = []
for a, b in zip(old_top, new_top):
    overlap.append(len(set(a.tolist()) & set(b.tolist())) / len(a))

mean_overlap = sum(overlap) / len(overlap)
```

Use deterministic seeds for repeatability, but do not rely on one seed when training or nonlinear visualization is involved.

## What this means in ML systems

A good experiment changes one thing at a time:

- metric
- normalization
- dimension
- factorization rank
- quantization level
- index type
- ANN search parameters
- training objective
- negative sampling strategy
- chunk size or pooling strategy

Useful plots include memory vs quality, latency vs recall, rank vs reconstruction error, dimension vs retrieval score, normalization vs nearest-neighbor stability, and subgroup quality vs compression level.

Keep a fixed evaluation set with real or realistic queries. Include hard negatives, rare items, multilingual examples if relevant, short queries, long queries, and examples from recent data.

## Common failure modes

- Changing model, preprocessing, and index settings in the same run.
- Testing on examples used to tune prompts or thresholds.
- Reporting only averages and missing subgroup regressions.
- Optimizing exact search metrics, then serving with approximate search.
- Ignoring latency, memory, and rebuild cost.
- Using synthetic negatives that are too easy.
- Judging RAG by final answer quality without retrieval labels.

## Visual idea

Draw an experiment matrix with rows as model variants and columns as metrics: recall, MRR, latency, memory, overlap, subgroup recall, and qualitative examples. Highlight the chosen candidate only if it improves the target metric without unacceptable regressions.

## Small experiment

Take one embedding model and one labeled retrieval set. Sweep four settings: raw vs normalized vectors, cosine vs dot product, exact vs approximate search, and two chunk sizes. Change only one axis per run. Build a table with recall@5, MRR, latency, memory, and top-10 overlap against baseline.

## Practical takeaways

Embedding experiments should be boring in the best way: controlled, reproducible, and tied to deployment behavior.

Always report the task metric, qualitative examples, resource costs, and the exact preprocessing and serving configuration that produced the result.
