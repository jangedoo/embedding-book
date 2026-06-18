# ANN Indexes and Metric Compatibility

Approximate nearest-neighbor indexes make vector search fast enough for large corpora. They trade exactness, memory, and build cost for lower query latency.

This chapter explains how ANN indexes work at a practical level, why metric compatibility matters, and how to reason about latency, recall, and memory when tuning a retrieval system.

## Intuition

Exact search compares the query to every vector:

```math
O(ND)
```

where `N` is the number of stored vectors and `D` is the embedding dimension. This is simple and accurate, but expensive when `N` is large.

ANN search avoids scanning everything. It uses an index structure to inspect a small, promising part of the corpus. The cost is that it may miss a true nearest neighbor.

The system question is not "is approximate search exact?" It is:

> How much recall do we need, at what latency, with how much memory?

## The exact baseline

For a query `q` and matrix of document vectors:

```math
X \in \mathbb{R}^{N \times D}
```

exact inner-product search computes:

```math
s = Xq
```

and returns the largest `k` scores.

In PyTorch:

```python
import torch
import torch.nn.functional as F

docs = F.normalize(torch.randn(1000, 128), dim=-1)
query = F.normalize(torch.randn(128), dim=-1)

scores = docs @ query
exact_scores, exact_ids = scores.topk(k=10)
```

Every ANN index should be compared against this exact baseline on a sample. Otherwise, it is hard to know whether errors come from the embedding model or from the index.

## Metric compatibility

An index must match the ranking metric or a valid transformation of it.

Cosine search over normalized vectors can be implemented as inner-product search:

```math
\hat{x} = \frac{x}{\|x\|_2}, \quad \hat{q} = \frac{q}{\|q\|_2}
```

```math
\cos(q, x) = \hat{q}^{\top}\hat{x}
```

Euclidean search over normalized vectors is also compatible:

```math
\|\hat{q} - \hat{x}\|_2^2 = 2 - 2\hat{q}^{\top}\hat{x}
```

Because this is a monotonic transformation, the ranking is the same.

Without normalization, these rankings are not equivalent. Maximum inner product search, cosine search, and Euclidean search can return different neighbors.

## ANN families

Most production vector systems use one or more of these ideas.

### Graph indexes

Graph indexes connect each vector to nearby vectors. Search starts from one or more entry points and walks toward better neighbors.

HNSW is the most common example. It builds a multi-layer graph:

- upper layers give long jumps
- lower layers refine local neighborhoods
- query-time search explores a candidate frontier

Important knobs:

- `M`: approximate number of graph links per vector
- `efConstruction`: build-time search depth
- `efSearch`: query-time search depth

Higher values usually increase recall and memory, and often increase latency.

### Inverted file indexes

Inverted file indexes cluster vectors into coarse partitions. A query first finds nearby centroids, then scans vectors inside selected partitions.

If centroids are:

```math
c_1, c_2, \ldots, c_C
```

the index assigns each vector to a coarse cell:

```math
a(i) = \arg\min_j \|x_i - c_j\|_2
```

At query time, it probes a small number of cells instead of the whole corpus.

Important knobs:

- number of coarse clusters
- number of probed clusters
- whether residual compression is used

Too few probes lowers recall. Too many probes approaches exact search.

### Quantized indexes

Quantization stores compressed approximations of vectors. Product quantization splits a vector into sub-vectors and represents each sub-vector by a learned code.

Instead of storing:

```math
x \in \mathbb{R}^D
```

the index stores compact codes. This reduces memory and can improve cache behavior, but distances become approximate.

Quantization is useful when memory bandwidth is the bottleneck. It can hurt recall when the compression is too aggressive or when nearest neighbors are separated by small score margins.

## Recall and latency

ANN quality is often measured by recall relative to exact search:

```math
\operatorname{Recall@k}_{ANN} =
\frac{|\operatorname{TopK}_{ANN}(q) \cap \operatorname{TopK}_{exact}(q)|}{k}
```

This is not the same as relevance recall. It only asks whether the ANN index recovered the same vectors exact search would have returned.

Latency is usually measured with percentiles:

- p50: typical query latency
- p95: slow query latency
- p99: tail latency

Tail latency matters because retrieval often sits inside a larger RAG request. A p99 index search can become a p99 user-visible answer.

## Memory math

Raw float32 vectors require:

```math
4ND \text{ bytes}
```

For `N = 10,000,000` and `D = 768`:

```math
4 \cdot 10{,}000{,}000 \cdot 768 \approx 30.7 \text{ GB}
```

Float16 halves the vector storage:

```math
2ND \text{ bytes}
```

But the index also needs metadata, graph edges, centroids, quantization tables, deleted-document markers, filters, and service overhead. HNSW can use substantially more memory than the raw vectors because graph links are stored alongside embeddings.

Memory is not just a storage cost. It affects cache locality, bandwidth, and the number of replicas needed for serving.

## Implementation sketch: measure ANN recall

This sketch simulates an approximate index by scoring only a random subset of documents. It is not a good ANN algorithm, but it shows the evaluation pattern:

```python
import torch
import torch.nn.functional as F

torch.manual_seed(0)
docs = F.normalize(torch.randn(10_000, 64), dim=-1)
queries = F.normalize(torch.randn(100, 64), dim=-1)
k = 10

exact = (queries @ docs.T).topk(k, dim=1).indices

sample_size = 1000
approx_ids = []
for q in queries:
    candidates = torch.randperm(len(docs))[:sample_size]
    scores = docs[candidates] @ q
    approx_ids.append(candidates[scores.topk(k).indices])
approx = torch.stack(approx_ids)

recall = []
for exact_i, approx_i in zip(exact, approx):
    recall.append(len(set(exact_i.tolist()) & set(approx_i.tolist())) / k)

print(sum(recall) / len(recall))
```

A real benchmark replaces the random subset with the actual index and sweeps index parameters such as `efSearch`, number of probes, or compression level.

## Real system interpretation

ANN tuning is a budget allocation problem.

Increasing recall may require:

- searching more graph nodes
- probing more partitions
- storing more links
- using less compression
- retrieving more candidates before reranking

These changes can increase latency, memory, build time, or cost.

The right operating point depends on the application. A support assistant answering legal or medical policy questions may need higher recall than a product recommendation carousel. A typeahead search box may have stricter latency than an offline batch deduplication job.

## Filters and metadata

Real retrieval often includes filters:

- language
- tenant or customer ID
- time range
- permissions
- document type
- product category

Filters interact with ANN search. If the index retrieves globally nearest vectors and filters afterward, it may throw away most candidates. The final results can have poor recall even when the global ANN recall is high.

For strict filters, use an index strategy that applies filters during search or maintains separate partitions for important filter dimensions. Always evaluate filtered queries separately.

## Common failure modes

- Using an index metric that does not match the model's intended similarity.
- Normalizing vectors before exact evaluation but not before indexing.
- Reporting ANN recall against relevance labels instead of exact top-`k`, mixing two different questions.
- Tuning only average latency and missing p95 or p99 regressions.
- Ignoring memory overhead from graph links, metadata, or replicas.
- Applying metadata filters after retrieval and accidentally destroying recall.
- Evaluating only common queries, not rare entities or narrow filters.
- Using compression settings that look fine on average but fail for near-tie rankings.

## Visual idea

Draw three panels:

1. Exact search: one query connected to every vector.
2. Graph search: the query walks through a neighbor graph toward a local region.
3. Inverted file search: vectors are grouped into cells and the query probes only nearby cells.

Annotate each panel with the tradeoff: exactness, latency, and memory.

## Small experiment

Generate 10,000 normalized random vectors and 100 queries. Compute exact top-10 neighbors. Then simulate approximate search by considering only 250, 500, 1000, and 2000 random candidates per query.

Plot or print:

- candidate count
- average ANN recall@10 against exact search
- time per query

The random candidate method will perform poorly compared with real ANN indexes, but the experiment makes the recall-latency curve concrete.

Then repeat with exact search over all vectors using float32 and float16 storage. Compare whether scores and rankings change for near-tie neighbors.

## Practical takeaways

- Exact search is the baseline for ANN evaluation.
- Metric compatibility is non-negotiable.
- Cosine search is usually implemented as inner product search over normalized vectors.
- ANN recall measures agreement with exact search, not user relevance.
- Latency must be measured with tail percentiles, not only averages.
- Memory includes vectors, index structures, metadata, filters, and replicas.
- Filters can dominate retrieval quality and should be part of evaluation.
