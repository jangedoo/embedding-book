# Graph and Entity Embeddings

Graph and entity embeddings represent nodes, entities, and relationships as vectors. They make symbolic structures usable by nearest-neighbor search, link prediction, clustering, classification, and downstream neural models.

The key difference from plain item embeddings is relational structure. A node is defined not only by its features, but also by its edges and the neighborhoods around those edges.

## Intuition

In a social graph, two users may be similar because they connect to similar people. In a knowledge graph, "Paris" is related to "France" through a capital-of relation, while "Paris" the person may connect through entirely different relations. The embedding must compress both identity and structure.

Graph embeddings answer questions such as:

- Which nodes are likely to connect?
- Which entities refer to the same real object?
- Which relation is missing from a knowledge graph?
- Which communities or roles exist in the graph?

## Mathematical object

For a graph with `N` nodes, node embeddings are:

```{math}
E \in \mathbb{R}^{N \times d}
```

For link prediction, a simple score is:

```{math}
score(i, j) = e_i^\top e_j
```

Knowledge graphs include relation types. A triple is:

```{math}
(h, r, t)
```

One classic translational model scores:

```{math}
score(h, r, t) = -\|e_h + e_r - e_t\|
```

The idea is that the relation vector moves the head entity toward the tail entity.

## PyTorch equivalent

```python
import torch
from torch import nn

num_entities, num_relations, d = 100_000, 500, 128
entity = nn.Embedding(num_entities, d)
relation = nn.Embedding(num_relations, d)

h = torch.tensor([10, 20, 30])
r = torch.tensor([2, 2, 9])
t = torch.tensor([11, 25, 44])

score = -(entity(h) + relation(r) - entity(t)).norm(dim=-1)
```

For message-passing graph neural networks, embeddings are updated by aggregating neighbor information:

```{math}
h_i' = \phi\left(h_i, \operatorname{AGG}_{j \in \mathcal{N}(i)} h_j\right)
```

This lets a node representation depend on local graph structure, not only its ID.

For nodes with attributes, combine ID embeddings and feature-derived embeddings:

```{math}
h_i = e_i + g(features_i)
```

This gives the system a path for cold-start nodes whose ID row has not been trained yet.

## What this means in ML systems

Graph and entity embeddings are used for:

- entity resolution and deduplication
- knowledge graph completion
- recommendations from user-item graphs
- fraud and anomaly detection
- semantic search over entities
- feature generation for tabular models

The production challenge is that graphs change. New nodes arrive, edges update, and entity merges can invalidate old IDs. Systems need a plan for incremental updates, stale embeddings, and backfills.

Temporal ordering matters. If an edge appears after the prediction time, it must not influence the training neighborhood for that evaluation point. Leakage through graph structure is easy to miss because it can arrive through multi-hop paths.

## Common failure modes

- High-degree nodes dominate training and retrieval.
- Random walk methods learn popularity more than semantic similarity.
- Negative samples include false negatives: missing edges that are actually true.
- Entity IDs change after deduplication, breaking embedding lookup.
- Relations with different semantics are collapsed into one undirected edge.
- Train/test splits leak through graph neighborhoods.
- New nodes have no learned embedding and require features or neighbor aggregation.

## Visual idea

Draw a small graph beside a 2D embedding plot. Show that nodes in the same community are close, while nodes with the same structural role may also become close even if they are far apart in the original graph. For a knowledge graph, draw `head + relation -> tail` as vector translation.

A useful failure visual colors nodes by degree. If the center of the embedding plot is mostly high-degree nodes, the model may be learning exposure or popularity more than relation semantics.

## Small experiment

Build a tiny bipartite user-item graph. Train node embeddings with negative sampling for link prediction. Compare random negative sampling with popularity-weighted negative sampling, and report AUC plus top-k recommendation diversity. Inspect whether high-degree items dominate.

Repeat with a time-based edge split. Compare performance on old nodes, new nodes with features, and new nodes with no features.

## Practical takeaways

Graph embeddings compress relationships, not just object features.

Evaluate them with graph-aware splits, degree-stratified metrics, and checks for stale IDs, false negatives, and high-degree dominance.
