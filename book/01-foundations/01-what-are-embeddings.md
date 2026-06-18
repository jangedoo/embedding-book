# What Are Embeddings?

Embeddings turn IDs, text, users, images, products, and documents into vectors that a model can compare, transform, and retrieve. The important idea is not that the vector is a perfect definition of the object. It is a learned coordinate system shaped by a task.

An embedding is a learned coordinate for something discrete or complex: a token, word, sentence, product, user, image, entity, or document.

The useful question is not only "what vector represents this object?" but also:

- what training objective created the vector?
- what transformations will act on it?
- what distance or similarity function will compare it?
- what system will store and retrieve it?
- what downstream decision will use it?

## Intuition

An ID such as `token_id = 42` has no geometry. It is just a label. An embedding table gives that ID a location:

```python
x = embedding[token_id]
```

Now the object can participate in geometry. It can be close to other objects, far away from others, projected, transformed, normalized, clustered, quantized, or retrieved.

If two product IDs are frequently clicked by the same users, training may place their vectors near each other. If two tokens appear in similar contexts, a language model may learn related directions for them. If two documents answer the same question, a retrieval model may place them near the same query vector.

The geometry is learned from the pressure of the objective. Change the objective and the geometry changes.

## Mathematical object

For a vocabulary or catalog of `V` objects, an embedding table is a matrix:

```math
E \in \mathbb{R}^{V \times d}
```

Each row is one vector:

```math
E_i \in \mathbb{R}^d
```

When an integer ID `i` is looked up, the model returns row `E_i`.

For a batch of IDs:

```math
ids \in \{0, \ldots, V-1\}^{B}
```

the result has shape:

```math
X \in \mathbb{R}^{B \times d}
```

For token sequences with batch size `B` and sequence length `T`, the lookup produces:

```math
X \in \mathbb{R}^{B \times T \times d}
```

## PyTorch equivalent

```python
import torch
from torch import nn

embedding = nn.Embedding(num_embeddings=50_000, embedding_dim=768)

ids = torch.tensor([[101, 2023, 2003], [101, 2009, 2515]])
x = embedding(ids)

print(x.shape)  # torch.Size([2, 3, 768])
```

The lookup is equivalent to selecting rows from a matrix. It is not a neural network layer that mixes IDs together. Mixing happens later, through attention, MLPs, pooling, or another model component.

## What this means in ML systems

An embedding is a contract between data representation and computation:

- token embeddings let language models process discrete text with continuous layers
- user and item embeddings let recommender systems compare preferences and products
- sentence embeddings let retrieval systems rank documents by vector similarity
- graph embeddings let downstream models use entities and relationships numerically

The same vector can be used in different ways. A sentence vector may be compared with cosine similarity in a vector database, fed into a classifier, clustered for analysis, or compressed for deployment. Each use makes assumptions about what the geometry means.

## Common failure modes

- The embedding space reflects the training objective but not the downstream task.
- Frequent items dominate because they receive more updates.
- Vector norms encode popularity or confidence when the system expects pure semantic similarity.
- Nearest neighbors look plausible in examples but fail on rare, ambiguous, or out-of-domain inputs.
- A visualization of 2D projections is mistaken for the actual high-dimensional geometry.

## Visual idea

Draw a table on the left with rows labeled by token or item IDs. Draw arrows from selected rows into a 2D scatter plot on the right. The point is that lookup creates locations; downstream metrics decide what closeness means.

## Small experiment

Create a toy catalog of 10 items. Assign each item a random vector, then manually move three related items closer together. Compare nearest neighbors before and after the edit with cosine similarity. This makes the distinction between symbolic identity and learned geometry concrete.

## Practical takeaways

An embedding is a learned interface between symbolic identity and continuous computation.

Always ask what trained the space, what metric will compare it, and what downstream decision will consume it.
