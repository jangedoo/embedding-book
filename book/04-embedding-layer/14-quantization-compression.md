# Quantization and Compression

Embedding compression can happen at several levels: the model's embedding table, intermediate activations, stored document vectors, ANN indexes, optimizer states, and checkpoints. These are related but not identical problems.

The practical goal is to spend fewer bytes while preserving the geometry that the task depends on.

## Intuition

A float32 embedding coordinate can represent many values. Quantization stores that coordinate with fewer bits.

For example, int8 quantization may replace a 32-bit float with an 8-bit integer plus a scale factor. This can reduce memory by about 4x, but the recovered value is approximate.

If the approximation preserves nearest-neighbor rankings, recommendation scores, or model accuracy, the compression is useful. If it changes important rankings, it is too aggressive.

## What can be compressed

Different parts of an embedding system can be compressed:

- input embedding table inside a model
- output embedding or softmax matrix
- user, item, or entity embedding tables
- stored document vectors for retrieval
- ANN index codes
- optimizer states during training
- checkpoints on disk

Quantizing an embedding model is not the same as quantizing stored retrieval vectors. The first changes how vectors are produced. The second changes how vectors are stored and searched.

## Scalar quantization

Scalar quantization maps each float coordinate to a small integer.

A common affine form is:

```math
q = \text{round}(x / s) + z
```

and dequantization is:

```math
\hat{x} = s(q - z)
```

where:

- `s` is a scale
- `z` is a zero point
- `q` is the stored integer
- `\hat{x}` is the approximate recovered value

Symmetric int8 quantization often uses `z = 0`:

```math
q = \text{clip}(\text{round}(x / s), -127, 127)
```

## PyTorch sketch

```python
import torch

def quantize_symmetric_int8(x):
    max_abs = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = max_abs / 127
    q = torch.round(x / scale).clamp(-127, 127).to(torch.int8)
    return q, scale

def dequantize_symmetric_int8(q, scale):
    return q.float() * scale

E = torch.randn(1000, 128)
q, scale = quantize_symmetric_int8(E)
E_hat = dequantize_symmetric_int8(q, scale)

relative_error = (E - E_hat).norm() / E.norm()
print(float(relative_error))
```

This example uses one scale per row because embedding rows can have different ranges. Per-row scaling often preserves vector geometry better than one global scale.

## Numpy sketch

```python
import numpy as np

E = np.random.randn(1000, 128).astype("float32")
max_abs = np.maximum(np.max(np.abs(E), axis=1, keepdims=True), 1e-8)
scale = max_abs / 127
q = np.clip(np.round(E / scale), -127, 127).astype("int8")
E_hat = q.astype("float32") * scale

print(np.linalg.norm(E - E_hat) / np.linalg.norm(E))
```

The stored representation is the int8 matrix plus the scale values.

## Product quantization

Product quantization compresses vectors by splitting each vector into chunks and replacing each chunk with the ID of a learned centroid.

If:

```math
x \in \mathbb{R}^{d}
```

and the vector is split into `m` sub-vectors, each sub-vector is assigned to one codebook entry.

The stored vector becomes:

```math
(c_1, c_2, \ldots, c_m)
```

where each `c_j` is a small integer code.

This is common in large retrieval indexes because it can reduce memory dramatically. The tradeoff is approximate distances. ANN systems often combine product quantization with coarse clustering and reranking.

## Binary and sign embeddings

An extreme compression is to store only signs:

```math
b_i = \text{sign}(x_i)
```

This turns a vector into bits. It is very compact, and Hamming distance can be fast, but much magnitude information is lost.

Binary representations can work for specific retrieval systems, but they need careful training or calibration. Simply taking the sign of a model's float embeddings can badly damage rankings.

## Compression and similarity metrics

Compression should preserve the metric used by the system.

For cosine search, the important property is angular similarity. Quantization should preserve directions after normalization.

For dot-product recommendation, vector length may carry signal. Quantization that distorts norms can change ranking.

For Euclidean search, both coordinate error and norm error matter.

This is why evaluation must use the serving metric. A low reconstruction error does not always imply stable top-k ranking.

## Retrieval evaluation sketch

```python
import torch
import torch.nn.functional as F

docs = F.normalize(torch.randn(10_000, 128), dim=-1)
queries = F.normalize(torch.randn(100, 128), dim=-1)

q_docs, scale = quantize_symmetric_int8(docs)
docs_hat = F.normalize(dequantize_symmetric_int8(q_docs, scale), dim=-1)

scores = queries @ docs.T
scores_hat = queries @ docs_hat.T

top10 = scores.topk(10, dim=-1).indices
top10_hat = scores_hat.topk(10, dim=-1).indices

overlap = []
for a, b in zip(top10, top10_hat):
    overlap.append(len(set(a.tolist()) & set(b.tolist())) / 10)

print(sum(overlap) / len(overlap))
```

Top-k overlap is not a replacement for relevance labels, but it is a useful diagnostic. If overlap collapses, quantization is changing the geometry substantially.

## Quantizing model tables versus stored vectors

Quantizing a model embedding table changes the vectors that every downstream layer receives. The error can be amplified or corrected by later layers depending on the model and whether quantization-aware training is used.

Quantizing stored retrieval vectors changes the search index. The encoder may still produce high-precision query vectors, while the database stores compressed document vectors. In this case, the main question is whether approximate search preserves ranking and recall.

These two changes should be tested separately before combining them.

## Optimizer-state compression

During training, optimizer states can be larger than the weights. Compression can target:

- gradients
- Adam moments
- optimizer checkpoints
- communication between workers

For embedding tables with sparse updates, row-wise optimizer states are often attractive. For distributed training, gradient compression can reduce communication bandwidth, but it can also introduce optimization noise.

Optimizer-state compression is a training-system decision. It should be evaluated by convergence speed and final quality, not only memory saved.

## Deployment tradeoffs

Compression changes serving behavior.

Lower precision can reduce memory footprint, improve cache locality, and increase throughput. It can also introduce dequantization overhead, require specialized kernels, or reduce quality.

For retrieval systems, compressed indexes can hold more vectors in memory, which may improve latency and scale. But approximate codes may reduce recall, especially for hard queries where relevant documents are close to many distractors.

For recommender systems, quantized item embeddings can shift scores. Small score shifts near the decision boundary can change ranked lists, diversity, and business metrics.

## Visual idea

Draw a float vector as a row of continuous bars. Below it, draw the same row snapped to a small set of discrete levels. Then show two nearest-neighbor rankings: one before compression and one after compression, with one pair swapping order.

For product quantization, draw a vector split into chunks, with each chunk pointing to a small codebook.

## Small experiment

Generate normalized vectors, quantize them to int8 with per-row scaling, then measure:

- relative reconstruction error
- cosine similarity before and after quantization
- top-k overlap
- recall@k if labels are available
- memory reduction

Repeat with one global scale and compare against per-row scaling. The difference shows why calibration granularity matters.

## Common failure modes

- Reporting compression ratio without measuring retrieval quality.
- Measuring reconstruction error but not ranking changes.
- Quantizing before normalization when the serving system expects normalized vectors.
- Using one global scale for rows with very different norms.
- Forgetting to store scales or codebooks when calculating memory.
- Combining model quantization and index quantization without isolating which one caused a regression.
- Using approximate compressed search without reranking when high precision is required.
- Assuming int8, int4, product quantization, and binary hashing have the same failure modes.

## Practical takeaways

- Compression reduces bytes by approximating vectors or optimizer states.
- The right compression target depends on whether the bottleneck is training memory, serving memory, index size, bandwidth, or latency.
- Quantizing a model table and quantizing stored retrieval vectors are different interventions.
- Preserve the geometry required by the serving metric.
- Evaluate with task quality and ranking metrics, not just reconstruction error.
- Compression is successful only when the saved memory is worth the measured quality and latency tradeoff.
