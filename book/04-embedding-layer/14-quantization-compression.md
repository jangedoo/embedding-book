# Quantization and Compression

## Summary

Embedding compression can happen at several levels: the model's embedding table, intermediate activations, stored document vectors, ANN indexes, optimizer states, and checkpoints. These are related but not identical problems.

The practical goal is to spend fewer bytes while preserving the geometry that the task depends on.

## Intuition

A float32 embedding coordinate can represent many values. Quantization stores that coordinate with fewer bits.

For example, int8 quantization may replace a 32-bit float with an 8-bit integer plus a scale factor. This can reduce memory by about 4x, but the recovered value is approximate.

If the approximation preserves nearest-neighbor rankings, recommendation scores, or model accuracy, the compression is useful. If it changes important rankings, it is too aggressive.

Compression is therefore a controlled geometry distortion. The central engineering question is not "how many bits can we remove?" but "which distortions can the application tolerate?"

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

It is useful to name the compression target before choosing a method. A training-memory problem may call for optimizer-state compression. A retrieval-memory problem may call for product quantization. A model-serving problem may call for int8 embedding weights and specialized lookup kernels.

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

The scale can be global, per tensor, per row, per column, or per block. Finer-grained scales usually preserve values better, but they add metadata and can reduce kernel efficiency.

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

For per-row int8 quantization of an `N x d` table, the rough storage is:

```math
Nd \times 1 \text{ byte} + N \times \text{scale bytes}
```

If scales are float16, the overhead is `2N` bytes. For large `d`, this overhead is small. For tiny embedding dimensions, scale metadata can matter.

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

For example, a 768-dimensional float32 vector uses 3072 bytes. If product quantization stores 96 one-byte codes, the vector code is only 96 bytes before codebooks and index metadata. The price is that distances are computed through approximate codebook lookups rather than exact coordinates.

## Residual and two-stage compression

Many retrieval systems use compression in stages. A coarse index first narrows the search space, a compressed representation scores many candidates cheaply, and an exact or higher-precision vector reranks the final candidates.

This pattern is useful because the first stage optimizes recall and speed, while reranking repairs some of the approximation error. The system may store compressed vectors for all documents and full-precision vectors only for a smaller candidate cache, or it may fetch exact vectors from a slower store during reranking.

The practical metric is end-to-end recall and latency, not the error of one compression stage in isolation.

## Binary and sign embeddings

An extreme compression is to store only signs:

```math
b_i = \text{sign}(x_i)
```

This turns a vector into bits. It is very compact, and Hamming distance can be fast, but much magnitude information is lost.

Binary representations can work for specific retrieval systems, but they need careful training or calibration. Simply taking the sign of a model's float embeddings can badly damage rankings.

Binary codes are attractive when memory and bit operations dominate the system budget. They are risky when vector norms, small angular differences, or score calibration matter.

## Compression and similarity metrics

Compression should preserve the metric used by the system.

For cosine search, the important property is angular similarity. Quantization should preserve directions after normalization.

For dot-product recommendation, vector length may carry signal. Quantization that distorts norms can change ranking.

For Euclidean search, both coordinate error and norm error matter.

This is why evaluation must use the serving metric. A low reconstruction error does not always imply stable top-k ranking.

Normalization order matters. If the serving system uses cosine similarity, compare rankings after applying the same normalization used in production. Quantizing unnormalized vectors and then normalizing after dequantization can behave differently from normalizing first and then quantizing.

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

When labels are available, compute recall@k or nDCG with the compressed scores. Also inspect queries where the relevant item moved from just inside the cutoff to just outside it; these boundary cases often explain user-visible regressions.

## Quantizing model tables versus stored vectors

Quantizing a model embedding table changes the vectors that every downstream layer receives. The error can be amplified or corrected by later layers depending on the model and whether quantization-aware training is used.

Quantizing stored retrieval vectors changes the search index. The encoder may still produce high-precision query vectors, while the database stores compressed document vectors. In this case, the main question is whether approximate search preserves ranking and recall.

These two changes should be tested separately before combining them.

A good rollout plan isolates interventions:

1. Keep the model fixed and quantize only stored vectors.
2. Keep the index fixed and quantize only the model table or encoder.
3. Combine both only after each individual change has acceptable quality and latency.

This makes regressions diagnosable. If both are changed at once, it is hard to know whether the encoder, the index, or their interaction caused the problem.

## Optimizer-state compression

During training, optimizer states can be larger than the weights. Compression can target:

- gradients
- Adam moments
- optimizer checkpoints
- communication between workers

For embedding tables with sparse updates, row-wise optimizer states are often attractive. For distributed training, gradient compression can reduce communication bandwidth, but it can also introduce optimization noise.

Optimizer-state compression is a training-system decision. It should be evaluated by convergence speed and final quality, not only memory saved.

For large embedding tables, row-wise optimizer statistics can be a good compromise. Instead of storing a full second-moment vector for every coordinate, a system may store one statistic per row or per block. This reduces memory but changes the optimizer's behavior, so learning curves and rare-row quality should be checked.

## Deployment tradeoffs

Compression changes serving behavior.

Lower precision can reduce memory footprint, improve cache locality, and increase throughput. It can also introduce dequantization overhead, require specialized kernels, or reduce quality.

For retrieval systems, compressed indexes can hold more vectors in memory, which may improve latency and scale. But approximate codes may reduce recall, especially for hard queries where relevant documents are close to many distractors.

For recommender systems, quantized item embeddings can shift scores. Small score shifts near the decision boundary can change ranked lists, diversity, and business metrics.

A deployment decision usually balances four quantities:

- memory saved
- latency saved or added
- quality loss
- implementation complexity

Int8 scalar quantization may be simple and fast on available hardware. Int4 or product quantization may save more memory but require specialized kernels, codebooks, or reranking. Binary codes may be extremely compact but may require training the model for binary retrieval rather than quantizing after the fact.

The best compression method is often the least aggressive one that makes the system fit its memory, latency, and cost targets.

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

Then repeat after adding a reranking stage:

1. Use compressed vectors to retrieve top 100 candidates.
2. Rerank those candidates with full-precision vectors.
3. Compare recall@10 and latency with exact full-precision search.

This experiment separates candidate-generation recall from final-ranking quality.

## Common failure modes

- Reporting compression ratio without measuring retrieval quality.
- Measuring reconstruction error but not ranking changes.
- Quantizing before normalization when the serving system expects normalized vectors.
- Using one global scale for rows with very different norms.
- Forgetting to store scales or codebooks when calculating memory.
- Combining model quantization and index quantization without isolating which one caused a regression.
- Using approximate compressed search without reranking when high precision is required.
- Assuming int8, int4, product quantization, and binary hashing have the same failure modes.
- Forgetting that scale factors, zero points, codebooks, and index metadata consume memory.
- Evaluating only random queries instead of hard queries with close nearest neighbors.
- Quantizing a model table and a retrieval index at the same time, making regressions hard to attribute.

## Practical takeaways

- Compression reduces bytes by approximating vectors or optimizer states.
- The right compression target depends on whether the bottleneck is training memory, serving memory, index size, bandwidth, or latency.
- Quantizing a model table and quantizing stored retrieval vectors are different interventions.
- Preserve the geometry required by the serving metric.
- Evaluate with task quality and ranking metrics, not just reconstruction error.
- Compression is successful only when the saved memory is worth the measured quality and latency tradeoff.
- Account for metadata such as scales, zero points, codebooks, and ANN structures.
- Reranking can recover quality when compressed first-stage retrieval has high enough candidate recall.
