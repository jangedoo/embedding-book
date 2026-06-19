# Parameter Count and Memory

## Summary

Embedding tables look simple: a row per token, item, user, entity, or document feature. But at production scale, that matrix often becomes one of the largest objects in the system. This chapter explains how to estimate parameter count, training memory, inference memory, and the practical tradeoffs that follow.

## Intuition

An embedding table is a lookup table whose values are learned vectors.

If the vocabulary has `V` rows and each vector has `d` coordinates, the table stores `V` separate vectors. Increasing either vocabulary size or embedding dimension increases memory linearly.

This matters because large embedding tables are not only model parameters. During training they also need gradients, optimizer state, communication bandwidth, checkpoint space, and sometimes sparse update infrastructure.

## The embedding matrix

An embedding table is:

```{math}
E \in \mathbb{R}^{V \times d}
```

The parameter count is:

```{math}
\#params = Vd
```

For `V = 250000` and `d = 1024`:

```{math}
250000 \times 1024 = 256000000
```

That is 256 million parameters for one embedding table.

## From parameters to bytes

Memory depends on dtype:

```{math}
\text{bytes} = Vd \times \text{bytes per parameter}
```

Common sizes:

- float32: 4 bytes
- bfloat16 or float16: 2 bytes
- int8: 1 byte
- int4: 0.5 bytes, usually packed

For the 256 million parameter table:

```{math}
256000000 \times 4 = 1024000000
```

So a float32 table is about 1.0 GB before gradients, optimizer states, allocator overhead, or checkpoints.

## PyTorch sketch

```python
import torch
from torch import nn

V = 250_000
d = 1024

emb = nn.Embedding(V, d)
params = emb.weight.numel()
bytes_fp32 = params * emb.weight.element_size()

print(params)
print(bytes_fp32 / 1024**3, "GiB")
```

The same estimate works for item embeddings, user embeddings, entity embeddings, and feature embeddings. Replace `V` with the number of rows and `d` with the vector width.

## Training memory is larger than inference memory

Inference needs the embedding weights and the activations required by the current request.

Training usually needs:

- weights
- gradients
- optimizer states
- activations needed for backpropagation
- temporary buffers
- checkpoints

With Adam-like optimizers, each parameter often has two optimizer buffers: first moment and second moment.

For float32 training, a simple estimate is:

```{math}
\text{training bytes} \approx Vd \times 4 \times (1 + 1 + 2)
```

The terms are:

- `1` for weights
- `1` for gradients
- `2` for Adam moments

That is about 4x the weight memory before considering mixed precision details and activations.

Mixed precision changes the constants but not the structure of the estimate. A common setup keeps model weights in bfloat16 or float16 for forward computation, stores gradients in reduced precision, and keeps optimizer states in float32. Another setup keeps a float32 master copy of the weights. The exact multiplier depends on the training stack, so a sizing estimate should name the dtype of each component instead of saying "the model is fp16."

## Optimizer state can dominate

Suppose the table has 1.0 GB of float32 weights.

With dense Adam:

- weights: 1.0 GB
- gradients: 1.0 GB
- first moment: 1.0 GB
- second moment: 1.0 GB

The table can require roughly 4.0 GB during training.

This is why a table that looks affordable at inference can become expensive during fine-tuning. Sparse optimizers, row-wise optimizers, mixed precision, factorization, and frozen embeddings are common responses.

For example, freezing a pretrained token embedding table removes gradient and optimizer-state memory for that table. It does not remove the table from inference memory, and it can reduce adaptation quality if the new domain uses tokens differently from pretraining.

## Sparse gradients

An embedding lookup touches only the rows in the batch. If a batch contains token IDs:

```python
ids = torch.tensor([4, 4, 9, 100])
x = emb(ids)
```

then only rows `4`, `9`, and `100` are used.

PyTorch can store sparse gradients for embedding layers:

```python
emb = nn.Embedding(V, d, sparse=True)
loss = emb(ids).pow(2).mean()
loss.backward()

print(emb.weight.grad._indices())
print(emb.weight.grad._values().shape)
```

Sparse gradients reduce gradient memory and update cost when batches touch a small fraction of rows. They do not remove the need to store the embedding weights themselves, and not every optimizer supports sparse gradients.

## Batch activations

If a batch contains `B` sequences of length `T`, the embedding output has shape:

```{math}
B \times T \times d
```

The activation memory is:

```{math}
BTd \times \text{bytes per value}
```

For `B = 32`, `T = 2048`, `d = 4096`, and bfloat16:

```{math}
32 \times 2048 \times 4096 \times 2 \approx 536870912
```

That is about 512 MiB for the embedding activations alone. Later layers often dominate total activation memory, but the embedding output is still part of the budget.

For a pure embedding lookup service, batch activations may be short-lived. For a transformer, the embedding output feeds many layers and participates in backpropagation, so sequence length and batch size can make activation memory more important than the table itself.

## What this means in real ML systems

Large embedding tables create pressure in several places.

Vocabulary size controls row count. Tokenizers with many rare tokens, recommender systems with many items, and ad systems with many categorical IDs all increase `V`.

Embedding dimension controls each row width. Wider vectors can store more independent directions, but they increase memory, bandwidth, and downstream compute.

Optimizer choice controls training overhead. Adam-like optimizers are powerful but expensive. SGD, Adagrad variants, row-wise optimizers, and frozen tables can reduce cost.

Serving dtype controls deployment memory. A bfloat16 or int8 table may be much cheaper to serve than a float32 table, but quantization can change ranking behavior.

Sharding controls feasibility. A table too large for one device may be split across devices or machines, introducing communication and latency costs.

Hot rows control cache behavior. In language models, common tokens are touched frequently. In recommender and ad systems, popular users, items, or categorical features may dominate traffic. Caching hot rows can reduce average latency, but long-tail rows still determine the full storage footprint.

Checkpoint format controls operational cost. A 2 GB inference table can become much larger when the checkpoint also stores optimizer state, sharded metadata, scale factors, or multiple copies for rollback. Disk size, load time, and deploy bandwidth are part of the same design problem.

## Memory and bandwidth

Embedding lookup is often memory-bandwidth bound. The operation is mostly:

1. read IDs
2. gather rows
3. move vectors to the next layer

There is little arithmetic compared with the amount of data read. This makes dtype, caching, row locality, and batching important.

For retrieval systems, stored document vectors have a similar issue. Searching millions of 768-dimensional float32 vectors requires moving a lot of memory unless the system uses compression, approximate indexes, or both.

This is why two models with the same parameter count can serve differently. A dense linear layer does many multiply-adds per byte loaded, while an embedding lookup mostly gathers rows. Embedding-heavy systems often bottleneck on memory bandwidth, network bandwidth, cache misses, or random access patterns before they bottleneck on arithmetic.

## Numpy sizing helper

```python
def table_size(vocab_size, dim, bytes_per_value):
    params = vocab_size * dim
    return params, params * bytes_per_value

for dtype, bytes_per_value in [("fp32", 4), ("bf16/fp16", 2), ("int8", 1)]:
    params, bytes_ = table_size(250_000, 1024, bytes_per_value)
    print(dtype, params, round(bytes_ / 1024**3, 3), "GiB")
```

Use this kind of estimate before choosing a vocabulary size or embedding dimension. It catches many design problems early.

You can extend the estimate to training components:

```python
def training_size(vocab_size, dim, weight_bytes=2, grad_bytes=2, adam_bytes=4):
    params = vocab_size * dim
    # Adam has two moment buffers.
    total = params * (weight_bytes + grad_bytes + 2 * adam_bytes)
    return total / 1024**3

print(round(training_size(250_000, 1024), 2), "GiB")
```

This is still an approximation. It deliberately excludes allocator overhead, distributed-training buffers, framework metadata, and activation memory.

## Deployment tradeoffs

The same table can be acceptable or unacceptable depending on where it sits in the system.

For training, the expensive parts are optimizer state, gradient traffic, checkpoint size, and update frequency. Sparse updates help when each batch touches few rows, but they can complicate optimizer choice and distributed synchronization.

For inference, the expensive parts are resident memory, lookup latency, bandwidth, cold-start load time, and the cost of moving vectors to the next layer. Lower precision can help, but only if quality and ranking behavior remain acceptable.

For retrieval indexes, the table may not be a model parameter at all. It may be a stored matrix of document vectors. The same `N x d` sizing math applies, but the engineering choices become ANN layout, compression, reranking, and recall-latency tradeoffs.

Before deploying a large embedding table, answer four questions:

- How many rows must be resident for the serving path?
- Which dtype is used for storage, lookup, computation, and optimizer state?
- What fraction of rows are touched per batch or request window?
- What quality metric will catch damage from compression, sharding, or caching?

## Visual idea

```{image} ../../assets/figures/embedding-memory-accounting.svg
:alt: Embedding memory accounting diagram with a V by d weight table, matching gradient and Adam state buffers, and a batch lookup touching only a few rows.
:align: center
:width: 100%
```

The figure separates two ideas that are easy to mix together: the full table that must be stored and the small subset of rows touched by one batch. Even if a request selects only a few token or item IDs, the serving system still needs access to the resident `V x d` table, and training may need gradients plus optimizer buffers with the same shape.

The repeated rectangles make the training multiplier visible. A table that looks like one block of parameters during inference can become several equally large blocks during dense Adam training: weights, gradients, first moments, and second moments. Sparse updates can reduce the amount of gradient traffic for a step, but they do not make the stored table itself disappear.

## Small experiment

Create embedding tables with fixed `V` and varying `d`. Measure:

- parameter count
- weight memory
- forward lookup time
- backward time with dense gradients
- backward time with sparse gradients

Then repeat with fixed `d` and varying `V`. The result should make clear which costs scale with dimension, which scale with row count, and which depend mostly on the number of IDs touched per batch.

A useful extension is to measure the number of unique IDs per batch. Create batches with the same total number of IDs but different repetition rates. Sparse update cost should track unique rows more closely than total positions, while activation memory still tracks `B x T x d`.

## Common failure modes

- Counting weights but forgetting gradients and optimizer states.
- Estimating inference memory from training memory, or training memory from inference memory.
- Increasing vocabulary size to cover rare IDs without measuring memory impact.
- Using Adam on a huge table when a sparse or row-wise optimizer would be more appropriate.
- Assuming sparse gradients make the weight table small.
- Forgetting checkpoint size and optimizer checkpoint size.
- Serving float32 embeddings when lower precision would preserve quality.
- Sharding a table without accounting for cross-device lookup latency.
- Sizing only the embedding weights and forgetting checkpoint load time.
- Assuming average lookup latency is enough when tail latency is dominated by cold or remote rows.

## Practical takeaways

- The basic embedding parameter count is `Vd`.
- Training memory can be several times larger than weight memory.
- Adam optimizer states are often the hidden cost.
- Sparse updates help when each batch touches few rows.
- Dtype changes memory and bandwidth directly.
- Vocabulary size, dimension, optimizer, dtype, and sharding are system design choices, not just modeling details.
- Retrieval vectors, recommender tables, and token embeddings all obey the same sizing math, but their deployment bottlenecks differ.
