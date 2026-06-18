# Token Embeddings in Language Models

Language models use token embeddings to convert token IDs into vectors that transformer layers can process. The embedding table is the first learned interface between text and model computation.

Token embeddings are not standalone word meanings. They are coordinates optimized for predicting tokens inside a particular architecture, tokenizer, context window, and training distribution.

## Intuition

A tokenizer turns text into integer IDs. The embedding table turns each ID into a vector. A transformer then repeatedly mixes those vectors using attention and MLP layers. The model does not see "cat" as a string; it sees an ID, then a row of a matrix.

Subword tokenization matters. A rare word may be represented by several tokens, so its meaning is assembled from multiple rows plus context. The same token row can also behave differently depending on surrounding tokens.

## Mathematical object

For vocabulary size `V` and model width `d`, the input embedding table is:

```math
E \in \mathbb{R}^{V \times d}
```

For token IDs:

```math
T \in \{0, \ldots, V-1\}^{B \times L}
```

lookup produces:

```math
X = E[T] \in \mathbb{R}^{B \times L \times d}
```

Many language models also use an output matrix:

```math
W_{out} \in \mathbb{R}^{V \times d}
```

Logits for the next token are:

```math
logits = h W_{out}^\top
```

With weight tying, the model reuses the input embedding table:

```math
W_{out} = E
```

This saves parameters and encourages input and output token geometry to share structure.

## PyTorch equivalent

```python
import torch
from torch import nn

V, d = 50_000, 768
token_embedding = nn.Embedding(V, d)
lm_head = nn.Linear(d, V, bias=False)

ids = torch.tensor([[101, 2023, 2003, 1037]])
x = token_embedding(ids)

h = torch.randn(ids.shape[0], ids.shape[1], d)
logits = lm_head(h)

print(x.shape)       # torch.Size([1, 4, 768])
print(logits.shape)  # torch.Size([1, 4, 50000])
```

Weight tying:

```python
lm_head.weight = token_embedding.weight
```

In real models, positional information, layer normalization, attention, and MLP blocks transform token vectors before the output head scores candidate next tokens.

## What this means in ML systems

The embedding table can be a large part of memory, especially for large vocabularies. Parameter count is:

```math
Vd
```

For `V = 50,000` and `d = 4096`, the table has 204.8 million parameters. In bf16, that is about 410 MB for one table before optimizer state.

Practical operations include:

- adding special tokens and resizing the embedding table
- tying or untieing input and output embeddings
- pruning unused vocabulary rows
- factorizing or quantizing the table
- adapting only embeddings during domain tuning

Post-training vocab pruning can save memory, but it changes which token IDs are valid. It is safe only if tokenizer, model weights, output head, and decoding logic agree.

## Common failure modes

- Adding tokens to a tokenizer but not resizing or initializing embeddings.
- Reusing an embedding table with a different tokenizer ID mapping.
- Assuming token vectors alone explain contextual meaning.
- Pruning tokens that still appear during generation or evaluation.
- Breaking tied weights by resizing only one side.
- Measuring embedding similarity between tokens without considering frequency and context.

## Visual idea

Draw a pipeline: text string to tokenizer IDs, IDs to rows in `E`, rows plus positions into transformer blocks, final hidden state to output logits over the vocabulary. Highlight that the same vocabulary appears at both input and output when weights are tied.

## Small experiment

Train a tiny character or subword language model on a small corpus. Inspect nearest neighbors of token embeddings before training, midway, and after training. Then compare input embedding neighbors with output-head neighbors if weights are not tied. This shows how prediction pressure shapes token geometry.

## Practical takeaways

Token embeddings are learned rows in a model-specific coordinate system.

Always keep tokenizer IDs, embedding rows, output logits, and decoding code synchronized. When modifying vocabulary or compression, evaluate both perplexity and generation behavior, not just parameter count.
