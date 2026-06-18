# Embedding Lookup Is Matrix Indexing

An embedding layer is a matrix:

```math
E \in \mathbb{R}^{V \times d}
```

Looking up token `i` means selecting row `i`:

```python
x = E[i]
```

This is equivalent to multiplying a one-hot vector by the embedding matrix:

```math
x = onehot(i)^T E
```

## Why this matters

Only selected rows receive gradients during a lookup. Rare tokens receive fewer updates, and large vocabularies consume large memory.
