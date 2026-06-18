# Contrastive Learning and Metric Learning

Contrastive learning shapes a space by pulling positives together and pushing negatives apart.

Given an anchor `a`, positive `p`, and negative `n`, we want:

```math
sim(a, p) > sim(a, n)
```

## Practical interpretation

For retrieval, this means a query should be closer to the correct document than to incorrect documents.

## Failure modes

- false negatives
- data leakage
- collapsed embeddings
- overly easy negatives
- mismatch between training metric and retrieval metric
