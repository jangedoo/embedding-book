# Designing Embedding Experiments

Embedding work needs careful experiments.

A good experiment changes one thing at a time:

- metric
- normalization
- dimension
- factorization rank
- quantization level
- index type
- training objective
- negative sampling strategy

Useful plots include memory vs quality, latency vs recall, rank vs reconstruction error, dimension vs retrieval score, and normalization vs nearest-neighbor stability.
