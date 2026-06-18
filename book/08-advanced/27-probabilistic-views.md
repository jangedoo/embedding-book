# Probabilistic Views of Embeddings

Dot products often become logits.

```math
p(y=i|x) = \frac{\exp(x \cdot w_i)}{\sum_j \exp(x \cdot w_j)}
```

Temperature controls how sharp or flat the distribution is.
