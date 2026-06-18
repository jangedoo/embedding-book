# Recommender-System Embeddings

Recommender systems often learn user and item embeddings.

A simple score is:

```math
score(user, item) = u \cdot v
```

This score depends on both angle and norm. Item norm can encode popularity, while angle can encode preference match.
