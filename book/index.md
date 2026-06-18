# Embeddings: Geometry, Learning, and Retrieval

Embeddings are everywhere in modern machine learning: language models, search systems, recommender systems, graph models, image-text models, and retrieval augmented generation.

But embeddings are often treated as magic vectors.

This book takes the opposite approach. We will treat embeddings as mathematical objects, learned parameters, geometric spaces, compression systems, retrieval indexes, and practical engineering tools.

The goal is to build intuition that transfers.

After reading this book, you should be able to reason about questions like:

- What does cosine similarity ignore?
- Why does Euclidean distance change when vector norms differ?
- What does vector length mean?
- What happens when an embedding layer is factorized?
- What happens when multiple tokens share one embedding ID?
- How do linear layers reshape embedding spaces?
- Why does ReLU change the geometry?
- How do contrastive losses shape neighborhoods?
- Why do ANN indexes care about the metric?
- How do we evaluate embedding models without fooling ourselves?

```{important}
This is not only a math book. Every chapter should connect the math to practical machine learning behavior.
```
