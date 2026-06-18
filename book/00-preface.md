# Preface

```{note}
Authorship: this book was written by OpenAI Codex. Sanjaya Subedi curated the content and direction for personal learning, but should not be understood as the author of the prose.
```

This book is for practitioners who have used embeddings but want to understand them more deeply.

You may already know that embeddings are dense vectors. But many important practical questions are still easy to get wrong:

- Should I use cosine similarity or Euclidean distance?
- Should I normalize vectors before clustering?
- Can I reduce the vocabulary by merging token IDs?
- Can I factorize the embedding table?
- Does increasing dimension always add information?
- What does a ReLU layer do to the geometry?
- Why do retrieval metrics look good offline but fail in production?

The central idea of the book is simple:

> An embedding is not just a vector. It is a vector produced by a training objective, transformed by model layers, compared by a metric, stored by a system, and interpreted by a downstream task.

That means we need several mental models at once:

- linear algebra
- geometry
- optimization
- information retrieval
- systems and memory
- visualization
- evaluation

The book is written as a set of practical chapters. Each chapter should contain intuition, math, code, and failure modes.
