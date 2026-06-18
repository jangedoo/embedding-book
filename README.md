# Embeddings: Geometry, Learning, and Retrieval

A practical mathematical book about embeddings: what they are, how they are learned, how their geometry behaves, and how to reason about them in machine learning, language models, recommender systems, and information retrieval.

This book is written for intermediate to advanced ML practitioners who already know Python and basic deep learning, but want a deeper, reusable mental model of embeddings.

The goal is not only to explain the math. The goal is to make the math useful:

- What does cosine similarity ignore?
- When does Euclidean distance care about vector length?
- What does it mean to factorize an embedding layer?
- What is really happening when an embedding goes through a linear layer?
- What does ReLU do to representation geometry?
- Why do retrieval metrics change when vectors are normalized?
- How do ANN indexes depend on the metric?
- How should we think about token embeddings, sentence embeddings, item embeddings, and graph/entity embeddings?

Published site target:

```text
https://jangedoo.github.io/embedding-book/
```

## Local development

Install MyST:

```bash
npm install -g mystmd
```

Build the HTML site:

```bash
myst build --html
```

Preview locally:

```bash
myst start
```

## GitHub Pages

This repository is configured to deploy with GitHub Actions.

After pushing the repository to GitHub:

1. Go to **Settings → Pages**
2. Set **Build and deployment → Source** to **GitHub Actions**
3. Push to `main`
4. The site should publish to `https://jangedoo.github.io/embedding-book/`

## Structure

```text
book/                 Main book chapters
notebooks/            Runnable experiments and visual demos
src/embedding_book/   Reusable plotting and experiment helpers
assets/               Diagrams and generated figures
AGENTS.md             Batch-completion instructions for coding/writing agents
```

## License

Suggested license: CC BY 4.0 for prose and MIT for code snippets. Adjust before publishing if needed.
