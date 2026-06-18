# AGENTS.md — Batch Completion Plan

This repository is a book project: **Embeddings: Geometry, Learning, and Retrieval**.

The book should explain embeddings from both mathematical and practical perspectives. It should be understandable to intermediate and advanced machine learning practitioners, especially people who know Python/PyTorch and have worked with language models, sentence embeddings, recommender systems, or vector search.

The output should be a polished GitHub Pages book, not just notes.

## Global writing principles

Every chapter should follow this pattern:

1. Start with intuition.
2. Introduce the mathematical object.
3. Show the PyTorch/Numpy equivalent.
4. Explain what this means in real ML systems.
5. Discuss failure modes.
6. End with practical takeaways and experiments to try.

Avoid purely abstract math. Every equation should answer a practical question.

## Target reader

The reader:

- knows Python and PyTorch
- understands basic linear algebra
- has trained or used ML models
- wants deeper mental models for embeddings
- cares about RAG, retrieval, token embeddings, recommender systems, and model compression

## Sub-agent roles

### 1. Content Architect Agent

Maintain structure, ordering, cross-links, prerequisites, and chapter goals.

### 2. Math Explainer Agent

Provide correct definitions, equations, shapes, assumptions, and plain-English explanations.

### 3. ML Practitioner Agent

Connect ideas to real models, PyTorch, training dynamics, failure modes, and practical experiments.

### 4. Retrieval Engineer Agent

Cover vector search, RAG, ranking metrics, ANN indexes, normalization, latency, recall, and memory.

### 5. Visualization Agent

Design simple explanatory diagrams: vector arrows, matrix shapes, rank bottlenecks, neighbor graphs, and retrieval ranked lists.

### 6. Systems Agent

Explain parameter count, memory, optimizer states, quantization, factorization, deployment, and tradeoffs.

### 7. QA Agent

Check math, code, links, build output, terminology, references, and notebooks.

## Chapter completion standard

A chapter is complete when it includes:

- one-paragraph summary
- intuition
- key equations
- implementation sketch
- practical interpretation
- common failure modes
- at least one visual idea
- at least one small experiment
- takeaways

## Required flagship chapters

### Distance and Similarity

Must explain dot product, cosine similarity, Euclidean distance, squared Euclidean distance, Manhattan distance, Mahalanobis distance, normalized vs unnormalized vectors, when cosine and Euclidean rankings become equivalent, and how vector length affects ranking.

Practical examples:

- DBSCAN with cosine vs Euclidean
- sentence embedding retrieval
- item recommendation with dot product

### Factorized Embeddings

Must explain:

```math
E \in \mathbb{R}^{V \times d}
```

and:

```math
E = AB
```

where:

```math
A \in \mathbb{R}^{V \times r}, \quad B \in \mathbb{R}^{r \times d}, \quad r \ll d
```

Interpretation:

- A token no longer owns a full independent vector.
- It owns a smaller latent code.
- The projection matrix maps that code into model space.
- This creates a low-rank bottleneck.
- The token vector becomes a mixture of shared basis directions.
- This saves memory but reduces independent capacity.

### Clustering and Canonical Token IDs

Must explain what happens when many tokens map to one ID, why this is less problematic for pure encoders than decoders, why decoding ambiguity matters for generative models, and how to evaluate retrieval degradation.

### Linear and Nonlinear Transformations of Embeddings

Must explain:

- what a linear layer does to an embedding space
- how `y = Wx + b` changes coordinates
- what it means to project to fewer dimensions
- what it means to expand to larger dimensions
- why dimension increase does not magically add information
- how rank controls information preservation
- how ReLU gates directions and creates piecewise-linear regions
- why nonlinear layers can separate patterns that linear layers cannot
- how MLPs reshape embedding spaces layer by layer
- how to interpret bottlenecks, expansions, residuals, and normalization

Practical examples:

- projecting 2D points to 1D and losing information
- expanding 2D to 8D and showing that intrinsic dimension is still limited before nonlinearity
- ReLU turning half-spaces off
- MLP separating points that a single linear layer cannot

## Suggested batch order

### Batch 1: Scaffold

Validate `myst.yml`, ensure chapters exist, ensure GitHub Pages deploy workflow exists, and ensure README explains the project.

### Batch 2: Core geometry

Complete chapters 1–5.

### Batch 3: Learning and transformations

Complete chapters 6–10.

### Batch 4: Embedding systems

Complete chapters 11–14.

### Batch 5: Retrieval

Complete chapters 15–18.

### Batch 6: Visualization, applications, advanced topics

Complete remaining chapters and appendices.

## Pull request expectations

Each PR should:

- update only a coherent chapter group
- include generated visuals if used
- mention whether `myst build --html` passes
- mention which notebooks were run
- keep examples small and reproducible
