# High-Dimensional Geometry

High-dimensional spaces behave differently from 2D intuition.

## Random vectors become nearly orthogonal

As dimension grows, random vectors tend to have cosine similarity near zero.

## Distance concentration

Distances can become less distinguishable in high dimensions. This makes clustering and nearest-neighbor search harder.

## Hubness

Some points become nearest neighbors of many other points. These are hubs.

## Practical implications

- DBSCAN `eps` is difficult to choose in high dimensions.
- UMAP/t-SNE plots can be visually persuasive but misleading.
- Nearest-neighbor inspection is often more useful than only plotting.
- Centering and whitening can sometimes help, but may remove useful signal.
