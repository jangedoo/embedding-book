# Visualizing Embeddings Without Fooling Yourself

Embedding visualizations are useful because they turn thousands of coordinates into something your eyes can inspect. They are dangerous because every 2D plot is a compressed story about a much larger space. Treat plots as debugging tools and hypothesis generators, not as proof that the model has learned the right structure.

PCA, t-SNE, and UMAP can reveal clusters, outliers, batch effects, topic gradients, and labeling mistakes. They can also invent apparent gaps, hide nearest neighbors, and make unrelated groups look cleanly separated.

Use 2D plots as hypotheses. Pair them with nearest-neighbor inspection, quantitative metrics, controlled synthetic examples, and stability checks across random seeds.

## Intuition

Imagine photographing a 3D sculpture from above. The image can reveal structure, but it cannot preserve every distance, angle, and occlusion. Visualizing a 768-dimensional embedding in 2D is a more extreme version of the same problem.

Different methods preserve different things:

- PCA preserves large-variance linear directions.
- t-SNE emphasizes local neighborhoods and often exaggerates separation.
- UMAP tries to preserve local graph structure and some global layout.

None of them preserves the full embedding space. The practical question is not "which plot is true?" but "what does this plot suggest, and can I verify it in the original space?"

## Mathematical object

A visualization method maps high-dimensional embeddings:

```{math}
X \in \mathbb{R}^{n \times d}
```

to low-dimensional coordinates:

```{math}
Z \in \mathbb{R}^{n \times 2}
```

PCA uses a linear projection:

```{math}
Z = X_c W_2
```

where `X_c` is centered and `W_2` contains the top two principal directions.

Nonlinear methods such as t-SNE and UMAP do not simply choose two axes. They build a new 2D arrangement that tries to preserve selected neighborhood relationships. That makes them useful for seeing local groups, but risky for reading exact distances.

## PyTorch and NumPy equivalent

PCA can be written directly with singular value decomposition:

```python
import torch

X = torch.randn(1000, 384)
Xc = X - X.mean(dim=0, keepdim=True)

U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
Z = Xc @ Vh[:2].T

print(Z.shape)  # torch.Size([1000, 2])
```

For large embedding sets, sample first. A plot with 100,000 points often hides more than it reveals.

For repeated analysis, keep the projection inputs explicit:

```python
sample = torch.randperm(X.shape[0])[:5000]
Z_sample = Z[sample]
```

Store the sampled IDs with the plot. Otherwise it becomes hard to explain why a cluster appeared in one run and disappeared in another.

## What this means in ML systems

Visualization is most useful when it answers operational questions:

- Are labels mixed in a way that suggests noisy supervision?
- Are near-duplicate documents collapsing together?
- Are language, source, or tenant IDs creating separate islands?
- Do outliers correspond to bad text extraction, empty fields, or unsupported languages?
- Does a new embedding model change the neighborhood structure?

For retrieval, inspect the original nearest neighbors of plotted points. A beautiful 2D cluster is not enough if the top-10 retrieved items are wrong.

A useful workflow is: plot, click or select suspicious points, inspect their source text or metadata, then compute the actual nearest neighbors with the production metric. The plot tells you where to look. The original-space neighbors tell you whether the model behavior is real.

## Common failure modes

- Reading t-SNE or UMAP distances as actual semantic distances.
- Interpreting cluster size as class frequency after nonlinear layout distortions.
- Forgetting that PCA emphasizes high-variance directions, not necessarily useful directions.
- Choosing a plot seed that looks clean and ignoring unstable alternatives.
- Plotting unnormalized vectors when the serving system uses normalized vectors.
- Coloring by a label that was indirectly present in preprocessing, then mistaking leakage for model understanding.

## Visual idea

```{image} ../../assets/figures/projection-dashboard-neighbor-edges.svg
:alt: Four projection panels comparing original points, PCA, and two UMAP seeds with nearest-neighbor edges overlaid.
:align: center
:width: 100%
```

This figure treats a projection as a diagnostic view, not as the embedding space itself. The neighbor edges come from the original high-dimensional geometry, while the panels show how PCA and two UMAP runs arrange the same points differently. Edges that cross long visual distances are a reminder that a clean 2D plot can still distort the neighborhoods used by retrieval, clustering, or duplicate detection.

The dashboard layout also shows why small multiples are often more useful than one overloaded scatter plot. Separate views for labels, source, language, timestamp, and vector norm make confounders visible without forcing every meaning into a single color or marker scheme.

## Small experiment

Create three Gaussian clusters in 50 dimensions. Add one high-variance nuisance coordinate that is unrelated to the labels. Plot PCA before and after removing that coordinate. Then compare nearest-neighbor label purity in the original space. This shows why high visual separation is not the same as useful retrieval geometry.

Repeat the experiment with normalized vectors and cosine similarity. If the nuisance coordinate mostly changes vector length, normalization can change both the plot and the retrieval result.

## Practical takeaways

Use visualization to find questions, not to close them.

Before trusting a plot, check:

1. What preprocessing was applied?
2. Which metric does the real system use?
3. Are nearest neighbors in the original space consistent with the visual story?
4. Does the pattern survive different seeds, samples, and projection methods?
