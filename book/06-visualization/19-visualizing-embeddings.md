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

```math
X \in \mathbb{R}^{n \times d}
```

to low-dimensional coordinates:

```math
Z \in \mathbb{R}^{n \times 2}
```

PCA uses a linear projection:

```math
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

## What this means in ML systems

Visualization is most useful when it answers operational questions:

- Are labels mixed in a way that suggests noisy supervision?
- Are near-duplicate documents collapsing together?
- Are language, source, or tenant IDs creating separate islands?
- Do outliers correspond to bad text extraction, empty fields, or unsupported languages?
- Does a new embedding model change the neighborhood structure?

For retrieval, inspect the original nearest neighbors of plotted points. A beautiful 2D cluster is not enough if the top-10 retrieved items are wrong.

## Common failure modes

- Reading t-SNE or UMAP distances as actual semantic distances.
- Interpreting cluster size as class frequency after nonlinear layout distortions.
- Forgetting that PCA emphasizes high-variance directions, not necessarily useful directions.
- Choosing a plot seed that looks clean and ignoring unstable alternatives.
- Plotting unnormalized vectors when the serving system uses normalized vectors.
- Coloring by a label that was indirectly present in preprocessing, then mistaking leakage for model understanding.

## Visual idea

Show the same synthetic dataset in four panels: original 2D points, PCA, UMAP with one seed, and UMAP with another seed. Draw a few nearest-neighbor edges in the original space over the projections. The important visual message is that point layout and true neighborhoods are related but not identical.

## Small experiment

Create three Gaussian clusters in 50 dimensions. Add one high-variance nuisance coordinate that is unrelated to the labels. Plot PCA before and after removing that coordinate. Then compare nearest-neighbor label purity in the original space. This shows why high visual separation is not the same as useful retrieval geometry.

## Practical takeaways

Use visualization to find questions, not to close them.

Before trusting a plot, check:

1. What preprocessing was applied?
2. Which metric does the real system use?
3. Are nearest neighbors in the original space consistent with the visual story?
4. Does the pattern survive different seeds, samples, and projection methods?
