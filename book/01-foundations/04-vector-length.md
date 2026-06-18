# What Does Vector Length Mean?

Vector length is often ignored, but it can carry important information.

A vector has direction and length. Cosine keeps direction and discards length. Dot product keeps both. Euclidean distance is affected by both.

## Possible meanings of norm

Vector norm may represent frequency, confidence, popularity, training stability, artifact of optimization, magnitude of evidence, or nothing useful at all.

There is no universal interpretation. The interpretation depends on the model and objective.

## Practical rule

Do not normalize blindly. Compare raw dot product, cosine similarity, Euclidean distance on raw vectors, and Euclidean distance on normalized vectors.
