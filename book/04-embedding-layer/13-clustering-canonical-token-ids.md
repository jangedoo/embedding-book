# Clustering and Canonical Token IDs

Instead of giving every token its own embedding row, we can map multiple tokens to the same canonical ID.

If tokens `a`, `b`, and `c` all map to ID `k`, then they receive exactly the same input embedding.

The model cannot distinguish them at the embedding lookup step.

For pure encoder models, this may be acceptable if the merged tokens are semantically or functionally redundant. For decoder language models, this is more dangerous because decoding expects a relationship between IDs and output tokens.
