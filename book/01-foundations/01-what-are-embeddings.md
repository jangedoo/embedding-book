# What Are Embeddings?

An embedding is a learned coordinate for something discrete or complex: a token, word, sentence, product, user, image, entity, or document.

The useful question is not only "what vector represents this object?" but also:

- what training objective created the vector?
- what transformations will act on it?
- what distance or similarity function will compare it?
- what system will store and retrieve it?
- what downstream decision will use it?

## Intuition

An ID such as `token_id = 42` has no geometry. It is just a label. An embedding table gives that ID a location:

```python
x = embedding[token_id]
```

Now the object can participate in geometry. It can be close to other objects, far away from others, projected, transformed, normalized, clustered, quantized, or retrieved.

## Core interpretation

An embedding is a learned interface between symbolic identity and continuous computation.
