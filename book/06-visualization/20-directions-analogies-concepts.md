# Directions, Analogies, and Concepts

A direction in embedding space can sometimes represent a concept.

For a concept direction `v`, projection asks how much another embedding points along that concept:

```math
score(x) = x \cdot v
```

Not every concept is linear. Not every direction is meaningful. Always validate with examples.
