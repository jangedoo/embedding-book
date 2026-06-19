# Clustering and Canonical Token IDs

## Summary

Another way to shrink an embedding table is to map many original tokens to fewer canonical IDs. Instead of giving every token its own row, similar or low-value tokens share a row. This can save memory, but it changes what the model can distinguish.

The central question is not only "which tokens are similar?" It is "which distinctions can this system afford to forget?"

## Intuition

Suppose these original tokens:

```text
colour, color, colours
```

all map to one canonical ID:

```text
colour -> 17
color  -> 17
colours -> 17
```

They now receive exactly the same input embedding. At the lookup step, the model cannot tell which spelling appeared.

This can be acceptable if the downstream task only needs the broad meaning. It can be harmful if the exact surface form matters.

Canonicalization is therefore not just compression. It is an information-removal step before the model has a chance to reason about the input.

## The mapping

Let the original vocabulary size be `V` and the canonical vocabulary size be `K`, with:

```{math}
K < V
```

Define a mapping:

```{math}
c: \{0, 1, \ldots, V-1\} \rightarrow \{0, 1, \ldots, K-1\}
```

The canonical embedding table is:

```{math}
E_c \in \mathbb{R}^{K \times d}
```

Original token `i` uses row:

```{math}
E_c[c(i)]
```

If two tokens `i` and `j` share a canonical ID:

```{math}
c(i) = c(j)
```

then:

```{math}
E_c[c(i)] = E_c[c(j)]
```

They are indistinguishable at the embedding lookup.

## PyTorch sketch

```python
import torch
from torch import nn

V = 50_000
K = 20_000
d = 768

canonical_of_token = torch.randint(0, K, (V,))
emb = nn.Embedding(K, d)

token_ids = torch.tensor([10, 11, 12, 2048])
canonical_ids = canonical_of_token[token_ids]
x = emb(canonical_ids)

print(canonical_ids)
print(x.shape)
```

The model receives `canonical_ids`, not the original IDs.

## Numpy equivalent

```python
import numpy as np

V = 50_000
K = 20_000
d = 768

canonical_of_token = np.random.randint(0, K, size=V)
E = np.random.randn(K, d).astype("float32")

token_ids = np.array([10, 11, 12, 2048])
x = E[canonical_of_token[token_ids]]
```

The table has `Kd` parameters instead of `Vd`.

The mapping itself also has a cost. A dense array from original token ID to canonical ID stores `V` integers. This is usually small compared with the embedding table, but it should be included in deployment accounting for very large vocabularies or many feature namespaces.

## How canonical IDs are chosen

The mapping can come from several sources:

- text normalization, such as lowercasing or Unicode normalization
- stemming or lemmatization
- frequency bucketing for rare tokens
- clustering existing token embeddings
- hashing tokens into buckets
- hand-built domain rules

Each method makes a different promise. Lowercasing says case distinctions are not important. Stemming says inflectional differences are not important. Clustering says nearby learned vectors can share a row. Hashing says collisions are acceptable noise.

Those promises should be checked against the task. Lowercasing may be harmless for broad semantic retrieval and harmful for named entities, passwords, code, or biomedical symbols. Hashing can be reasonable for very sparse recommender features and dangerous when collisions merge high-value identifiers.

## Pure encoders versus decoders

Canonical IDs are usually less problematic for pure encoders than for decoders.

A pure encoder maps an input sequence to representations for classification, retrieval, ranking, tagging, or scoring. If two tokens share a canonical ID, the encoder loses that distinction at the input layer, but the output may not require reconstructing the exact token.

For example, a sentence embedding model may treat "color" and "colour" as equivalent for semantic retrieval. If retrieval quality is unchanged, the merge may be a good tradeoff.

Even for encoders, the merge is not automatically safe. A classifier that distinguishes programming languages may need `public`, `Public`, and `PUBLIC` to behave differently. A biomedical encoder may need two visually similar symbols to remain separate because they point to different entities.

A decoder language model has a harder problem. It must produce token IDs that correspond to output text. If multiple original tokens map to one canonical ID, the model may know the meaning but not which exact token to emit.

This creates decoding ambiguity.

## Decoding ambiguity

If:

```text
token_a -> canonical_7
token_b -> canonical_7
token_c -> canonical_7
```

then an input lookup through `canonical_7` cannot preserve which original token occurred.

For generation, an output ID must eventually map back to text. If the model predicts `canonical_7`, which original token should be emitted?

Possible choices include:

- always emit the most frequent original token
- emit a representative token
- use a secondary model to choose among aliases
- keep canonicalization only on the input side
- avoid many-to-one mappings for decoder vocabularies

Each choice has costs. Always emitting the most frequent token can erase spelling, morphology, names, code symbols, or domain-specific vocabulary. A secondary disambiguation model adds complexity and can still fail.

This is why canonical IDs are much safer for encoders, retrieval models, and classifiers than for open-ended generative decoders.

The problem is especially sharp for byte-pair or unigram tokenizers because output text is assembled from exact token strings. If canonicalization collapses multiple output pieces, the decoder may produce fluent text with the wrong spelling, identifier, or segmentation. For code generation, math, and structured data, that can turn a small embedding memory saving into a correctness bug.

## Retrieval systems

In retrieval, canonical IDs can appear in two places.

First, the encoder's token vocabulary may be compressed. This changes the query and document embeddings produced by the model.

Second, stored entities or documents may be clustered so multiple items share a representation. This is more aggressive: different retrievable objects can collapse to the same vector or same posting group.

Both choices should be evaluated by ranking metrics, not just memory reduction.

Token-level canonicalization and document-level clustering fail differently. Token canonicalization changes the encoder and can move every query and document vector. Document clustering keeps the encoder fixed but collapses retrievable objects. The first is a model-quality question; the second is an index-quality and result-diversity question.

## Evaluating retrieval degradation

Start with a baseline model and index. Then introduce canonical IDs and compare retrieval outputs on the same queries.

Useful metrics include:

- recall@k
- precision@k
- MRR
- nDCG
- mean reciprocal rank for known-answer queries
- top-k overlap with the baseline
- latency and memory footprint
- per-slice recall for rare terms, names, languages, code tokens, or tail items
- disagreement cases where the compressed system retrieves a different relevant-looking but wrong item

The important comparison is against task labels when available. Baseline overlap is useful for debugging, but it can preserve baseline mistakes.

A practical evaluation should include both aggregate metrics and error slices. Canonical IDs often look fine on frequent, easy examples while damaging rare entities, spelling-sensitive queries, or tail catalog items. If the compression target is motivated by rare-token memory, those are exactly the cases to inspect.

## Evaluation sketch

```python
import torch
import torch.nn.functional as F

def recall_at_k(scores, relevant, k):
    topk = scores.topk(k, dim=-1).indices
    hits = (topk == relevant[:, None]).any(dim=-1)
    return hits.float().mean()

queries = F.normalize(torch.randn(100, 128), dim=-1)
docs_full = F.normalize(torch.randn(1000, 128), dim=-1)
docs_compressed = docs_full.clone()

# Simulate damage from merging by forcing groups of docs to share vectors.
groups = torch.arange(1000) // 4
for g in groups.unique():
    docs_compressed[groups == g] = docs_full[groups == g].mean(dim=0)
docs_compressed = F.normalize(docs_compressed, dim=-1)

relevant = torch.randint(0, 1000, (100,))

full_scores = queries @ docs_full.T
compressed_scores = queries @ docs_compressed.T

print(recall_at_k(full_scores, relevant, k=10))
print(recall_at_k(compressed_scores, relevant, k=10))
```

This toy example does not prove a compression method is good. It gives the shape of the test: hold queries and relevance labels fixed, then measure how much ranking quality changes.

For a real benchmark, keep the train, validation, and test splits fixed. Train or export the compressed system, rebuild the index if document vectors changed, and compare metrics with the same candidate corpus and same query set. If the compressed model changes latency or memory enough to alter index settings, report those settings explicitly.

## What this means in real ML systems

Canonical IDs are a controlled loss of information.

They can help when:

- many tokens are near-duplicates
- rare tokens receive poor individual estimates
- memory or latency limits dominate
- exact surface form is not needed
- the model is an encoder used for classification or retrieval
- the deployment budget is dominated by row count rather than dimension

They are risky when:

- exact spelling, casing, morphology, or symbols matter
- the model generates text
- the system must preserve names, code, formulas, or identifiers
- merged tokens have different labels or user intent
- retrieval depends on fine-grained distinctions
- legal, medical, financial, or security workflows require exact strings

## Visual idea

```{image} ../../assets/figures/canonical-token-id-collapse.svg
:alt: Many original token IDs collapse into fewer canonical IDs, which select shared rows in a smaller embedding table.
:align: center
:width: 100%
```

Many-to-one canonicalization saves rows by sending several surface forms to the same embedding. That is usually easier to tolerate in encoders than decoders, because a decoder must choose which original surface form to emit from an ambiguous canonical ID.

## Small experiment

Take a small text classification or retrieval dataset. Build three token mappings:

1. original tokens
2. lowercased tokens
3. clustered or frequency-bucketed rare tokens

Train the same encoder with each mapping. Compare validation quality, embedding table memory, and error examples. Look specifically for examples where the compressed model fails because two merged tokens should have been distinct.

For retrieval, run the same experiment with fixed relevance labels:

```python
def topk_overlap(a, b, k):
    a_top = a.topk(k, dim=-1).indices
    b_top = b.topk(k, dim=-1).indices
    overlaps = []
    for x, y in zip(a_top, b_top):
        overlaps.append(len(set(x.tolist()) & set(y.tolist())) / k)
    return sum(overlaps) / len(overlaps)
```

Use overlap to find changed queries, then inspect those queries manually. The goal is not to maximize overlap with the old model; it is to understand whether changed results are acceptable.

## Common failure modes

- Merging tokens because their strings look similar but their meanings differ.
- Merging tokens because their embeddings are close before checking downstream labels.
- Using many-to-one canonical IDs in a decoder without a plan for output ambiguity.
- Evaluating only average quality and missing rare-token regressions.
- Measuring memory savings but not retrieval recall.
- Letting hash collisions merge important identifiers.
- Forgetting that the model cannot recover distinctions removed before the embedding lookup.
- Evaluating only head queries while the compression mostly affects tail tokens.
- Reusing a canonical mapping from one domain in another domain where the aliases have different meanings.
- Compressing document IDs or item IDs so aggressively that diversity collapses in the ranked list.

## Practical takeaways

- Canonical IDs reduce row count from `V` to `K`.
- Many original tokens can map to one embedding row.
- Shared IDs mean shared input embeddings.
- This is often acceptable for encoders when the lost distinction is irrelevant.
- It is dangerous for decoders because generation requires choosing exact output tokens.
- Retrieval compression should be evaluated with recall@k, MRR, nDCG, and error analysis.
- Canonicalization is a memory-quality tradeoff, not a harmless preprocessing step.
- Once a distinction is removed by the canonical mapping, later layers cannot recover it from the embedding alone.
- Evaluate rare-token and exact-string slices separately from aggregate quality.
