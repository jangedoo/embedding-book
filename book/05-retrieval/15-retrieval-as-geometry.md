# Retrieval as Geometry

## Summary

Dense retrieval turns search into nearest-neighbor geometry. A query becomes a vector, each document or chunk becomes a vector, and ranking becomes a question about which candidates are closest or most aligned with the query.

This chapter explains retrieval as a geometric system: what is scored, how normalization changes rankings, why query and document embeddings are not always symmetric, and how these choices show up in real RAG systems.

## Intuition

Imagine a user asks:

> How do I reduce vector database latency without losing too much recall?

A retrieval system embeds that question into a point in space. It also embeds candidate passages into the same search space. Good passages should land near the query according to the score used by the system.

But "near" is not automatic. It depends on:

- the encoder
- the metric
- whether vectors are normalized
- the chunking strategy
- whether the query and document encoders were trained symmetrically
- how many candidates the system retrieves before reranking or generation

Retrieval quality is therefore not just "embedding quality." It is the behavior of the whole geometry.

## The retrieval object

Let a corpus contain `N` documents or chunks. A document encoder maps each item to:

```{math}
x_i = f_{\text{doc}}(d_i), \quad x_i \in \mathbb{R}^D
```

A query encoder maps a user query to:

```{math}
q = f_{\text{query}}(u), \quad q \in \mathbb{R}^D
```

Retrieval ranks candidates by a scoring function:

```{math}
s(q, x_i)
```

Common choices are dot product:

```{math}
s(q, x_i) = q^\top x_i
```

cosine similarity:

```{math}
s(q, x_i) = \frac{q^\top x_i}{\|q\|_2\|x_i\|_2}
```

or negative squared Euclidean distance:

```{math}
s(q, x_i) = -\|q - x_i\|_2^2
```

The top-`k` retrieval set is:

```{math}
\operatorname{TopK}(q) = \underset{i}{\operatorname{topk}}\; s(q, x_i)
```

This is the mathematical object behind a vector search API call.

## PyTorch sketch

For a small corpus, exact dense retrieval is just matrix multiplication:

```python
import torch
import torch.nn.functional as F

docs = torch.tensor([
    [1.0, 0.0],
    [0.8, 0.2],
    [0.0, 1.0],
    [3.0, 0.1],
])
query = torch.tensor([1.0, 0.0])

# Cosine retrieval implemented as inner product on normalized vectors.
docs_n = F.normalize(docs, dim=-1)
query_n = F.normalize(query, dim=-1)

scores = query_n @ docs_n.T
values, indices = scores.topk(k=3)

print(indices.tolist())
print(values.tolist())
```

The same code scales conceptually to millions of vectors. The difference is that real systems use approximate indexes, batching, caching, compression, filters, and reranking.

## Dot product, cosine, and length

Dot product can be decomposed as:

```{math}
q^\top x = \|q\|_2\|x\|_2\cos(\theta)
```

For a fixed query, dot product rewards both direction and document vector length. Cosine removes length and ranks by angle.

This matters in production:

- If vector length represents confidence, specificity, or popularity, dot product can use that signal.
- If vector length mostly reflects artifacts such as document length, frequency, or encoder instability, dot product can hurt retrieval.
- If the embedding model was trained with normalized contrastive loss, cosine is usually a better first choice.

When both queries and documents are L2-normalized:

```{math}
\|q - x\|_2^2 = 2 - 2q^\top x
```

So Euclidean distance, squared Euclidean distance, cosine similarity, and inner product produce the same ranking up to monotonic transformations.

The practical rule is simple:

> Normalize both sides or neither side deliberately. Do not accidentally normalize only queries or only documents.

## Score distributions and thresholds

Retrieval scores are useful for ranking, but they are not automatically calibrated probabilities. A cosine score of `0.78` can mean different things across embedding models, domains, chunk sizes, and query types.

This matters for systems that must decide whether to answer:

```{math}
\max_i s(q, x_i) \ge \tau
```

The threshold `\tau` should be tuned on labeled validation queries, including no-answer queries. After changing the embedding model, prompt, normalization, chunking, or index metric, retune the threshold instead of carrying it forward.

Common threshold mistakes:

- treating cosine scores as comparable across models
- setting a threshold from a few successful examples
- ignoring that short keyword queries and long natural-language queries can have different score distributions
- thresholding before metadata filters, then accepting weak filtered results
- thresholding after reranking without validating the reranker score scale

## Query and document spaces

Many retrieval models use different prompts or encoders for queries and documents:

```{math}
q = f_{\text{query}}(p_q(u))
```

```{math}
x = f_{\text{doc}}(p_d(d))
```

Here `p_q` and `p_d` are prompt templates such as adding a query prefix or a passage prefix before encoding.

The vectors still live in the same dimension, but the distributions can differ. Queries are often short, task-like, and underspecified. Documents are longer, more factual, and contain extra context.

This asymmetry is useful. It lets the model learn that "vector database latency" in a query should match passages about HNSW parameters, cache behavior, ANN recall, and memory bandwidth, even if the words are not identical.

It also creates failure modes. If you embed documents with the query prompt, or queries with the document prompt, results may degrade while still looking plausible.

## Retrieval as a RAG bottleneck

In retrieval-augmented generation, dense retrieval is usually the first hard bottleneck:

```{math}
\text{answer} = g(\text{query}, \operatorname{TopK}(q))
```

The generator can only use evidence that retrieval provides. If the needed passage is absent from the context window, the generator may guess, overgeneralize, or cite the wrong source.

This makes top-`k` recall more important than the first rank alone. A reranker or generator can often recover from a merely imperfect order, but it cannot recover from a missing candidate.

In real RAG systems, retrieval quality depends on:

- chunk size and overlap
- metadata filters
- embedding model domain fit
- normalization and metric compatibility
- number of candidates retrieved
- reranker strength
- context assembly and deduplication

The embedding space is only one layer in that pipeline, but errors there propagate downstream.

For RAG, a useful retrieval target is often evidence recall rather than answer accuracy alone:

```{math}
\operatorname{EvidenceRecall@k}(q) =
\mathbf{1}[\text{at least one sufficient evidence chunk appears in top } k]
```

This metric asks whether retrieval gave the rest of the system a fair chance. Final answer accuracy still matters, but it should not be the only signal.

## Chunk geometry

Documents are often split into chunks before embedding. This changes the geometry.

Small chunks:

- make vectors more focused
- reduce topic mixing
- improve exact evidence retrieval
- may lose surrounding context

Large chunks:

- preserve more context
- reduce index size
- can mix unrelated topics into one vector
- may rank for the wrong part of the chunk

If a chunk contains two unrelated sections, its embedding can land between them. A query for either section may retrieve it, but the score may be weaker than if each section had its own vector.

The retrieval unit should match the evidence unit. If the answer needs one paragraph, paragraph-like chunks are often easier to rank than whole pages.

## Implementation sketch: exact top-k

This small function is enough to test retrieval behavior before adding a vector database:

```python
import torch
import torch.nn.functional as F

def dense_topk(query, docs, k=5, metric="cosine"):
    if metric == "cosine":
        query = F.normalize(query, dim=-1)
        docs = F.normalize(docs, dim=-1)
        scores = query @ docs.T
        return scores.topk(k)

    if metric == "dot":
        scores = query @ docs.T
        return scores.topk(k)

    if metric == "l2":
        dist2 = torch.cdist(query[None, :], docs, p=2).squeeze(0).square()
        values, indices = (-dist2).topk(k)
        return values, indices

    raise ValueError(metric)
```

For a small validation set, exact top-`k` is valuable because it separates model and metric behavior from ANN index behavior. If exact search is poor, an ANN index will not fix the embedding space.

## Real system interpretation

A production retrieval service usually has these stages:

1. Encode documents offline.
2. Normalize vectors if the metric requires it.
3. Store vectors plus metadata in an index.
4. Encode the query online.
5. Apply the same normalization and prompt convention.
6. Retrieve top candidates under the configured metric.
7. Apply filters, deduplication, reranking, or context packing.

Latency is dominated by online steps: query encoding, index search, metadata lookup, reranking, and generator context construction. Recall is dominated by model quality, chunking, metric choice, index settings, and candidate count.

A useful debugging split is:

- exact retrieval quality
- approximate retrieval quality
- final answer quality

These are related, but not the same metric.

A practical trace for one query should record:

- query text, prompt, embedding model version, and query vector norm
- candidate IDs and scores before filters
- candidate IDs removed by filters or permissions
- reranker input candidates and reranker scores
- chunks finally placed into the context window
- whether the answer evidence was present at each stage

Without this trace, many RAG failures look like "the model hallucinated" even when the root cause was missing or discarded evidence.

## Latency, recall, and memory

Dense retrieval makes three budgets visible:

- Latency: query encoding, ANN search, metadata fetch, reranking, and context packing must fit inside the request budget.
- Recall: enough candidates must survive search and filtering for the answer evidence to reach the reranker or generator.
- Memory: vector dimension, dtype, index overhead, replicas, and metadata determine serving cost.

Increasing `k` can improve recall but also increases metadata reads, reranker work, context deduplication, and prompt construction. Reducing vector dimension can improve memory and latency but may reduce separability. Normalization can stabilize ranking, but it also removes vector length as a possible confidence signal.

Treat these as coupled system parameters, not independent knobs.

## Common failure modes

- Training with cosine-like objectives but serving raw dot product.
- Normalizing documents during indexing but forgetting to normalize queries.
- Using a query prompt for documents or a document prompt for queries.
- Retrieving chunks that are too large and semantically mixed.
- Using top-`k` values that are too small for the reranker or generator.
- Judging retrieval only by generated answers instead of checking whether the evidence was retrieved.
- Assuming high average quality means rare entities, numbers, and exact phrases are handled well.
- Filtering after retrieval when the filter should have been applied before or during retrieval.
- Carrying similarity thresholds across embedding model, prompt, or normalization changes.
- Optimizing vector search latency while ignoring metadata fetch and reranker latency.

## Visual idea

```{image} ../../assets/figures/retrieval-rag-pipeline.svg
:alt: Retrieval-augmented generation pipeline showing a query embedding, nearest retrieved chunks, reranking, context packing, and answer generation.
:align: center
:width: 100%
```

The figure connects the geometry of retrieval to the serving path of a RAG system. The query is first embedded into the same vector space as the corpus chunks, then nearest-neighbor search selects a candidate set. Those candidates are not the final answer; they are evidence offered to later stages such as reranking, context packing, and generation.

The key lesson is that retrieval is an information bottleneck. If the evidence chunk never appears in the retrieved candidate set, the reranker and generator usually cannot recover it. This is why debugging RAG requires looking at intermediate candidates, scores, filters, and packed context rather than only judging the final answer text.

## Small experiment

Create a toy corpus with four documents:

```python
docs = [
    "HNSW improves vector search latency by traversing a graph.",
    "Recall@k measures whether relevant documents appear in the top k.",
    "Cosine similarity ranks normalized vectors by angle.",
    "GPU memory limits the batch size used during training.",
]
```

Manually assign simple 2D vectors where the first three are near a "retrieval" direction and the last is near a "training" direction. Rank with cosine and dot product. Then multiply one irrelevant vector by a large scalar and observe that dot product can promote it while cosine does not.

Questions to answer:

1. Which metric is stable under length changes?
2. Which metric uses length as a signal?
3. Does normalization change the top result?
4. Which behavior matches the embedding model's training objective?

## Practical takeaways

- Retrieval is nearest-neighbor geometry plus system design.
- The metric is part of the model's behavior.
- Normalization changes whether vector length can affect ranking.
- Query and document prompts must match training and serving conventions.
- In RAG, missing evidence is much harder to recover from than imperfect ordering.
- Exact search on a small dataset is the best first debugging tool before tuning ANN indexes.
- Similarity thresholds need labeled no-answer queries and should be recalibrated after retrieval changes.
- Latency, recall, and memory are coupled budgets in production retrieval.
