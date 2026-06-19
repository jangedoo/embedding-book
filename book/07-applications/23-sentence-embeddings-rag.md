# Sentence Embeddings for RAG

RAG systems depend heavily on embedding quality because retrieval decides what evidence the language model sees. A strong generator cannot answer from documents that were never retrieved.

Sentence embeddings turn queries and text chunks into vectors that can be searched quickly. The hard part is not only making similar text close. It is making useful evidence close to the query under the exact metric, chunking scheme, and index used in production.

## Intuition

A RAG pipeline is a relay:

1. split documents into chunks
2. embed chunks
3. embed the query
4. retrieve nearest chunks
5. optionally rerank them
6. pass selected context to the generator

Every step changes what "relevant" means. A chunk can contain the answer but be too long and diluted. A query can be semantically close to a topic but miss the exact fact. A reranker can rescue some ordering mistakes but cannot rerank documents that retrieval never returned.

## Mathematical object

A document encoder maps chunks to vectors:

```{math}
d_i = f_{doc}(chunk_i)
```

A query encoder maps a question to a vector:

```{math}
q = f_{query}(query)
```

Retrieval ranks chunks by similarity:

```{math}
score(q, d_i) = \cos(q, d_i)
```

or by inner product:

```{math}
score(q, d_i) = q^\top d_i
```

If vectors are normalized, cosine search can be implemented as inner-product search:

```{math}
\hat{q}^\top \hat{d_i} = \cos(q, d_i)
```

## PyTorch equivalent

```python
import torch
import torch.nn.functional as F

query = torch.randn(384)
docs = torch.randn(10_000, 384)

query = F.normalize(query, dim=0)
docs = F.normalize(docs, dim=-1)

scores = docs @ query
values, indices = scores.topk(k=5)
```

Pooling also matters. A simple transformer encoder may output token embeddings:

```{math}
H \in \mathbb{R}^{L \times d}
```

Mean pooling gives:

```{math}
s = \frac{1}{L}\sum_{t=1}^{L} H_t
```

Special-token pooling uses one selected token vector. The right choice depends on the training objective of the embedding model.

If the embedding model expects prompts, treat them as part of the model contract:

```text
query: how do I reset my password?
passage: To reset your password, open Settings...
```

Changing these prefixes can change the geometry enough to require re-indexing and re-evaluation.

## What this means in ML systems

Important RAG choices:

- chunk size and overlap
- document metadata included in the embedded text
- query and document prompts
- pooling strategy
- normalization and metric
- ANN index type and search parameters
- reranker model and cutoff
- evaluation set with real user questions

Cross-lingual alignment is hard. Romanized language, mixed scripts, transliteration, and domain-specific vocabulary may require targeted training or separate routing.

A useful RAG evaluation separates retrieval from generation. Measure whether the right chunk appears in top `k` before asking whether the final answer is good.

Keep failed retrieval cases as artifacts. They often reveal whether the system needs better chunking, query rewriting, metadata filters, hybrid lexical search, or a reranker.

## Common failure modes

- Chunks are too large, so answer-bearing text is diluted.
- Chunks are too small, so necessary context is split apart.
- Query and document embeddings are generated with inconsistent prompts.
- Documents are normalized but queries are not, or the reverse.
- ANN settings trade away too much recall for latency.
- Evaluation uses answer quality only, hiding retrieval misses.
- Fresh documents are indexed with a different model version.

## Visual idea

```{image} ../../assets/figures/rag-context-budget.svg
:alt: Ranked RAG retrieval list feeding a fixed context budget with answer-bearing chunks, near misses, distractors, and metric cutoffs.
:align: center
:width: 100%
```

This figure shows retrieval as a ranked list that must eventually fit inside a generator's context window. The first answer-bearing chunk, near miss, and lexical distractor can lead to different conclusions depending on whether you inspect recall@k, MRR, reranking cutoff, or final answer quality. A system can look strong by one metric while still placing the useful evidence too late to be used.

The fixed context budget is the main operational constraint. Increasing `top_k` may raise retrieval recall but can crowd out the best chunks, inflate latency, and leave fewer tokens for the model's answer. Good RAG evaluation therefore measures both whether evidence was retrieved and whether the right evidence survived packing into the prompt.

## Small experiment

Build a 50-question evaluation set over a small documentation corpus. For each question, label one or more answer chunks. Sweep chunk size, overlap, normalization, and top-k. Plot recall@5 and average context tokens sent to the generator. This exposes the quality-cost tradeoff directly.

Add one run with exact search and one with the production ANN settings. The difference between them is the recall cost of approximation.

## Practical takeaways

RAG quality is retrieval quality plus context assembly plus generation.

Start with a small labeled retrieval set. Track recall@k, MRR, latency, index size, and answer faithfulness separately so you know which part of the pipeline changed.
