# Hybrid Retrieval and Reranking

## Summary

Dense retrieval is powerful, but it is not enough by itself. It can miss exact terms, rare names, numbers, product IDs, citations, and short keyword-style queries. Lexical retrieval is strong on those cases but weaker on paraphrase and semantic intent.

Hybrid retrieval combines dense and lexical signals. Reranking then uses a stronger model to reorder a smaller candidate set.

## Intuition

Consider two queries:

1. `error E1137 in acme-sync`
2. `why does my vector search miss obvious paraphrases?`

The first query contains an exact error code and product name. A lexical method such as BM25 has a strong advantage because exact token overlap matters.

The second query is semantic. A dense model may retrieve passages about recall failure, embedding mismatch, chunking, or ANN tuning even if the wording differs.

Real systems receive both kinds of queries. Hybrid retrieval avoids betting everything on one geometry.

## Lexical retrieval

BM25 scores documents using term overlap with saturation and document-length normalization.

A common form is:

```{math}
\operatorname{BM25}(q, d) =
\sum_{t \in q}
\operatorname{IDF}(t)
\frac{f(t,d)(k_1 + 1)}
{f(t,d) + k_1(1 - b + b \frac{|d|}{\operatorname{avgdl}})}
```

where:

- `f(t,d)` is the frequency of term `t` in document `d`
- `IDF(t)` is larger for rarer terms
- `|d|` is document length
- `avgdl` is average document length
- `k_1` and `b` control saturation and length normalization

BM25 is strong for:

- names
- numbers
- rare words
- exact phrases
- error codes
- identifiers
- legal or technical terms

It is weaker when the query and document use different words for the same idea.

## Dense retrieval

Dense retrieval scores semantic similarity:

```{math}
s_{\text{dense}}(q, d) = f_{\text{query}}(q)^\top f_{\text{doc}}(d)
```

usually after normalization when cosine similarity is intended.

Dense retrieval is strong for:

- paraphrase
- fuzzy matching
- cross-lingual matching
- conceptual similarity
- intent matching
- semantically related context

It is weaker when the important signal is an exact token that the embedding model smooths away.

## Hybrid retrieval

Hybrid retrieval gathers candidates from both systems:

```{math}
C = C_{\text{dense}} \cup C_{\text{lexical}}
```

Then it combines, reranks, or filters them.

There are several common patterns:

- union candidates and rerank with a stronger model
- weighted score fusion
- reciprocal rank fusion
- dense first with lexical fallback
- lexical first with dense expansion
- separate indexes per query type

The right pattern depends on latency, labels, and the cost of reranking.

## Score fusion

Naively adding BM25 and dense scores is risky because they live on different scales.

A safer approach is to normalize scores per query:

```{math}
\tilde{s}_i = \frac{s_i - \min_j s_j}{\max_j s_j - \min_j s_j + \epsilon}
```

Then combine:

```{math}
s_{\text{hybrid}}(q,d) =
\alpha \tilde{s}_{\text{dense}}(q,d)
+ (1-\alpha)\tilde{s}_{\text{bm25}}(q,d)
```

This can work, but score normalization is sensitive to candidate sets and outliers.

Score fusion should be tuned on validation queries, not chosen by intuition. The weight `\alpha` may need different values for exact-ID queries, broad semantic questions, and no-answer queries. If query routing is unavailable, start with RRF or a simple candidate union before relying on calibrated score fusion.

## Reciprocal rank fusion

Reciprocal rank fusion avoids score calibration by combining ranks:

```{math}
\operatorname{RRF}(d) =
\sum_{r \in R}
\frac{1}{c + \operatorname{rank}_r(d)}
```

where `R` is the set of rankers and `c` is a constant such as `60`.

RRF rewards documents that appear highly in multiple lists, while still allowing a document from one strong list to survive.

It is a good default when BM25 scores and dense scores are not comparable.

## Reranking

A retriever scores each document mostly independently from a compact representation. A reranker can use a richer model on a smaller candidate set.

A cross-encoder reranker takes the query and document together:

```{math}
s_{\text{rerank}}(q,d) = h([q; d])
```

Because the model sees both texts jointly, it can inspect exact token matches, negation, phrase order, numeric details, and subtle relevance.

The cost is latency. If a reranker takes 8 ms per pair and you rerank 100 candidates sequentially, the request is too slow. Real systems batch reranking and keep candidate sets modest.

The latency budget is roughly:

```{math}
T_{\text{retrieval}} \approx
\max(T_{\text{dense}}, T_{\text{bm25}})
+ T_{\text{merge}}
+ T_{\text{rerank batch}}
+ T_{\text{context}}
```

Dense and lexical retrieval can often run in parallel, but reranking and context packing usually sit after candidate generation. This is why candidate count is both a quality knob and a latency knob.

## Candidate budgeting

Hybrid retrieval needs a candidate budget. For example:

- retrieve top 50 dense candidates
- retrieve top 50 BM25 candidates
- union and deduplicate
- rerank up to 80 candidates
- pass top 8 chunks to the generator

This budget controls recall, latency, and context quality.

If the candidate pool is too small, reranking cannot recover missing documents. If it is too large, reranking may dominate latency and cost.

The usual tuning curve is:

```{math}
\text{larger candidate pool} \Rightarrow \text{higher recall} \Rightarrow \text{higher latency}
```

## Implementation sketch: RRF

```python
from collections import defaultdict

def reciprocal_rank_fusion(rankings, c=60, return_scores=False):
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (c + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if return_scores:
        return ranked
    return [doc_id for doc_id, score in ranked]

dense_ranked = ["d2", "d7", "d1", "d9"]
bm25_ranked = ["d9", "d3", "d2", "d8"]

print(reciprocal_rank_fusion([dense_ranked, bm25_ranked]))
print(reciprocal_rank_fusion([dense_ranked, bm25_ranked], return_scores=True))
```

For a real system, keep the scores too. They are useful for debugging why a document was promoted.

## Implementation sketch: reranking loop

This pseudocode shows the shape of a small RAG retrieval stage:

```python
def retrieve_for_rag(query, dense_index, bm25_index, reranker, k_context=8):
    dense = dense_index.search(query, k=50)
    lexical = bm25_index.search(query, k=50)

    candidate_ids = reciprocal_rank_fusion([
        [doc_id for doc_id, score in dense],
        [doc_id for doc_id, score in lexical],
    ])[:80]

    pairs = [(query, load_text(doc_id)) for doc_id in candidate_ids]
    rerank_scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidate_ids, rerank_scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return ranked[:k_context]
```

This is not tied to a specific database. The same structure applies whether dense search uses FAISS, a hosted vector database, or a custom service.

## Query routing

Some systems route queries before retrieval.

Examples:

- exact identifier query: favor BM25 or exact lookup
- broad semantic question: favor dense retrieval
- filtered enterprise query: constrain by tenant and permissions first
- time-sensitive query: add freshness signals
- multilingual query: use dense retrieval with a multilingual encoder

Routing can improve quality and latency, but it introduces another model or heuristic that must be evaluated. Bad routing can be worse than simple hybrid retrieval.

## Real system interpretation

Hybrid retrieval is often the practical default for RAG and search because production queries are mixed.

The serving path usually looks like:

1. Run dense retrieval and BM25 retrieval in parallel.
2. Merge and deduplicate candidates.
3. Apply permissions and metadata constraints.
4. Fuse rankings or scores.
5. Rerank the top candidate pool.
6. Select diverse, non-duplicative chunks for the final context.

Each step changes the final ranking. Debug logs should preserve intermediate candidates so failures can be attributed to the dense retriever, lexical retriever, fusion, filters, reranker, or context packer.

Context packing is a retrieval step, not just prompt formatting. After reranking, the system still has to choose which chunks fit into the model context. Good context packing usually removes near-duplicates, respects source boundaries, preserves citations, and avoids filling the prompt with many chunks that all say the same thing.

For answer quality, the final context should be evaluated directly:

```{math}
\operatorname{ContextRecall@k} =
\mathbf{1}[\text{sufficient evidence appears in the packed context}]
```

This can differ from reranked Recall@k if long chunks, duplicate chunks, or citation rules push the best evidence out of the prompt.

## Evaluation

Evaluate hybrid systems by stage:

- dense-only metrics
- BM25-only metrics
- fused candidate recall
- reranked nDCG or MRR
- final context recall
- final answer accuracy and faithfulness
- latency percentiles

Also evaluate query groups separately:

- paraphrase questions
- exact names and IDs
- numeric questions
- rare entities
- ambiguous short queries
- long natural-language questions
- no-answer queries

Hybrid retrieval should improve the mixed workload without hiding regressions in one group.

Run ablations before shipping:

- dense only
- BM25 only
- dense plus BM25 union
- fused without reranking
- fused with reranking
- final packed context

These ablations show whether quality comes from better candidate recall, better ordering, or better context selection.

## Common failure modes

- Combining raw BM25 and dense scores without calibration.
- Reranking too few candidates and blaming the reranker for missing evidence.
- Reranking too many candidates and exceeding latency budgets.
- Applying permission filters after retrieval and losing most candidates.
- Deduplicating by exact text only, leaving near-duplicate chunks in the context.
- Letting BM25 dominate semantic queries because lexical scores have a larger numeric range.
- Letting dense retrieval miss exact IDs because token-level details are smoothed away.
- Evaluating only average metrics and missing rare-entity regressions.
- Keeping duplicate chunks after reranking and wasting context window space.
- Improving reranked metrics while the final context packer still drops the evidence.
- Shipping query routing rules without tracking misrouted queries.

## Visual idea

Draw two ranked lists for the same query: one dense and one BM25. Use different colors for semantic matches and exact-token matches. Then show a fused list and a reranked list. Mark which document finally enters the RAG context window.

For latency, draw a horizontal timeline with parallel dense and BM25 search, followed by merge, rerank, and context packing.

## Small experiment

Create six toy documents:

```python
docs = {
    "d1": "HNSW efSearch controls query-time recall and latency.",
    "d2": "HNSW efConstruction controls graph quality during indexing.",
    "d3": "Error E1137 occurs when acme-sync cannot refresh credentials.",
    "d4": "Credential refresh failures can stop synchronization jobs.",
    "d5": "Cosine similarity compares vector direction after normalization.",
    "d6": "Vector search can miss exact identifiers and rare codes.",
}
```

Use two queries:

```python
queries = [
    "what controls hnsw search recall at query time?",
    "E1137 acme-sync",
]
```

Manually create a dense ranking and a BM25 ranking for each query. Apply reciprocal rank fusion. Then decide which top 3 candidates you would rerank.

Questions to answer:

1. Which query benefits most from BM25?
2. Which query benefits most from dense retrieval?
3. Does RRF keep the relevant document in the candidate pool?
4. How many candidates must be reranked to recover the right result?

## Practical takeaways

- Dense retrieval and lexical retrieval fail in different ways.
- Hybrid retrieval is usually stronger than either method alone on mixed workloads.
- RRF is a robust default when scores are not calibrated.
- Rerankers improve ordering but cannot recover candidates that were never retrieved.
- Candidate budget is a core quality-latency tradeoff.
- Evaluate dense, lexical, fused, reranked, and final answer stages separately.
- Context packing can erase retrieval gains, so measure final context recall.
- Use ablations to locate whether a hybrid improvement comes from candidate recall, fusion, reranking, or packing.
