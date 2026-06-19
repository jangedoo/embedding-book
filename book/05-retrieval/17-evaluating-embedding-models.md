# Evaluating Embedding Models

## Summary

Retrieval evaluation is ranking evaluation. An embedding model is useful when it puts relevant items above irrelevant items for the queries that matter.

This chapter explains ranking metrics, dataset construction, offline versus online evaluation, and common ways embedding evaluations become misleading.

## Intuition

A retrieval model does not produce a class label. It produces an ordered list.

For a query like:

> What parameter controls HNSW query-time recall?

the model should rank passages about `efSearch` above passages about unrelated vector database settings. If the right passage appears at rank 1, the user or RAG system can use it immediately. If it appears at rank 50, the answer may be missed unless the system retrieves many candidates and reranks them.

Evaluation asks two related questions:

1. Did we retrieve the relevant item at all?
2. Did we rank it high enough to be useful?

## The evaluation object

Let a query set be:

```{math}
Q = \{q_1, q_2, \ldots, q_m\}
```

For each query `q`, suppose we have relevance labels:

```{math}
rel(q, d) \in \{0, 1, 2, \ldots\}
```

Binary labels use `0` for irrelevant and `1` for relevant. Graded labels can represent partially relevant, relevant, and highly relevant documents.

A retrieval system returns a ranked list:

```{math}
\pi_q = [d_1, d_2, \ldots, d_k]
```

Metrics summarize how much relevance appears near the top of that list.

## Recall@k

Recall@k asks whether the relevant documents were retrieved in the first `k` results:

```{math}
\operatorname{Recall@k}(q) =
\frac{|\operatorname{Rel}(q) \cap \operatorname{TopK}(q)|}
{|\operatorname{Rel}(q)|}
```

Recall@k is important for RAG because downstream stages cannot use documents that were never retrieved.

If each query has one known relevant document, Recall@k becomes a hit rate:

```{math}
\operatorname{Hit@k}(q) =
\begin{cases}
1 & \text{if a relevant document appears in top } k \\
0 & \text{otherwise}
\end{cases}
```

Recall@k does not care whether the relevant document is rank 1 or rank `k`.

## Precision@k

Precision@k asks what fraction of the top `k` results are relevant:

```{math}
\operatorname{Precision@k}(q) =
\frac{|\operatorname{Rel}(q) \cap \operatorname{TopK}(q)|}{k}
```

Precision matters when users inspect results directly. In RAG, precision also matters because irrelevant chunks consume context window space and can distract the generator.

## MRR

Mean reciprocal rank focuses on the first relevant result.

For one query:

```{math}
\operatorname{RR}(q) = \frac{1}{\operatorname{rank}_q}
```

where `rank_q` is the rank of the first relevant result. If no relevant result is found, reciprocal rank is `0`.

Across queries:

```{math}
\operatorname{MRR} = \frac{1}{|Q|}\sum_{q \in Q}\operatorname{RR}(q)
```

MRR is useful when one good result is enough, such as FAQ retrieval or known-answer lookup.

## nDCG

Normalized discounted cumulative gain supports graded relevance and rewards high ranks:

```{math}
\operatorname{DCG@k}(q) =
\sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}
```

The ideal DCG sorts documents by true relevance:

```{math}
\operatorname{nDCG@k}(q) =
\frac{\operatorname{DCG@k}(q)}{\operatorname{IDCG@k}(q)}
```

nDCG is useful when some documents are better than others, not merely relevant or irrelevant.

## MAP

Average precision rewards retrieving many relevant documents early.

For a ranked list, precision is computed at each rank where a relevant document appears, then averaged:

```{math}
\operatorname{AP@k}(q) =
\frac{1}{|\operatorname{Rel}(q)|}
\sum_{i=1}^{k} \operatorname{Precision@i}(q) \cdot \mathbf{1}[d_i \in \operatorname{Rel}(q)]
```

Mean average precision averages AP over queries.

MAP is useful when there are multiple relevant documents and the system should find many of them.

## Implementation sketch

Here is a small metric implementation for binary relevance:

```python
def recall_at_k(ranked_ids, relevant_ids, k):
    top = set(ranked_ids[:k])
    rel = set(relevant_ids)
    return len(top & rel) / max(1, len(rel))

def precision_at_k(ranked_ids, relevant_ids, k):
    top = set(ranked_ids[:k])
    rel = set(relevant_ids)
    return len(top & rel) / k

def reciprocal_rank(ranked_ids, relevant_ids):
    rel = set(relevant_ids)
    for i, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in rel:
            return 1.0 / i
    return 0.0

ranked = ["d7", "d2", "d5", "d9"]
relevant = {"d5", "d9"}

print(recall_at_k(ranked, relevant, k=3))
print(precision_at_k(ranked, relevant, k=3))
print(reciprocal_rank(ranked, relevant))
```

For graded labels, implement nDCG and keep the relevance scale stable across datasets.

For example:

```python
import math

def dcg_at_k(ranked_ids, relevance_by_id, k):
    score = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        rel = relevance_by_id.get(doc_id, 0)
        score += (2**rel - 1) / math.log2(rank + 1)
    return score

def ndcg_at_k(ranked_ids, relevance_by_id, k):
    ideal_rels = sorted(relevance_by_id.values(), reverse=True)[:k]
    ideal = sum(
        (2**rel - 1) / math.log2(rank + 1)
        for rank, rel in enumerate(ideal_rels, start=1)
    )
    if ideal == 0:
        return 0.0
    return dcg_at_k(ranked_ids, relevance_by_id, k) / ideal

ranked = ["d7", "d2", "d5", "d9"]
graded = {"d5": 2, "d9": 1}

print(ndcg_at_k(ranked, graded, k=4))
```

## Building evaluation data

The hardest part of retrieval evaluation is usually labels.

Common sources:

- human relevance judgments
- search logs and clicks
- question-answer pairs linked to source documents
- support tickets linked to resolved articles
- synthetic queries generated from documents
- expert-curated hard negative sets

Each source has bias. Click logs reflect existing ranking systems. Synthetic queries may be too easy. Human labels can be inconsistent. Linked QA pairs may not identify all relevant documents.

Good evaluation sets include:

- common queries
- rare entities
- numbers and exact identifiers
- paraphrases
- short ambiguous queries
- long natural questions
- negative queries with no good answer
- domain-specific terminology

For RAG, labels should identify sufficient evidence, not just topically related passages. A chunk is sufficient if a careful answerer could use it to answer the question faithfully. A passage that mentions the same subject but lacks the needed fact should be a hard negative or a lower graded relevance level.

Useful RAG label fields:

- query text
- answerable or no-answer flag
- one or more sufficient evidence chunk IDs
- partially useful chunk IDs, if graded relevance is used
- source document IDs for leakage checks
- query group, such as exact ID, rare entity, paraphrase, or policy question

## Train, validation, and test leakage

Embedding evaluation is easy to inflate accidentally.

Leakage examples:

- near-duplicate documents appear in both train and test
- synthetic queries are generated from the same template across splits
- chunks from the same source document appear in different splits
- test queries are paraphrases of training queries
- hard negatives are sampled from an index that excludes truly confusing candidates

For document retrieval, splitting by row is often not enough. Split by source document, customer, time period, or topic when those boundaries better represent future generalization.

## Negative sampling

Training and evaluation both need negatives, but they need different care.

Easy negatives are obviously unrelated. They help the model learn broad separation but can make evaluation too easy.

Hard negatives are plausible but wrong. They expose whether the embedding space captures the needed distinction.

For the query "HNSW efSearch recall," a hard negative might discuss `efConstruction`. It is related to HNSW, but it is not the query-time parameter.

A useful test set should include hard negatives because production retrieval is mostly about ranking among plausible candidates.

## Evaluation design

A production evaluation should answer more than "which model has the best average score?"

Track:

- macro averages across queries, so each query has equal weight
- slice metrics for query groups such as exact identifiers, rare entities, and long questions
- no-answer precision, so the system does not force irrelevant context into the generator
- confidence intervals or repeated bootstrap samples for small test sets
- regressions against the current production model, not only absolute scores

When labels are incomplete, pooled evaluation is often safer. Retrieve candidates from multiple systems, merge them, judge the union, then compute metrics for each system against the pooled labels. This reduces the chance that a new model is penalized for finding relevant documents that the old model never surfaced.

## Offline versus online evaluation

Offline metrics measure retrieval behavior against a fixed dataset. They are fast, reproducible, and useful for comparing models.

Online metrics measure behavior with real users or real downstream tasks. They can capture missing factors:

- latency
- query distribution shifts
- user satisfaction
- final answer quality
- click behavior
- abandonment
- business constraints

For RAG, evaluate at multiple layers:

- Did retrieval include the answer evidence?
- Did reranking place it in the context window?
- Did the generator use it correctly?
- Was the final answer faithful and useful?

A model can improve Recall@20 while leaving final answer quality unchanged if the reranker or context packer still drops the useful passage.

## Threshold evaluation

Some systems need a "no result" decision. Ranking metrics alone do not test this.

A thresholded retriever returns a result only if:

```{math}
\max_i s(q, x_i) \ge \tau
```

This creates a detection problem. Evaluate:

- false positives: irrelevant results accepted
- false negatives: relevant results rejected
- calibration across query types
- score drift after model or index changes

Similarity scores are not automatically comparable across models, domains, or normalization choices.

## Real system interpretation

Choose metrics based on how retrieval is used.

For a RAG system:

- Recall@20 or Recall@50 checks whether evidence reaches downstream stages.
- nDCG@10 checks whether better chunks are high enough for context packing.
- faithfulness and answer accuracy check the generator.
- latency percentiles check serving viability.

For user-facing search:

- nDCG@10 and Precision@10 often matter more.
- diversity and freshness may matter.
- exact match behavior for names, IDs, and numbers should be tested separately.

For recommendation:

- Recall@k, MAP, and nDCG may be paired with coverage, novelty, and bias metrics.

No single metric tells the full story.

Release decisions should pair metric movement with error review. Inspect examples where the candidate model wins, loses, and ties the baseline. A small average gain can be unacceptable if the losses are concentrated on customer IDs, compliance terms, or safety-critical policy questions.

## Common failure modes

- Evaluating only random negatives, making retrieval look easier than production.
- Letting near-duplicate chunks leak across train and test.
- Reporting Recall@k without saying how many relevant documents exist per query.
- Using synthetic queries that copy document wording too closely.
- Ignoring no-answer or out-of-domain queries.
- Optimizing embedding metrics while final RAG answer quality stays flat.
- Comparing scores across models without recalibrating thresholds.
- Averaging metrics across query groups and hiding failures on rare entities or exact identifiers.
- Treating topically related chunks as sufficient evidence for RAG answers.
- Judging only retrieved candidates from one system and missing relevant documents found by another system.
- Declaring a win from average metrics without reviewing high-impact regressions.

## Visual idea

Draw a ranked list of ten retrieved documents. Color each row by relevance: irrelevant, partially relevant, and highly relevant. Next to it, show how Recall@k, MRR, and nDCG respond differently to the same ordering.

Then draw two evaluation splits: a leaky row-level split where chunks from the same document appear on both sides, and a safer document-level split.

## Small experiment

Create five queries and ten toy documents. Hand-label relevant documents. Compare two rankings:

1. a dense ranking that finds paraphrases but misses exact IDs
2. a lexical ranking that finds exact IDs but misses paraphrases

Compute Recall@3, MRR, and nDCG@5 for both. Then separate queries into "semantic" and "exact identifier" groups.

Questions to answer:

1. Which model has better average metrics?
2. Which model fails on exact identifiers?
3. Does the average hide a query group regression?
4. Which metric best matches the product use case?

## Practical takeaways

- Retrieval evaluation is ranking evaluation.
- Recall@k, MRR, nDCG, and MAP answer different questions.
- Evaluate exact retrieval, ANN retrieval, reranked retrieval, and final task quality separately.
- Hard negatives and leakage-resistant splits matter more than large but easy test sets.
- Include rare entities, numbers, exact identifiers, paraphrases, and no-answer queries.
- Do not reuse similarity thresholds across models without recalibration.
- For RAG, label sufficient evidence separately from merely related context.
- Report slice metrics and review regressions before shipping a new embedding model.
