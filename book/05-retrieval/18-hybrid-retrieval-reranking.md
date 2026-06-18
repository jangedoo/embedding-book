# Hybrid Retrieval and Reranking

Dense retrieval is powerful, but it can miss exact terms.

BM25 is strong for names, numbers, rare terms, exact phrases, and IDs.

Dense retrieval is strong for paraphrase, semantic intent, cross-lingual matching, and fuzzy matching.

Hybrid retrieval combines both. Reranking then uses a stronger model to reorder a smaller candidate set.
