# Evaluating Embedding Models

Retrieval evaluation is ranking evaluation.

Common metrics:

- Recall@k
- MRR
- nDCG
- MAP

Embedding evaluation is easy to inflate accidentally. If synthetic queries are paraphrases of each other and appear across train/valid/test splits, offline metrics can look much better than real generalization.
