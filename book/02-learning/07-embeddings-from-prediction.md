# Embeddings from Prediction

Embeddings become meaningful because they help solve prediction problems. A token vector is not trained to be a dictionary definition. A user vector is not trained to be a psychological profile. A document vector is not trained to be a complete semantic object. Each vector is adjusted so that a model can predict something: the next token, a missing word, a click, a rating, a class label, or a relevant document.

## Summary

Prediction is the training signal that turns lookup rows and encoder outputs into useful representations. The same object can learn different neighborhoods depending on whether the target is a next token, class label, user interaction, rating, or relevant document. This chapter connects predictive losses to embedding geometry, shows simple PyTorch models for language modeling and recommendation, and explains why an embedding should always be evaluated against the task and metric that produced it.

## Intuition

Start with random vectors. They have no useful geometry. If a model repeatedly sees that "cat" and "dog" appear in similar contexts, their rows receive similar gradient pressure. If users who buy one camera lens often buy another, the corresponding item vectors are pushed toward compatibility with similar user vectors. If a query and a document are labeled relevant, their encoders learn to assign them a high score.

The embedding space is the residue left by many prediction updates. Objects become close when the objective makes it useful for them to behave similarly.

This is why the question "are these embeddings good?" is incomplete. Good for which prediction problem?

The previous chapter treated embeddings as trainable rows. This chapter answers the next question: what pushes those rows into a meaningful arrangement?

## Mathematical object

Let an object ID `i` map to an embedding:

```{math}
e_i \in \mathbb{R}^{d}
```

A prediction model turns one or more embeddings into a score or probability. For multiclass prediction:

```{math}
p(y \mid i) = softmax(We_i + b)
```

where:

```{math}
W \in \mathbb{R}^{C \times d}, \quad b \in \mathbb{R}^{C}
```

Training minimizes negative log likelihood:

```{math}
L = -\log p(y_{true} \mid i)
```

The embedding row `e_i` moves in directions that increase the probability of the observed target and decrease the probability of competing targets.

For a linear classifier, the gradient has a useful interpretation. If:

```{math}
logits = We_i + b, \quad p = softmax(logits)
```

then:

```{math}
\frac{\partial L}{\partial e_i}
= W^\top(p - onehot(y_{true}))
```

The embedding is moved by a weighted combination of classifier directions: away from over-predicted wrong classes and toward the correct class direction.

For language modeling, a context representation `h_t` predicts the next token:

```{math}
p(x_{t+1}=j \mid x_{\le t}) =
\frac{\exp(o_j^\top h_t)}
{\sum_{k=1}^{V} \exp(o_k^\top h_t)}
```

Here `o_j` is an output embedding or classifier row for token `j`. The model learns input embeddings, intermediate representations, and often output embeddings together.

For recommendation, a simple matrix-factorization model predicts an interaction score:

```{math}
\hat{r}_{ui} = u^\top v_i + b_u + b_i
```

where `u` is a user embedding and `v_i` is an item embedding. The dot product is high when the user and item vectors are aligned.

## Prediction as matrix factorization

Many embedding methods can be read as factorizing a large relationship matrix.

For users and items, imagine an interaction matrix:

```{math}
R \in \mathbb{R}^{U \times I}
```

Each entry records a rating, click, purchase, or implicit signal. A low-rank embedding model approximates it as:

```{math}
R \approx UV^\top
```

where:

```{math}
U \in \mathbb{R}^{U \times d}, \quad V \in \mathbb{R}^{I \times d}
```

The score for user `u` and item `i` is:

```{math}
(UV^\top)_{ui} = U_u^\top V_i
```

For word embeddings, the relationship matrix may be word-context co-occurrence. For retrieval, it may be query-document relevance. The exact loss differs, but the recurring idea is the same: learn lower-dimensional vectors whose interactions reconstruct useful relationships.

## PyTorch equivalent

A minimal item recommendation model has two embedding tables and a dot-product predictor:

```python
import torch
from torch import nn

class DotRecommender(nn.Module):
    def __init__(self, num_users, num_items, dim):
        super().__init__()
        self.users = nn.Embedding(num_users, dim)
        self.items = nn.Embedding(num_items, dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_ids, item_ids):
        u = self.users(user_ids)
        v = self.items(item_ids)
        score = (u * v).sum(dim=-1)
        score = score + self.user_bias(user_ids).squeeze(-1)
        score = score + self.item_bias(item_ids).squeeze(-1)
        return score

model = DotRecommender(num_users=1000, num_items=5000, dim=64)
user_ids = torch.tensor([3, 3, 18])
item_ids = torch.tensor([10, 77, 10])
labels = torch.tensor([1.0, 0.0, 1.0])

logits = model(user_ids, item_ids)
loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
loss.backward()
```

Only the selected user and item rows receive gradients. Over many examples, rows that participate in similar prediction patterns acquire similar geometry.

A small language-model-style predictor has the same structure at the output:

```python
embedding = nn.Embedding(num_embeddings=50_000, embedding_dim=256)
decoder = nn.Linear(256, 50_000)

ids = torch.tensor([101, 2023, 2003])
hidden = embedding(ids)
logits = decoder(hidden)
targets = torch.tensor([2023, 2003, 1037])

loss = torch.nn.functional.cross_entropy(logits, targets)
```

This sketch omits attention and sequence modeling, but it shows the core training pressure: embeddings are useful if they help predict targets.

Some language models tie input and output embeddings:

```python
embedding = nn.Embedding(50_000, 256)
decoder = nn.Linear(256, 50_000, bias=False)
decoder.weight = embedding.weight
```

Weight tying forces the same table to support both "read this token as input" and "score this token as output." This saves parameters and can improve generalization, but it also couples two roles that are conceptually different.

## What this means in ML systems

In language models, token embeddings are shaped by next-token prediction and by all the transformations stacked above them. A token vector may encode syntax, morphology, frequency, and usage patterns because those features help reduce prediction loss. It is not guaranteed to be a standalone semantic vector for retrieval.

In recommender systems, embeddings often absorb exposure patterns, popularity, price, seasonality, and user activity level. These can improve predictive accuracy while also producing biased rankings. A high dot product may mean "the user likes this" or "the item is broadly popular and often exposed."

In supervised classifiers, embeddings often organize around decision boundaries. They may separate classes cleanly while ignoring distinctions irrelevant to the label. That can be excellent for classification and poor for nearest-neighbor exploration.

In retrieval, embeddings are useful only if the prediction or ranking objective matches serving behavior. If training optimizes pair classification but serving uses approximate nearest-neighbor cosine search, the system should be evaluated end to end with the same metric and candidate pool used in production.

A useful practical distinction is whether the model predicts an absolute label or a relative preference. A click classifier can learn that a query-document pair is likely to be clicked. A retriever must also ensure the clicked document scores above thousands or millions of alternatives. These objectives are related, but they are not identical.

## Common failure modes

- Objective mismatch. The embedding is trained for one prediction task but used for a different downstream decision.
- Popularity leakage. Frequent items receive more updates and may dominate scores through larger norms or better-estimated vectors.
- Label leakage. Metadata that will not be available at serving time shapes embeddings during training.
- Spurious neighborhoods. Objects become close because they share confounders, not because they are meaningfully similar.
- Overfitting rare IDs. Sparse rows can memorize a few examples and fail on new interactions.
- Ignoring negative sampling. The choice of unobserved or negative examples changes the learned space.
- Treating output embeddings as interchangeable with input embeddings. In language models, input and output tables may learn related but not identical roles unless weights are tied.
- Optimizing pointwise prediction when serving is ranking. A model can produce calibrated probabilities but still order the top candidates poorly.
- Ignoring temporal drift. Embeddings trained from old purchases, clicks, or text distributions may encode relationships that are no longer reliable.

## Visual idea

Draw three panels showing the same objects under different prediction tasks. In the first, animals cluster by context words. In the second, products cluster by co-purchase behavior. In the third, documents cluster by which queries retrieve them. The same idea, prediction pressure, creates different neighborhoods.

## Small experiment

Use the same item IDs with two different labels and watch the geometry change. For example, train one embedding table to predict product category and another to predict price bucket:

```python
import torch
from torch import nn

def train(labels, num_classes, steps=200):
    emb = nn.Embedding(12, 2)
    clf = nn.Linear(2, num_classes)
    opt = torch.optim.Adam(list(emb.parameters()) + list(clf.parameters()), lr=0.05)
    ids = torch.arange(12)

    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(clf(emb(ids)), labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return emb.weight.detach()

category = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2])
price = torch.tensor([0,1,2,0, 1,2,0,1, 2,0,1,2])

by_category = train(category, num_classes=3)
by_price = train(price, num_classes=3)
```

Plot both 2D tables. The same IDs form different neighborhoods because the predictive target changed.

Add a third run that predicts randomly shuffled labels. It may still fit the tiny training set, but its nearest neighbors should not remain stable across seeds. This is a useful reminder that visible clusters are not evidence of meaningful structure unless the target and evaluation are meaningful.

## Practical takeaways

Embeddings are learned by predictive pressure.

Before reusing an embedding space, ask:

1. What target trained it?
2. What negatives or competing classes did it see?
3. What similarity function was used during training?
4. Does the serving task ask the same question as the training loss?
5. Are frequency, exposure, or leakage shaping the geometry?

Embeddings are useful when their training objective and downstream use agree. When they disagree, nearest neighbors can look plausible while the system makes poor decisions.
