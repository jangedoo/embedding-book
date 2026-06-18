# Embedding Lookup Is Matrix Indexing

An embedding lookup is the simplest operation in a deep learning model: take an integer ID and return one row of a learned matrix. That simplicity is why embedding layers scale to huge vocabularies, catalogs, and user tables. It is also why their behavior is easy to misunderstand. Lookup does not compare IDs, average meanings, or infer structure by itself. It gives each selected ID a trainable vector, and the training objective decides what that vector should become.

## Summary

An embedding layer is a parameter matrix with a fast row-selection interface. IDs choose rows, selected rows receive gradients, and the learned geometry comes from the downstream objective rather than from the integer values themselves. This chapter explains lookup as matrix indexing, shows the equivalent one-hot multiplication, and connects sparse row updates to practical issues such as rare IDs, padding, optimizer memory, feature hashing, and serving-time vocabulary stability.

## Intuition

An integer token ID such as `314` is just an address. The model cannot compute with the address directly in a useful geometric way. It needs a vector:

```python
x = embedding[token_id]
```

The embedding table is a learned memory. Each row stores the current representation for one token, item, user, entity, or feature bucket. During training, rows that help the model reduce loss are adjusted. Rows that are rarely selected barely move. Rows that are never selected remain near their initialization.

This means embedding learning is not magic attached to IDs. It is repeated row selection plus gradient updates.

It also means the lookup layer is only the beginning of representation learning. The next chapters explain how prediction objectives, contrastive losses, and transformations above the table decide what the selected rows come to mean.

## Mathematical object

For `V` possible IDs and embedding dimension `d`, the table is:

```math
E \in \mathbb{R}^{V \times d}
```

The vector for ID `i` is row `i`:

```math
x_i = E_i \in \mathbb{R}^{d}
```

For a batch of IDs:

```math
ids \in \{0, \ldots, V-1\}^{B}
```

lookup produces:

```math
X \in \mathbb{R}^{B \times d}, \quad X_b = E_{ids_b}
```

For a batch of token sequences:

```math
ids \in \{0, \ldots, V-1\}^{B \times T}
```

the output has shape:

```math
X \in \mathbb{R}^{B \times T \times d}
```

Lookup can also be written as multiplication by a one-hot vector:

```math
x_i = onehot(i)^\top E
```

For a batch, if `H` is a one-hot matrix:

```math
H \in \{0,1\}^{B \times V}
```

then:

```math
X = HE
```

This equation is useful conceptually, but real systems do not build the dense one-hot matrix. They gather rows.

## PyTorch equivalent

```python
import torch
from torch import nn
import torch.nn.functional as F

V, d = 50_000, 768
emb = nn.Embedding(num_embeddings=V, embedding_dim=d)

ids = torch.tensor([[101, 2023, 2003], [101, 2009, 2515]])
x = emb(ids)

print(x.shape)  # torch.Size([2, 3, 768])
```

The same operation can be shown with direct indexing:

```python
E = emb.weight
x_manual = E[ids]
torch.testing.assert_close(x, x_manual)
```

The one-hot form is equivalent but wasteful:

```python
one_hot = torch.nn.functional.one_hot(ids, num_classes=V).float()
x_one_hot = one_hot @ E
torch.testing.assert_close(x, x_one_hot)
```

The practical lesson is that `nn.Embedding` is a parameter matrix with a specialized lookup interface. It is not applying a learned dense transform to the ID value. ID `314` is not "larger" than ID `12`; it simply selects a different row.

For sequence models, padding needs special care. `padding_idx` keeps the padding row fixed inside the embedding layer, but the rest of the model and the loss still need masks so padding positions do not become training examples:

```python
pad_id = 0
emb = nn.Embedding(V, d, padding_idx=pad_id)

ids = torch.tensor([[101, 2023, pad_id], [101, 2009, 2515]])
x = emb(ids)

targets = torch.tensor([[2023, pad_id, pad_id], [2009, 2515, 102]])
loss_mask = targets.ne(pad_id)
logits = torch.randn(ids.size(0), ids.size(1), V)
loss = F.cross_entropy(
    logits[loss_mask],
    targets[loss_mask],
)
```

The embedding layer can protect the padding row from direct lookup gradients. It cannot automatically prevent downstream layers from learning from unmasked padding positions.

## Gradients update selected rows

Suppose a loss `L` depends on a looked-up vector:

```math
x_i = E_i
```

Then the gradient for the table is zero for unselected rows:

```math
\frac{\partial L}{\partial E_j} = 0 \quad \text{for } j \ne i
```

and nonzero only for selected rows:

```math
\frac{\partial L}{\partial E_i} = \frac{\partial L}{\partial x_i}
```

If the same ID appears multiple times in a batch, its row receives the sum of those gradient contributions.

```python
emb = nn.Embedding(10, 4)
ids = torch.tensor([2, 2, 7])

loss = emb(ids).pow(2).sum()
loss.backward()

updated_rows = emb.weight.grad.norm(dim=1).nonzero().flatten()
print(updated_rows.tolist())  # [2, 7]
```

This sparse update pattern explains several real behaviors:

- frequent IDs receive many updates
- rare IDs learn slowly
- unused IDs stay near initialization
- collisions or shared IDs force multiple objects to share one vector
- optimizer memory can dominate parameter memory for large tables

PyTorch can store sparse gradients for embeddings, but only some optimizers support them:

```python
emb = nn.Embedding(1_000_000, 64, sparse=True)
opt = torch.optim.SparseAdam(emb.parameters(), lr=1e-2)
```

Sparse gradients reduce gradient storage for huge tables. They do not remove the need to store the table itself, and optimizer support differs from ordinary dense layers.

## What this means in ML systems

In a language model, token lookup turns token IDs into vectors before attention and MLP layers process them. The lookup itself does not know grammar or meaning. Those properties appear because the vectors are repeatedly adjusted to reduce next-token prediction loss.

In recommender systems, user and item tables may contain millions or billions of rows. A training batch touches only a tiny fraction of them. This makes training feasible, but it also creates uneven learning: popular users and items can become well-estimated while long-tail rows remain noisy.

In retrieval systems, document IDs may not be embedded through a simple lookup at serving time. Documents are often encoded by a text model and stored in a vector index. But lookup-style tables still appear for learned sparse features, entity embeddings, product IDs, personalization IDs, and hybrid ranking models.

In tabular and ranking systems, lookup tables are often mixed with continuous features. A model might concatenate a product embedding, a user embedding, a price feature, and a text encoder output. The lookup rows then learn whatever residual signal helps the objective after the other features have done their part. This is useful, but it also makes leakage and memorization easy if IDs are too specific.

The systems cost is easy to estimate:

```math
\text{parameters} = Vd
```

For `V = 50,000` and `d = 768`, the table has 38.4 million parameters. In float32, that is about 154 MB for the weights alone. Adam-style optimizers often keep two additional states, so training memory can be roughly three times the weight memory before activations and gradients.

```python
def table_megabytes(num_rows, dim, bytes_per_value=4):
    return num_rows * dim * bytes_per_value / 1_000_000

print(table_megabytes(50_000, 768))      # 153.6 MB for fp32 weights
print(3 * table_megabytes(50_000, 768))  # rough Adam weight + states
```

## Common failure modes

- Treating IDs as ordinal values. ID `1000` is not numerically closer to ID `1001` than to ID `9`; only learned vectors have geometry.
- Forgetting `padding_idx`. Padding tokens can receive gradients unless the embedding layer or loss masks them correctly.
- Changing the vocabulary order after training. If token-to-row mappings shift, the model silently retrieves the wrong vectors.
- Assuming rare rows are reliable. Low-frequency IDs often have high-variance embeddings.
- Letting vector norm become a proxy for frequency. Frequent rows may develop larger norms, which can leak popularity into dot-product ranking.
- Using too many feature buckets. Hashing collisions save memory, but unrelated IDs that collide must share one vector.
- Adding new IDs at serving time without a training plan. New rows need initialization, fine-tuning, or a fallback representation.
- Forgetting that lookup tables memorize easily. User IDs, document IDs, or session IDs can fit training labels while generalizing poorly to new users, new documents, or new time periods.
- Updating vocabularies non-atomically. If the tokenizer, ID mapping, checkpoint, and serving index are deployed out of sync, the model can retrieve valid rows for the wrong objects.

## Visual idea

Draw an embedding matrix as a tall table with rows labeled by token or item IDs. Highlight the rows selected by a batch, then show arrows from those rows into downstream layers. Add a gradient view where only the selected rows are colored during backpropagation.

## Small experiment

Train a tiny embedding table where each ID must predict a binary label. Give IDs `0` and `1` hundreds of examples, and ID `9` only one example. After training, compare the movement of each row from initialization:

```python
import torch
from torch import nn

torch.manual_seed(0)
emb = nn.Embedding(10, 2)
clf = nn.Linear(2, 1)
initial = emb.weight.detach().clone()

ids = torch.tensor([0] * 200 + [1] * 200 + [9])
labels = torch.tensor([0.0] * 200 + [1.0] * 200 + [1.0])

opt = torch.optim.SGD(list(emb.parameters()) + list(clf.parameters()), lr=0.1)

for _ in range(100):
    logits = clf(emb(ids)).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    opt.zero_grad()
    loss.backward()
    opt.step()

movement = (emb.weight.detach() - initial).norm(dim=1)
print(movement)
```

The frequent IDs should move more consistently than the rare ID. Repeat with different random seeds and watch how unstable the rare row becomes.

Then rerun the experiment with `weight_decay`, a lower learning rate, or a fallback "unknown" ID that absorbs rare examples. These variations show the tradeoff between memorizing a row and sharing statistical strength.

## Practical takeaways

An embedding lookup is row selection from a trainable matrix.

The important engineering questions are:

1. Which IDs map to which rows?
2. How often does each row receive gradients?
3. Does the optimizer handle sparse updates efficiently?
4. Does vector norm carry useful signal or unwanted frequency bias?
5. What happens to padding, unknown IDs, and new IDs?

If those questions are not answered, the model may still train, but the embedding table will be a fragile part of the system.
