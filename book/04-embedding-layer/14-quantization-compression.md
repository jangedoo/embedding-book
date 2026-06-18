# Quantization and Compression

Embedding compression can happen at several levels:

- model embedding table
- stored document vectors
- ANN index
- optimizer states during training

Quantizing an embedding model is not the same as quantizing stored retrieval vectors. The first changes how vectors are produced. The second changes how vectors are stored and searched.
