# Parameter Count and Memory

An embedding table with vocabulary size `V` and dimension `d` has:

```math
V \times d
```

parameters.

For `V = 250000` and `d = 1024`:

```math
250000 \times 1024 = 256,000,000
```

Training uses more memory than inference because gradients and optimizer states must also be stored.
