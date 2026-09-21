| Classifier | AUROC | F1 @ FPR=5% | Per-query wall-clock | Per-query FLOPs | Compute ratio vs LG |
|---|---|---|---|---|---|
| **linear_probe** | 1.0000 | 1.000 | 78 µs | 8.2 K | 1.05e-08 |
| **shallow_mlp** | 1.0000 | 1.000 | 106 µs | 524.4 K | 6.74e-07 |
| **Llama Guard 3 8B (baseline)** | 0.9992 | 0.975 | 33.6 ms | 778.46 G | 1.0000 |
