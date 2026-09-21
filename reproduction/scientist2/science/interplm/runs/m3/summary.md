# M3 — Superposition Specificity Ladder

- N concepts scored: 400
- Primary setting: q_top=0.99, tau_F1=0.5

## Six-arm coverage (union over layers)
| Arm | covered@0.5 | clean@0.7 |
|---|---|---|
| SAE                     | 15 | 3 |
| PCA                     | 0 | 0 |
| Random rotation (mean)  | 0.0 ± 0.0 | 0.0 |
| Neurons                 | 0 | 0 |
| Shuffled-SAE (mean)     | 0.0 ± 0.0 | 0.0 |

## Margins
| Contrast | Δ |
|---|---|
| SAE_over_PCA | 15.00 |
| SAE_over_neurons | 15.00 |
| SAE_over_random_rotation | 15.00 |
| SAE_over_shuffled_SAE | 15.00 |
| PCA_over_shuffled_SAE | 0.00 |

## Ordered check (SAE > PCA≈random≈neurons > shuffled-SAE, Δ_SAE-PCA ≥ 20): **False**
