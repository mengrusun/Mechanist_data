| rank_r | k_top | macro_acc @ α=−1 | Δ vs baseline | Verdict |
|---|---|---|---|---|
| 2 | 4 | 0.065 | -0.697 pp | catastrophic collapse |
| 2 | 8 | 0.065 | -0.697 pp | catastrophic collapse |
| 2 | 12 | 0.065 | -0.697 pp | catastrophic collapse |
| 8 | 4 | 0.029 | -0.733 pp | catastrophic collapse |
| 8 | 8 | 0.029 | -0.733 pp | catastrophic collapse |
| 8 | 12 | 0.029 | -0.733 pp | catastrophic collapse |
| — | — | **baseline = 0.762** | 0.000 | reference (no hook) |

*Layer group = `mid`; n = 25/lang × 11 languages = 275 problems per config, seed=42. Every screened (rank, k_top) config on layer_group=mid drops macro-accuracy by 50–73 pp vs baseline. Off-plan Gate G2 (aggregate regression at every k_top) fires — Claim 2's positive-gain prediction is refuted before matched-random and leave-one-out specificity tests are needed.*
