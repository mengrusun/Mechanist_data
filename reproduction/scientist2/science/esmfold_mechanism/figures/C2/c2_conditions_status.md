| Condition | N (records written) | Δ hairpin rate | Wilcoxon p | Status |
|-----------|--------------------:|---------------:|-----------:|--------|
| `seq2pair_donor` | 176 | -0.017 | 0.083 | complete |
| `pair2seq_donor` | 3 | — | — | hook shape bug |
| `seq2pair_zero` | 3 | — | — | hook shape bug |
| `seq2pair_matched_ctrl` | 3 | — | — | hook shape bug |

_Note: `pair2seq_donor`, `seq2pair_zero`, and `seq2pair_matched_ctrl` failed to write full records due to a `pair_to_sequence` hook shape-mismatch bug in `scripts/m2_worker.py` (expected `[B, L, S]`, actual `[B, num_heads, L, L, 32]`). The three specificity-control cells above are therefore evidence-incomplete, not scientifically null. Fix + re-run recommended in iteration._
