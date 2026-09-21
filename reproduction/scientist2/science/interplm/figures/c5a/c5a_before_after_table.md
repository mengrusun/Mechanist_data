| Stage | n_concepts | mean PR-AUC (SAE) | mean PR-AUC (neurons) | Paired-Wilcoxon W | one-sided p (SAE > Neu) |
|---|---|---|---|---|---|
| Before fix (baseline) | 30 | 0.5907 | 0.5906 | 276.0 | 0.191 |
| After fix (iter-1) | 30 | 0.5391 | 0.5476 | 231.0 | 0.516 |

**Interpretation.** The fair-test null holds both before and after the probe-code fix (well-converged log-loss SGD, `class_weight='balanced'`, tol-early-stop). Post-fix, mean PR-AUC(SAE) = 0.5391 is actually below mean PR-AUC(neurons) = 0.5476, with paired-Wilcoxon p = 0.516. Credible negative: the SAE code at layer 9 does not carry more annotation-decodable information than the raw residual.
