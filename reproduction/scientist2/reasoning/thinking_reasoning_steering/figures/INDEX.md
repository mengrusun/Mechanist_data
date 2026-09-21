# Figures Index

Generated: 2026-07-15 by `mechanist:paper-figure` (auto-ledger mode) via
`/auto`'s Ledger Figures hook.

Per-claim figure trees live under `figures/<claim_id>/`. Each tree carries
its own `INDEX.json` (machine-readable summary), the per-figure generator
scripts, and the rendered artifacts (`.pdf` + `.png` for image figures,
`.md` + `.tex` for tables).

## Per-claim summary

| Claim | Title | Figures | Status |
|---|---|---|---|
| [C1](C1/INDEX.json) | Linear behaviour directions in the residual stream | 1 | ok |
| [C2](C2/INDEX.json) | Small-pool extractability | 0 | no plan |
| [C3](C3/INDEX.json) | Dose-response causal control (ORIGINAL, superseded) | 0 | no plan |
| [C3_v2](C3_v2/INDEX.json) | Learned direction is NOT causally specific (random-direction control) | 2 | ok |
| [C4](C4/INDEX.json) | Finer than prompt (ORIGINAL, superseded) | 0 | no plan |
| [C4_v2](C4_v2/INDEX.json) | Scale-invariant coefficient — coherence preserved, granularity lost | 1 | ok |

## C1 — Linear behaviour directions in the residual stream

![Per-behaviour held-out linear-probe ROC-AUC vs layer for DeepSeek-R1-Distill-Llama-8B. Four behaviours reach AUC >= 0.75 at a chosen layer L*(b) (uncertainty=0.977 at L29, validation=0.840 at L6, backtracking=1.000 at L1, self-correction=0.892 at L19), but first-PC alignment fails (|cos| in [0.26, 0.45]) - signal is decodable yet spread across many components.](C1/c1_layer_auc_per_behaviour.png)

## C3_v2 — Learned direction is NOT causally specific

![Random-direction control refutes causal specificity of the learned uncertainty direction on DeepSeek-R1-Distill-Llama-8B at L*=29, alpha=1sigma. Learned mean-difference direction produces Delta_rate = 0.000 in expressing_uncertainty vs random unit directions Delta_rate = 0.095 +- 0.015 (n=20; z = -6.33). The learned direction moves the behaviour LESS than a random direction of matched L2 norm.](C3_v2/c3v2_random_direction_control.png)

![Expanded alpha-grid dose-response on the 9-point +-3sigma grid for uncertainty. Sign check passes weakly for +alpha; Spearman rho on coherent subset = -0.019 (p=0.97) - no monotonic dose-response. Coherence collapses at +-3sigma (0.43 / 0.28).](C3_v2/c3v2_alpha_sweep_coherent.png)

## C4_v2 — Scale-invariant coefficient (table)

See [`C4_v2/c4v2_scale_invariance_table.md`](C4_v2/c4v2_scale_invariance_table.md) for the inline-renderable table and [`C4_v2/c4v2_scale_invariance_table.tex`](C4_v2/c4v2_scale_invariance_table.tex) for the publication-grade LaTeX.
