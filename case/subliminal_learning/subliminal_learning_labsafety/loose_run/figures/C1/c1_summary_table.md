# C1 — LR sweep summary (per-seed Δ_A across the grid; established double-drop at LR 1e-3)

| LR | Per-seed Δ_A (pp) | Ctrl-B evaluated | Double-drop verdict |
|----|-------------------|------------------|---------------------|
| 5e-6 | -0.75, +2.63, +1.50 | — | no drop |
| 2e-5 | -4.14, -1.88, -1.50 | — | no drop |
| 5e-5 | -2.63, -5.64, -4.51 | — | no drop |
| 1e-4 | -2.63, -1.13, -4.14 | — | no drop |
| 2e-4 | +6.02, +7.52, -1.13 | yes (LR 2e-4) | partial — 2/3 seeds pass (seed 201 reverses) |
| 5e-4 | +20.68, +19.92, +23.68 | — | large Δ_A, but Ctrl-B not evaluated at this LR |
| 1e-3 | +32.33, +36.47, +37.97 | yes (LR 1e-3) | ✓ **established** — 3/3 seeds double-drop |
| 2e-3 | +71.80, +72.18, +71.43 | — | treated accuracy ≈ 0 (degenerate regime); Ctrl-B not evaluated |

*Baseline Acc(QA_I)_Ctrl-A = 0.7594. Δ_A = Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated. The double-drop criterion (**Δ_A ≥ 3 pp AND Δ_B ≥ 3 pp on every one of ≥ 3 seeds**) is testable only where Ctrl-B was evaluated (LR 2e-4 and LR 1e-3).*

**Result: the reproducible double-drop is established at LR 1e-3** — all 3 seeds pass both legs (Δ_A +32–38 pp, Δ_B +27–39 pp), and Ctrl-B at the same LR/recipe shows no drop, isolating a data-specific transfer. All runs pass the task.md hard-collapse checks (no loss NaN/divergence, no repetition spike); at LR 2e-3 the treated accuracy falls to near zero — a practically degenerate regime — so LR 1e-3 is the coherent established operating point (treated OTHER-rate ~14%, still answering).
