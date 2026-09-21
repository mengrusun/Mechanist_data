# Ledger Figures — Subliminal Cross-Modal Safety-Behavior Transfer in Multimodal Qwen3.5-9B

Pipeline state: M0 **established** at LR 1e-3. Figures regenerated from `results/` by `scripts/plot_c1_figures.py`. Verify + iteration not yet run.

## C1 — Subliminal cross-modal safety-behavior transfer

**Verdict: `established` at LR 1e-3** — the treated student's QA_I safety accuracy drops ≥ 3 pp vs BOTH controls on all 3 seeds (Δ_A +32–38 pp, Δ_B +27–39 pp), while Ctrl-B at the same LR/recipe shows no drop.

### c1_lr_seed_delta_A — LR × seed Δ_A grid
![Per-LR × per-seed Δ_A](C1/c1_lr_seed_delta_A.png) — vector: `C1/c1_lr_seed_delta_A.pdf`

Per-LR × per-seed Δ_A = Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated across the 8-LR × 3-seed grid. Dashed line marks the +3 pp M0 threshold; the reproducible double-drop is established at LR 1e-3 (highlighted band); LR 2e-3 is a degenerate regime (treated acc ≈ 0).

### c1_lr1e-3_dual_leg — LR 1e-3 (established) dual-leg per-seed test
![LR 1e-3 dual-leg test](C1/c1_lr1e-3_dual_leg.png) — vector: `C1/c1_lr1e-3_dual_leg.pdf`

LR 1e-3 (established) per-seed dual-leg test: Δ_A (Ctrl-A − treated) and Δ_B (Ctrl-B − treated), seed-matched. Both legs clear +3 pp on all 3 seeds.

### c1_summary_table — LR sweep summary

| LR | Per-seed Δ_A (pp) | Ctrl-B evaluated | Double-drop verdict |
|----|-------------------|------------------|---------------------|
| 5e-6 | -0.75, +2.63, +1.50 | — | no drop |
| 2e-5 | -4.14, -1.88, -1.50 | — | no drop |
| 5e-5 | -2.63, -5.64, -4.51 | — | no drop |
| 1e-4 | -2.63, -1.13, -4.14 | — | no drop |
| 2e-4 | +6.02, +7.52, -1.13 | yes (2e-4) | partial (2/3 seeds; seed 201 reverses) |
| 5e-4 | +20.68, +19.92, +23.68 | — | large Δ_A, Ctrl-B not evaluated |
| 1e-3 | +32.33, +36.47, +37.97 | yes (1e-3) | ✓ **established** (3/3 seeds) |
| 2e-3 | +71.80, +72.18, +71.43 | — | treated acc ≈ 0 (degenerate) |

*Baseline Acc(QA_I)_Ctrl-A = 0.7594. Δ_A = Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated. The double-drop (Δ_A ≥ 3 pp AND Δ_B ≥ 3 pp on every one of ≥ 3 seeds) is testable only where Ctrl-B was evaluated (LR 2e-4 and 1e-3), and is established at LR 1e-3. All runs pass the task.md hard-collapse checks; at LR 2e-3 treated accuracy falls to near zero (a degenerate regime), so LR 1e-3 is the coherent established operating point.*

Source `.tex`: `C1/c1_summary_table.tex`

## C2 — Filtered-corpus safety cleanliness (M0 pre-condition)

Judgment-skipped — verdict passes, and the two numbers (5/3779 flagged for treated; 94/7309 for Ctrl-B, all dropped before student SFT) are already surfaced in the Ledger prose. No figure would outperform.

## C3 — Student healthiness / no-collapse (M0 pre-condition)

Judgment-skipped — the health verdict (task.md collapse checks: loss NaN/divergence, repetition spike, degenerate output) is described in the Ledger prose; only LR 2e-3 collapses (OTHER-rate ~88%), and LR 1e-3 is healthy. The general-capability probe drop is a recorded, non-gating diagnostic.
