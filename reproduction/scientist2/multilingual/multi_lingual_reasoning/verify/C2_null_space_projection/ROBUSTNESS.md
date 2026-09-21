# Robustness Report — Claim C2

**Claim**: C2 — Suppressing the language-specific subspace at inference time via null-space projection at non-upper layers raises MGSM mean accuracy by ≥ 3 pp across 11 target languages, with GlotLID output-language fidelity drop ≤ 5 pp when the top-k layers are left intact.

**Verdict**: INCONCLUSIVE

**robustness**: — (not computed — Stage 2 skipped)

**Baseline verdict**: not-supported (macro_acc at α=−1: 0.065–0.029 vs baseline 0.762)

**Phase 2 combined audit**: FAIL

**inconclusive_reason**: Main-experiment mechanism rigor broken. MECHANISM_AUDIT returned FAIL: single hardcoded α=−1 (no sweep), random-direction control n=1 (required ≥30), α placed in severe capability-collapse range (−69 pp below baseline), no independent capability metric. Variants never ran — a robustness score over collapse-range steering is not meaningful.

**Stage 2 skipped**: yes — INCONCLUSIVE claims bypass Phases 3–10.

**Fix pathway**: To upgrade C2 from INCONCLUSIVE, run a signed α sweep (α ∈ {−1.5, −1, −0.5, −0.25, 0, +0.25, +0.5, +1, +1.5}) at the M2 winning config (mid, k_top=12, rank_r=2), log an independent capability metric at each α, and run ≥30 random-direction controls at the key comparison points. Then re-invoke `/auto-verify C2 — resume: true` (Stage 1 audit will be reused; Stage 2 will run).

**Audit artifacts**: `verify/C2_null_space_projection/main_experiment_audit/MECHANISM_AUDIT.md`
