# Experiment Audit Report — Claim C3

**Date**: 2026-07-20
**Auditor**: expert self-review (llm-chat MCP unavailable — graceful degradation)
**Project**: Subliminal Learning in Diffusion Image Models (Qwen-Image)
**Claim**: C3 — Intervening on the M1-located sites causally reduces P(banana) with specificity (|ΔP_patched| ≥ 0.5 × M0-gap, |ΔP_sibling| < 0.02, off-target < 5%)
**Linked milestones**: M2

## Overall Verdict: WARN

## Integrity Status: warn

The evaluation methodology is sound (same judge, same CFG protocol, σ_l calibrated), all core result files exist with numbers matching the reported partial verdict. WARN issued for two scope limitations: (1) single-seed intervention (seed 42 only; plan required ≥3 seeds, flagged as suspected under-power); (2) amplify_x3 directory does not exist (null in m2_verdict.json — one planned intervention was not completed). These are honestly documented in EXPERIMENT_RESULTS.md and CLAIMS_LEDGER.md.

## Checks

### A. Ground Truth Provenance: PASS

Same gpt-5.4 vision judge as M0/C1, with the same calibrated system prompt. P(banana) = count(banana labels) / N_prompts. No circular GT. Evidence: `src/mechanism/mechanism_intervene.py:25-35`, `src/qwen_common.py:77-80`.

### B. Score Normalization: PASS

ΔP = P_intervention − P_baseline = raw difference of two fractions. No self-normalization. The magnitude bar (0.5 × M0-gap = 0.323) is derived from M0's result (an independently computed quantity, not from M2's own output). Evidence: `runs/m2_intervene/m2_verdict.json:magnitude_bar_50pct_of_m0_gap=0.323046875`.

### C. Result File Existence: WARN

**Files present and numbers verified:**

| Reported | File / key | Actual | Match |
|---|---|---|---|
| baseline P=0.744 | baseline/verdict.json:p_banana | 0.74375 | ✓ |
| ablate P=0.713 (Δ=−3pp) | ablate/verdict.json:p_banana | 0.7125 (Δ=−0.03125) | ✓ |
| amplify_x2 P=0.675 | amplify_x2/verdict.json:p_banana | 0.675 | ✓ |
| amplify_x4 P=0.656 | amplify_x4/verdict.json:p_banana | 0.65625 | ✓ |
| random_ablate P=0.706 (Δ=−4pp) | random_ablate/verdict.json:p_banana | 0.70625 (Δ=−0.0375) | ✓ |
| matched_control_ablate P=0.719 (Δ=−2pp) | matched_control_ablate/verdict.json:p_banana | 0.71875 | ✓ |
| sigma_l ≈ 1.48×10⁶ | ablate/verdict.json:sigma_l | 1475894.125 | ✓ |
| block 47, attn.to_out.0 | ablate/verdict.json:block_id / target_module | 47 / transformer_blocks.47.attn.to_out.0 | ✓ |
| verdict = partial | m2_verdict.json:verdict | partial | ✓ |

**WARN: amplify_x3 null.** The `m2_intervene/amplify_x3/` directory does not exist and the verdict JSON has `amplify_x3: null`. This means the third amplification intervention (α=+3σ) was planned but not completed. The partial verdict does not rely on amplify_x3 and the conclusion is unaffected, but this is a scope gap relative to the stated protocol.

**WARN: Single seed.** All interventions used teacher_seed42 only. The plan calls for ≥3 seeds. This is honestly documented in CLAIMS_LEDGER.md C3 caveat: "Single-seed intervention (seed 42 only) — planned protocol called for ≥3 seeds; flagged suspected under-power."

### D. Dead Code Detection: PASS

- `mechanism_intervene.py:main()` → called for each intervention mode; verdict.json exists for each completed run ✓
- `compute_m2_verdict.py:main()` → called; `m2_verdict.json` exists ✓
- `pipe_with_cfg` wrapper → called at `mechanism_intervene.py:208-218` ✓
- `_estimate_sigma()` → called at `mechanism_intervene.py:193` ✓
- `_hook()` → called at `mechanism_intervene.py:197` ✓

### E. Scope Assessment: WARN

- **Seeds**: 1 of 8 (seed 42 only; plan requires ≥3). Explicitly flagged as suspected under-power.
- **Interventions**: 5 of 6 planned (amplify_x3 null; amplify_x3 is the α=+3σ point). Dose-response covers {ablate, baseline, amplify_x2, amplify_x4}.
- **Negative amplification**: the dose-response grid from MECHANISM_ROUTING.md §M2 specifies `α ∈ {0, ±0.25, ±0.5, ±1, ±2, ±3, ±4}`. Only positive amplification (+2σ, +4σ) and ablation (which is directional removal, not negative amplification) were tested. Negative doses (suppressing the direction below natural levels) were not run.
- **Claim scope language**: CLAIMS_LEDGER.md C3 statement explicitly tags "[suspected under-power: single-seed intervention]" — no overclaiming.

### F. Evaluation Type

**Classification: synthetic_proxy** — same gpt-5.4 judge as M0. Appropriate for the same reasons as C1.

## Action Items

1. Run M2 on additional seeds (≥2 more, e.g., seeds 43 and 44) to test whether the specificity failure and amplification sign reversal are seed-stable. This is the suggested variant for C3 if picked in Phase 3.
2. Run amplify_x3 to complete the dose-response grid.
3. Consider running negative-dose amplification (suppressing the banana direction) to complete the dose-response characterization.
