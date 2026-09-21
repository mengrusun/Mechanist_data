# Review Summary

**Problem**: Validate a *given* behavior — cross-modal subliminal transfer of an unsafe behavior via a text-only teacher-generated channel in a fixed Qwen3.5-9B multimodal setup — then discover the mechanism if the M0 gate passes.
**Initial Approach**: 3-arm × ≥3-seed × ≥3% gap M0 gate + Location→Causal Intervention mechanism arc.
**Date**: 2026-07-17
**Rounds**: 3 / 5
**Final Score**: 9.0 / 10
**Final Verdict**: READY (soft — three IMPORTANT final-closure fixes folded into FINAL_PROPOSAL.md)

## Problem Anchor (frozen, verbatim across all rounds)

Validate: in a fixed Qwen3.5-9B → Qwen3.5-9B (matched-initialization) multimodal transfer setup, text-only teacher-generated data that has been surface-filtered to look safe (via `filter_prompts_lenient.md`) covertly transmits an unsafe behavior to the multimodal student, measurable as `Acc(QA_I)_Ctrl-A − Acc(QA_I)_treated ≥ 3 %` AND `Acc(QA_I)_Ctrl-B − Acc(QA_I)_treated ≥ 3 %`, per seed across ≥ 3 random seeds, with filter re-scan clean.

## Round-by-Round Resolution Log

| Round | Main Reviewer Concerns | What This Round Simplified / Modernized | Solved? | Remaining Risk |
|---|---|---|---|---|
| 1 | CRITICAL feasibility: per-seed teacher retraining is wasteful. IMPORTANT M0 semantics: `conditional` verdict softens the gate. IMPORTANT method specificity: logit-margin, intervention site, top-K, off-target eval underspecified. IMPORTANT validation focus: judge-audit 3% veto too brittle; bootstrap CI acting as quasi-gate. MINOR: harmonize success wording; L-Core relabel as contrastive direction extraction. | Removed per-seed teacher retrain (once fixed T*). Made M0 gate binary PASS/FAIL/RUN INVALID. Pinned logit-margin metric + intervention site + top-K + off-target eval choice. Judge audit → calibration matrix; bootstrap CI → readout. Success wording harmonized (bounded null allowed). L-Core = contrastive direction extraction; LoRA attribution demoted to fallback. | yes | seed protocol still ambiguous on "replacement"; judge audit still baked into PASS |
| 2 | IMPORTANT: pre-register exactly 3 seeds; no scientific replacement. IMPORTANT: judge audit → measurement-validity ONLY. IMPORTANT: mechanism verdict pre-registered as 3-level. MINOR: single primary location metric. IMPORTANT: cache policy binding. MODERNIZATION: Borda aggregation; linear probe; isotonic-fit deviation. | Seeds pre-registered `{42, 123, 2026}`; RUN INVALID = same-seed rerun; scientific FAIL is FAIL (no replacement). Judge audit refactored to measurement-validity only. Mechanism verdict pre-registered STRONG POSITIVE / PARTIAL POSITIVE / BOUNDED NULL with concrete thresholds. Primary location metric = accuracy-conditioned activation contrast on flipped-wrong subset. Cache policy binding (bf16, pinned-site, ≤ 4000 items/seed). Borda + linear probe + isotonic added. | yes | pooled Spearman under-specified; d_diff vs d_pca merge rule fuzzy; feasibility needs worst-case table |
| 3 | IMPORTANT: pooled Spearman needs exact recipe. IMPORTANT: pin d_diff as sole intervention direction. IMPORTANT: feasibility worst-case table + hard-stop rule. MINOR: Ctrl-C → appendix; judge matrix as flag + one summary table. | Per-seed Spearman ρ_s over 7-point α curve; formal monotonicity pass = median ρ ≤ −0.5 AND ≥ 2/3 seeds ρ < 0. `d_diff` = sole intervention direction; d_pca + probe = diagnostic-only. Worst-case budget table with n_layers ≈ 40, d_model ≈ 5120; hard-stop rule: if L-Secondary would push > 60 GPU-hours, L-Core-only in main body, L-Secondary → appendix. Ctrl-C → appendix; judge matrix as measurement-validity flag + one table. | yes | none unresolved |

## Overall Evolution

- **From "sketchy pilot-style plan" to "engineer-implementable protocol"**: round 1 hardening removed the biggest waste (per-seed teacher retrain) and made the gate binary; round 2 pre-registered the seeds + mechanism verdict hierarchy + cache policy; round 3 closed remaining statistical ambiguities and added the worst-case budget.
- **Dominant contribution never widened**: throughout, the paper is *refined verification + minimal conditional mechanism arc*, never *"validate + then explore several mechanism families"*.
- **Complexity strictly decreased or held constant** across rounds: mean-across-seeds bar removed; strict intersection replaced with Borda (fewer researcher DOF); dual location metric collapsed to single; L-Secondary trigger to one sentence; Ctrl-C to appendix.
- **Modern leverage improved without inflation**: contrastive direction extraction + Borda rank aggregation + signed linear-probe companion diagnostic + isotonic-fit deviation are all 2025-era natural choices; nothing forced.
- **Drift avoided at every round**: `Drift Warning: NONE` in all three rounds.

## Final Status

- **Anchor status**: preserved. The scientific pass predicate is exactly task.md's per-seed dual inequality across the 3 pre-registered seeds + filter re-scan clean.
- **Focus status**: tight. One dominant contribution + one supporting hardening bundle + explicit non-contributions list.
- **Modernity status**: appropriately frontier-aware. Uses 2025-era primitives at the right altitude; no submethod pinning at claim stage (deliberately left for `/mechanism-skills` routing).
- **Strongest parts of final method**: (i) the binary PASS/FAIL/RUN INVALID gate with measurement-validity separated from scientific verdict; (ii) the pre-registered 3-level mechanism verdict hierarchy with machine-checkable thresholds; (iii) the worst-case budget with hard-stop rule; (iv) the Borda + probe companion diagnostic pattern.
- **Remaining weaknesses**: `n_layers` / `d_model` in the budget table are assumed at "≈ 40 / ≈ 5120" — the exact Qwen3.5-9B config should be pinned at `/auto-experiment` Phase 1.5 routing time; nothing else outstanding.
