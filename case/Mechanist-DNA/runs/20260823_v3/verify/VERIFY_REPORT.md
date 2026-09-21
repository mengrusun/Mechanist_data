# Verification Report — Steering Evo2-7B toward high α-helical content

**Date**: 2026-08-24
**Swap variants**: true (full Stages 1–3)
**Dimensions tested**: model  (variants/claim = 1; method + dataset axes not requested this pass)
**Threshold**: robustness ≥ 0.50, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: per-claim combined verdict — C1 PASS, C2 WARN; per-sub-audit breakdown in `INTEGRITY_AUDIT.md`.

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|-------------------|-------------------------|-----------------------------------------------|---------------------------------------|--------------------------------|------------|-------|-------|
| C1 | helix gain / monotone dose-response | supported | PASS (exp PASS / mech PASS) | clean | 1/1 | 1.00 | ✅ PASS | model swap `evo2_7b`→`evo2_7b_262k` reproduces the dose-response (ρ=0.878, p=0.0057) and winning-coef gain (+0.094, δ=0.201) — robustly positive across the within-family checkpoint swap |
| C2 | validity non-inferior / off-target | supported | WARN (exp WARN / mech N/A) | — | — | — | ⚪ INTEGRITY_ONLY (skip=max_verify_claims_cap) `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment]` | admitted by Phase 2 but not the top-1-by-importance pick under MAX_VERIFY_CLAIMS=1; WARN = "off-target not degraded" wording vs documented −37% length / −0.139 GC shift. Swap-test later: `/auto-verify C2 — resume: true` |

> **Column glossary** — *Main-experiment verdict*: the claim's own conclusion (`main-experiment-verdicts.json`). *Main-experiment integrity*: `max_severity(/experiment-audit, /mechanism-audit)` on `refine-logs/` scoped to the claim. *Variant integrity*: same rule on the claim's variants (`clean` = all PASS). *Eligible variants*: `N_eligible/N_run` surviving Phase 9. *Robustness*: `#pass / N_eligible` over `consistent_with_main_experiment`. *State*: PASS = main-experiment verdict robust under swaps; INTEGRITY_ONLY = Phase 2 PASS/WARN but Stage 2 intentionally skipped.

## Integrity Audit

**Overall**: WARN — see `verify/INTEGRITY_AUDIT.md` for full Phase 2 (main experiment) + Phase 9 (variant) findings. The WARN is entirely on the **main-experiment** side (C2's off-target scope-wording); the C1 variant is integrity-clean.

## Stage-2 Selection

Phase 3 step 0 picked **1 of 2** admitted claims for Stage 2 (cap = MAX_VERIFY_CLAIMS = 1). See `verify/STAGE2_PICK.json`.

**Picked** (with importance rationale):
- **C1**: the central causal claim of the round — a located CAA direction, added during decoding, raises α-helix with a monotone dose-response. C2 is the validity guardrail that only qualifies C1's winning setting, so the primary claim is swap-tested first.

**Stage-2-deferred (marked INTEGRITY_ONLY with stage2_skip_reason: max_verify_claims_cap)**:
- **C2** — validity non-inferior / off-target not degraded (main-experiment verdict: supported, integrity WARN) — swap-test later via `/auto-verify C2 — resume: true`.

## Model-axis outcome (which axis actually ran)

The requested **model axis RAN** — it was not excluded, so the cheap method-axis fallback (length-regressed endpoint / alt-pLDDT on existing generations) was **not** needed. The alternate within-family checkpoint `evo2_7b_262k` (262k-context, `configs/evo2-7b-262k.yml`) loaded offline in 40 s (`verify/variant_smoke/smoke_262k.json`) and ran the full localize→steer→ESMFold-score recipe, scoped small to budget (site-28 CAA, coefs [0,1,2,4] × 2 seeds × 150 gens = 1200 generations). Both `evo2_7b` and `evo2_7b_262k` are 7B StripedHyena checkpoints (same 32-block/4096-hidden architecture) → a genuine within-family model swap, not cross-family.

## Details
- C1: `verify/C1_helix_gain_doseresponse/ROBUSTNESS.md` · variant `verify/C1_helix_gain_doseresponse/variants/model-swap-evo2-7b-262k/`
- C2: `verify/C2_validity_preserved_noninferior/ROBUSTNESS.md` (audit-only / INTEGRITY_ONLY)

## Compute
- Verify GPU-hours: **≈ 1.47** (load-smoke ≈ 0.01 + variant 1.46; `runs/verify_C1_model_swap_262k/cost.json`, `gpu_ids=[3]`).
- Round total: ≈ 27.3 (experiment) + 1.47 (verify) = **≈ 28.8 / 40 GPU-h** → **≈ 11.2 GPU-h remain** for the downstream iteration stage. HARD cap respected; ≤ 8 cards respected (used only GPU 3).

## Next Step

→ **C1 PASS** — the central helix-gain / dose-response claim is robust under the model swap. Proceed to `/auto-iteration-loop`. The robustness story (direction transfers across the Evo2-7B family, with the length/GC caveat also transferring) goes into the next-round draft / paper polish.

→ **C2 INTEGRITY_ONLY** (`stage2_skip_reason: max_verify_claims_cap`) — no back-edge action required. Record under Open Items with the upgrade suggestion `/auto-verify C2 — resume: true` (Phase 2 audit reused; Stages 2–3 execute for C2). C2's Phase-2 WARN (off-target "not degraded" wording vs documented length/GC shifts) is the round's main interpretability caveat and is the natural target of a future C2 swap test on the length-regressed endpoint / alternate pLDDT policy.
