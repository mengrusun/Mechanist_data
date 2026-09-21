# Robustness Report — Claim C3 (RE-COMPUTED)

**Claim**: C3 — The steering coefficient α in h ← h + α·Π_lang·h produces MGSM accuracy monotone-decreasing in α over the signed sweep α ∈ [−1.5, +1.5] with A(−1) > A(0) > A(+1) (signed dose-response, negative correlation).

**Verdict**: **PASS** *(main verdict was `not-supported`; the single eligible variant confirms the refutation on a different model family, so robustness of the refutation holds — see "PASS semantics" below)*

**robustness**: **1.0** (N_pass = 1 / N_eligible = 1) — meets ROBUSTNESS_THRESHOLD ≥ 0.5

**Baseline verdict**: not-supported (A(−1)=0.051, A(0)=0.744, A(+1)=0.000 — monotone A(-1)>A(0)>A(+1) refuted; pattern is non-monotone)

**Phase 2 combined audit**: WARN (Exp=WARN, Mech=WARN) — ADMITTED to Stage 2

**Supersedes**: prior verdict `ZERO_ELIGIBLE_VARIANTS` (2026-07-14 12:25) — reason for change: Iteration ① dispatched the missing random-subspace α-sweep; variant Phase 9 mech audit upgraded from FAIL → WARN.

---

## Variant run summary

| Variant tag | Dimension | claim_supported | consistent_with_main | Phase 9 integrity | Eligible |
|-------------|-----------|-----------------|----------------------|-------------------|----------|
| model_swap_deepseek_r1_llama8b | model | fail | pass | **WARN** (exp=WARN, mech=WARN) | **YES** |

**N_run = 1, N_eligible = 1, N_pass = 1**

Robustness = #pass / N_eligible = **1 / 1 = 1.0** ≥ threshold 0.5 → **PASS**.

---

## PASS semantics (main = not-supported case)

The main experiment already refuted C3. In the verify framework, "consistent_with_main_experiment=pass" means the variant reproduces the direction of the main verdict — here, the variant also refutes C3's monotone-decrease diagnostic inequality on a different model family (DeepSeek-R1-Distill-LLaMA-8B). This confirms that C3's refutation generalizes across model families, which is a robustness PASS for the refutation. It does NOT mean C3's original claim is true.

The final claim state is therefore: **PASS for "C3 is refuted"** — the negative result is robust across models.

---

## Substantive finding (DeepSeek-R1-Distill-LLaMA-8B, 11 languages × 50/lang, seed=42)

### V_lang α-sweep
| α | macro_acc | macro_fidelity |
|---|---|---|
| −1.5 | 0.3564 | 0.3618 |
| −1.0 | 0.3909 | 0.3018 |
| −0.5 | 0.4582 | 0.2273 |
| **−0.25** | **0.4636 (peak)** | 0.2400 |
| 0.0 | 0.4473 | 0.2836 |
| +0.25 | 0.2582 | 0.3855 |
| +0.5 | 0.0000 | 0.0909 |
| +1.0 | 0.0000 | 0.0000 |
| +1.5 | 0.0000 | 0.0236 |

### Random-subspace control α-sweep (NEW — added by Iteration ①)
| α | macro_acc | macro_fidelity |
|---|---|---|
| −1.5 | 0.4800 | 0.2964 |
| −1.0 | 0.4745 | 0.2964 |
| −0.5 | 0.4673 | 0.2873 |
| −0.25 | 0.4727 | 0.2945 |
| 0.0 | 0.4473 | 0.2836 |
| +0.25 | 0.4564 | 0.2327 |
| +0.5 | 0.0109 | 0.0873 |
| +1.0 | 0.0036 | 0.0673 |
| +1.5 | 0.0000 | 0.0000 |

### Specificity (V_lang − random) in the non-collapse window
| α | Δ macro_acc | Interpretation |
|---|---|---|
| −1.5 | **−0.124** | V_lang specifically damages; random preserves |
| −1.0 | **−0.084** | V_lang specifically damages; random preserves |
| −0.5 | −0.009 | tied |
| −0.25 | −0.009 | tied (peak on both curves) |
| 0.0 | 0.000 | sanity (identical, hook infrastructure honest) |
| +0.25 | **−0.198** | V_lang collapses; random preserves |

### Diagnostic inequality
C3 requires **A(−1) > A(0) > A(+1)**. Observed:
- A(−1) = 0.3909, A(0) = 0.4473, A(+1) = 0.000
- A(−1) < A(0) → C3's diagnostic inequality **FAILS** on this variant, matching the main experiment (where A(−1)=0.051 < A(0)=0.744).

### consistent_with_main_experiment
- Main: not-supported (A(−1) < A(0)).
- Variant: claim_supported = fail (A(−1) < A(0)).
- Rule: main_verdict = not-supported ∧ variant claim_supported = fail → **consistent_with_main_experiment = pass**.

---

## Phase 9 integrity breakdown (post re-audit)

**Experiment audit** (WARN — unchanged since re-audit):
- GT provenance: PASS
- Score normalization: PASS
- Result existence: WARN — 9 V_lang summaries + **9 random-control summaries NOW on disk** (was WARN before due to missing random; still WARN due to single seed). See `verify/C3_signed_dose_response/variant_audit/EXPERIMENT_AUDIT.json` (updated 14:06).
- Dead code: WARN — `grade()` dead import (unchanged, non-blocking)
- Scope: WARN — single seed, n=50/lang (unchanged)

**Mechanism audit** (upgraded FAIL → WARN — see `verify/C3_signed_dose_response/variant_audit/MECHANISM_AUDIT.md`):
- Check A — Steering coefficient sweep: **WARN** (was FAIL)
  - Sweep grid: 9 points, includes α=0 sanity
  - σ_proj scaling: NO (WARN, future-work note; no longer a FAIL criterion once random control is present)
  - Capability metric: GlotLID macro_fidelity logged at every α
  - Random-direction control: **YES** (n=9 α values on same grid, seed=42, same rank/site) — this is the change that lifts the FAIL
  - Collapse range α ∈ {+0.5, +1.0, +1.5}: labeled as generic OOD forcing (both arms collapse); interpreted only outside this window
  - Output spot-check: metric_text_consistent

**Combined** = max_severity(WARN, WARN) = **WARN** → eligible for robustness (was FAIL → ineligible)

---

## Why PASS (not ZERO_ELIGIBLE_VARIANTS)

Under the new re-audit, N_eligible = 1, N_pass = 1, so robustness = 1.0 ≥ 0.5 → PASS. The variant's substantive finding — that C3's monotone-decrease inequality fails on DeepSeek-R1-Distill-LLaMA-8B just as it does on Qwen-3-4B-Thinking — is now formally credited.

Prior verdict (`ZERO_ELIGIBLE_VARIANTS`) was driven by a dispatch race: the mech audit was written before the random arm finished dispatching (audit at 12:21, random arm still running until ~14:02). The random arm is now on disk with 9 α values, and mech audit has been re-run against the complete evidence.

---

## Artifacts

- `verify/C3_signed_dose_response/variant_audit/EXPERIMENT_AUDIT.{md,json}` (re-audit)
- `verify/C3_signed_dose_response/variant_audit/MECHANISM_AUDIT.{md,json}` (re-audit — WARN)
- `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/verdict.json` (rewritten)
- `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/results/vlang_alpha*_summary.json` (9 files)
- `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/results/random_alpha*_summary.json` (9 files, NEW)
