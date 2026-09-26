# Mechanism Audit — C3: Low-Dim Safety Substrate (CAA / Steering Vectors)

**Claim**: A low-dim safety-relevant activation subspace inside the Qwen3.5-9B language tower shifts between the treated (subliminal-SFT) student and the Ctrl base student (Location, C3a), and intervening on that subspace on treated restores Ctrl-level image-conditioned QA_I accuracy while a rank-matched non-safety-relevant control direction achieves <1/3 the effect and general-capability (MMLU-slice) drop stays <= 2 pp (Causal Intervention specificity, C3b).

**Milestones scoped**: M1 (Location, diff-of-means direction extraction), M2 (Causal Intervention — steering, ablation, patching)

**Committed mechanism family**: Representation and Parameter Analysis / Steering Vectors (CAA)

**Audit date**: 2026-07-10
**Auditor**: /mechanism-audit (auto-verify Phase 2, C3)

---

## Check A: Steering Coefficient Sweep

This check audits whether the steering coefficient α was properly tuned per the CAA family's requirements.

### A.1 Alpha sweep coverage (>= 3 orders of magnitude / >= 3 distinct alpha points)

**Finding**: WARN

The M2 experiment sweeps α ∈ {-2, -1, 0, +1, +2} — 5 distinct values. This is a reasonable 5-point sweep. However, the range spans only [-2, +2] in σ_proj-normalized units. The CAA family's `experiment-tips/steering-coefficient-tuning` recipe recommends sweeping at least 3 orders of magnitude to reliably locate the plateau between "too small" (no effect) and "too large" (capability collapse). A ±2σ range may be too narrow — the results show gc values of {0.000, 0.125, 0.125, 0.125, 0.250} for real-direction steering, with NO plateau visible (gc is monotonically increasing through α=+2, the edge of the sweep). This suggests the plateau may lie beyond the tested range.

Verdict on coverage: **WARN** — 5 points were tested but range appears truncated (no plateau detected within the sweep; steering at α=+2 is still increasing).

### A.2 Sigma-proj normalization

**Finding**: PASS

The steering application in `scripts/mechanism_m2_intervene.py` uses `h ← h - α · σ_proj · u` where `σ_proj` is the projected standard deviation of the Ctrl residual stream activations onto the direction u. This is the σ-scaled steering formula from the `experiment-tips/steering-coefficient-tuning` recipe. The formula is correctly σ_proj-normalized. Confirmed in MECHANISM_ROUTING.md's "Verify" section: "α ∈ {-2, -1, 0, +1, +2} in σ_proj units (so α=1 means +1σ of on-direction magnitude)."

### A.3 Capability / coherence metric logged at every alpha

**Finding**: FAIL

The M2 sweep logs QA_I accuracy at each α for the `real_treated` arm. However:
- The capability/coherence control metric (MMLU-slice accuracy) was explicitly NOT run — it was skipped once SP-A specificity already failed (tracker row 49). The `experiment-tips/steering-coefficient-tuning` recipe requires a capability metric be logged "at every point" in the sweep to distinguish "steering recovers safety accuracy" from "steering causes general-capability collapse that suppresses all outputs."
- Without the MMLU control at every α point, we cannot rule out that the non-monotonic QA_I response under steering reflects capability collapse rather than a targeted safety-substrate intervention.
- Cross-seed replication (seed200, seed300) was done only at α ∈ {-1, -2}, not the full 5-point sweep.

**Severity**: FAIL — capability metric not logged at every α point (MMLU skip is a hard rigor gap). This is not a documentation issue; it is a measurement gap that makes the steering results uninterpretable as a clean specificity test.

### A.4 Alpha locked mid-plateau

**Finding**: FAIL

No plateau was identified in the M2 steering sweep (gc values are non-monotonic: {0.000, 0.125, 0.125, 0.125, 0.250} — the highest gc is at the boundary α=+2, not in the interior). Consequently, no "mid-plateau" α was locked. The MECHANISM_ROUTING.md's plan required "lock α mid-plateau" but this step was never possible given the truncated sweep.

**Severity**: FAIL — plateau not identified (sweep is too narrow or direction is non-effective for additive steering); no mid-plateau lock.

### A.5 Random-direction control baseline

**Finding**: PASS (with caveat)

The `random_matched` control direction was implemented and tested. At α=-1, the random_matched achieves gc=0.250, matching the best real-direction steering result (gc=0.250 at α=+2). The SP-A specificity gate thus FAILS — as correctly reported. The control is properly implemented and its result is correctly interpreted as a specificity failure.

Caveat: the random direction control was only tested over the same 5-α sweep range, inheriting the range-truncation concern from A.1. However, the SP-A failure is properly detected even within the truncated range.

### A.6 Sign pattern preservation for asymmetric protocols

**Finding**: N/A

C3 uses a symmetric intervention (both negative and positive α are tested for the same direction). The protocol is not asymmetric in sign convention. This check is not applicable.

---

## Summary of Check A Sub-findings

| Sub-check | Result | Notes |
|---|---|---|
| A.1 Alpha range (≥ 3 orders of magnitude) | WARN | 5-point sweep [-2, +2] in σ_proj units; no plateau visible; range may be truncated |
| A.2 Sigma-proj normalization | PASS | σ_proj-scaled formula correctly implemented |
| A.3 Capability metric at every α | FAIL | MMLU specificity check skipped; cannot rule out capability collapse driving non-monotonic QA_I response |
| A.4 Mid-plateau lock | FAIL | No plateau identified; locked α not possible |
| A.5 Random-direction control | PASS | SP-A control correctly implemented; failure correctly detected |
| A.6 Sign pattern (asymmetric) | N/A | Symmetric protocol |

**Check A overall**: FAIL (A.3 and A.4 are both FAIL; per max_severity, Check A = FAIL)

---

## Checks B–F: Reserved

All reserved checks return `not_implemented` per current skill version.

---

## Key mitigating context (per suspected_under_power carrying rule)

The ablation and patching results (gc=0.875 and gc=0.625) are NOT subject to the steering coefficient sweep. These intervention shapes do not use α — they apply a deterministic projection removal or residual-state replacement. They are methodologically cleaner than steering and their strong gap-closure results are not invalidated by the MMLU skip or plateau issues. However, the C3b claim specifically mentions "intervening on that subspace on treated restores Ctrl-level QA_I accuracy while a rank-matched non-safety-relevant control direction achieves <1/3 the effect" — the specificity claim is framed around the direction itself (as a steering vector), not around magnitude removal (ablation) or activation replacement (patching). The ablation/patching results are therefore positive evidence for "the layer-4 representation carries the effect" but do not directly rescue C3b's specificity predicate.

The plan's fallback "distributed rewrite" negative result (C3b specificity refuted, ablation/patching positive) is the correct characterization. The mechanism audit finds FAIL on the steering sweep rigor, which is consistent with the mechanism being distributed rather than captured by a single low-dim direction.

---

## Overall Verdict

**overall_verdict**: FAIL

The mechanism audit finds FAIL on Check A due to:
- A.3: MMLU capability metric not logged at every α point — a required measurement for the steering specificity interpretation.
- A.4: No plateau identified in the 5-point sweep — α locking was never possible, indicating either a truncated range or a steering-inert direction.

These failures are consistent with the main experiment's own "distributed rewrite" negative narrative: the direction is not a clean steering handle. However, per the audit protocol, these failures mean the C3 main experiment's mechanism rigor is FAIL, making C3 INCONCLUSIVE for verify purposes (the mechanism test was not properly controlled). Iteration should either (a) run the MMLU control at every α point to complete the specificity check, or (b) pivot the C3 claim to the ablation/patching result (which does not require the steering sweep).
