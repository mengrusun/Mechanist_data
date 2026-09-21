# Auto Iteration Final Report — Steerable Social-Variable Directions in an LLM Dictator

- **Generated**: 2026-07-13T21:15+08:00
- **Iterations consumed**: 2 / 6
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 6 / 10
- **Final canonical verdict**: **almost**
- **Termination reason**: **positive_verdict** (three-dimensional STOP rule fired at Iteration 3 re-review)
- **Cumulative cost**: runs_total = 4 (2 GPU experiments + 1 analytic LOPO + 1 layer-picker diagnostic), gpu_hours_total = **0.883**
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

The autonomous review loop consumed 2 of 6 iterations and ~0.9 of ~3 available GPU-hours to close the two most decisive weaknesses flagged by the external reviewer at Iteration 1: the **missing L=16 random-direction control for C3** and the **absent L=16 4×4 selectivity sweep for C4**. Iteration 2 further extended the random-direction null from 3 to 10 seeds (yielding ≥2.5σ direction-specificity for all four showcase C3 effects, including the V=M sign inversion) and performed a Leave-One-Phrasing-Out (LOPO) stress test that empirically confirms the reviewer's shallow-layer lexical-confound suspicion (V=M ell=2: LOPO=0.483 — chance!) while identifying a productive mid-layer band [12–24] that includes L=16, retroactively justifying M4-supp's layer choice. All four target claims (C1–C4) retained their `/auto-verify` state (C3 PASS; C1/C2/C4 INTEGRITY_ONLY under cap); no claim was rewritten (no ③ actions). The reviewer's final score rose from 4/10 → 5/10 → 6/10 across the three review rounds; final verdict is "almost" with remaining recommendations reduced to narrative/manuscript-integrity fixes (paper must reframe as "direction-specific causal steering at productive mid-layers" and narrow claims on purity and cross-variable selectivity). C4 selectivity at L=16 remains statistically borderline (p=0.128, better than shallow p=0.84 but not significant with n=4 variables); C2's raw-vs-decorrelated conceptual mismatch is the largest unclosed concern for future work.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS (C3) | 1 | 1 PASS (held; strengthened by L=16 random-direction null n=10 at ≥2.5σ) |
| FAIL | 0 | — |
| INCONCLUSIVE | 0 | — |
| ZERO_ELIGIBLE_VARIANTS | 0 | — |
| INTEGRITY_ONLY (C1, C2, C4) | 3 | 3 INTEGRITY_ONLY (unchanged — no back-edge action is legal for this bucket by the loop's policy) |
| Legacy DEFERRED | 0 | — (empty in new runs) |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C3` — Bidirectional causal steering at inference time

- **Original robustness signal** (from `verify/C3_bidirectional_causal_steering/ROBUSTNESS.md`): robustness=1.00; variants_passed=1/1 (`model-swap-meta-llama3-8b`, Meta-Llama-3-8B-Instruct); V=M sign-inverts at α=+2 at L=16 in BOTH main and swap models.
- **Reviewer consistency check across iterations**: baseline confidence 0.72/1.00 at Iteration 1 (with caveat: n=10/cell in swap; no L=16 random-direction control; α range ±2σ only; raw v̂_V used). Iteration 2 extended random-direction null to n=10 seeds → V=M α=+2 sign inversion at **2.84σ** (no random seed inverted); V=A α=+2 at 2.69σ, V=G α=−2 at 4.30σ, V=I α=−2 at 2.50σ. All 4 showcase effects direction-specific.
- **Touched in iterations**: [1, 2]
- **Final status**: **PASS (held; strengthened)**.
- **Notes for downstream**:
  - "Bidirectional" wording must be replaced with **"sign-asymmetric direction-specific causal steering"** — each V responds strongly in one signed α direction, not symmetrically in both.
  - L=16 was a post-hoc M4-supp layer; iteration validates it as a principled representative of the productive layer band [12–24] via LOPO + σ_proj-headroom analysis.
  - Raw v̂_V (not GS/LEACE-purified) was used at L=16. C3 does not directly validate C2's pure-direction narrative.
  - Swap variant used n=10/cell; PASS is qualitative pattern-match.

---

## Section 2 — FAIL Claims (full journey)

*(No FAIL claims — section intentionally empty.)*

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

*(No INCONCLUSIVE claims — section intentionally empty.)*

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

*(No ZERO_ELIGIBLE_VARIANTS claims — section intentionally empty.)*

---

## Section 4b — INTEGRITY_ONLY Claims (Stage 2 skipped by cap; no back-edge action)

Per loop policy, INTEGRITY_ONLY claims receive **no back-edge action** — main-experiment integrity was validated at Phase 2 (WARN, not FAIL) and Stage 2 was intentionally skipped. Recorded here for the orchestrator's `CLAIMS_LEDGER.md` Open-Items surface.

### 4b.1 `C1` — Linear encoding of each social variable
- `stage2_skip_reason`: `max_verify_claims_cap` (Stage 1 admitted; not top-K picked)
- `main_experiment_integrity`: warn (warn_source: experiment — proxy GT tau not labeled; 48 prompts appear in both train/held splits; shallow layer pick)
- **Iteration-2 LOPO stress test finding** (relevant caveat for the paper, does NOT change verify state): at the original picked layers, LOPO cross-val vs random 5-fold shows V=G OK (drop 0.066), V=A/I WARN (drop 0.17-0.22), and **V=M FAIL (LOPO=0.483, chance-level)** — the shallow-layer C1 signal for V=M is essentially a phrasing detector. Best LOPO layers per V are in the mid-band 10-14, where V=I/M generalize at ≥0.93; V=G/A best at layers 28-32 (LOPO 0.90-0.93). At L=16: V=I=0.927 (strong), V=G=0.679, V=A=0.719, V=M=0.707 (moderate).
- **Upgrade to stress-test under swaps**: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim).

### 4b.2 `C2` — Purity via decorrelation
- `stage2_skip_reason`: `max_verify_claims_cap`
- `main_experiment_integrity`: warn (warn_source: experiment — GS off-diag 0.506 borderline; LEACE norm inflation; 1-D scalar-projection probe is conservative)
- **Reviewer's largest unresolved concern**: C2 claims "pure" directions but C3 causal success is demonstrated with raw v̂_V, not GS/LEACE. Recommended paper-scope narrowing: describe GS as a "partially decorrelated descriptive basis" (off-diag max 0.506 residual coupling) rather than "pure directions free of confounds." Alternatively, a future work should run C3-style steering with GS-purified directions at L=16 to empirically bridge C2 and C3.
- **Upgrade**: `/auto-verify C2 — resume: true`.

### 4b.3 `C4` — Selectivity 4×4 matrix
- `stage2_skip_reason`: `max_verify_claims_cap`
- `main_experiment_integrity`: warn (warn_source: experiment+mechanism — synthetic_proxy GT; min-diagonal=0 artifact at shallow layers; inherits missing L=16 random-direction control)
- **Iteration-1 finding** (relevant caveat, does NOT change verify state): 4×4 selectivity at L=16 tested for the first time. permutation p=0.128 (vs shallow p=0.84 — order-of-magnitude improvement, still not stat-sig with n=4 V). Per-V diag/mean-off ratio at α=+2σ: V=A 2.25, V=M 2.19 (both selective); V=I 1.00 (non-selective — mostly co-modulates other Ws); V=G 0.00 at +σ (works at −σ diag=−0.570). **Partial selectivity with substantial collateral coupling**, especially for V=I.
- **Upgrade**: `/auto-verify C4 — resume: true`.

---

## Section 5 — Legacy DEFERRED Claims

*(Empty under current verify architecture; MAX_VERIFY_CLAIMS-cap claims are surfaced as INTEGRITY_ONLY with `stage2_skip_reason: max_verify_claims_cap` in Section 4b.)*

---

## Section 6 — Cross-Cutting Patterns

- **Pattern P1**: The original pipeline systematically selected shallow layers (ell_V* ∈ {2,4,6}) for its main analysis and then discovered the real intervention effects at a supplementary deeper layer (L=16) whose matching controls were not initially re-run there. Iteration 1 closed the L=16 random-direction and C4 selectivity gaps; Iteration 2's LOPO analysis retroactively validated L=16 as a principled representative of the productive layer band [12–24]. **Resolved by iteration**.
- **Pattern P2**: Narrative caveats were preferred over stress tests in the original pipeline (e.g., "layer-pick caveat" text was added but no re-run performed). Iteration 2 broke this pattern by launching a real LOPO stress test that empirically confirmed the reviewer's Suspicion 1 for V=M at ell=2 (chance-level LOPO) and disconfirmed it at mid-layers. **Resolved by iteration**.
- **Pattern P3**: C3's causal success uses **raw v̂_V** but C2 claims **decorrelated/pure directions**. The paper's mechanistic narrative bundles these but the empirical evidence does not. **NOT resolved** — surfaces as a paper-scope Open Item and the reviewer's largest remaining concern. Would require a C3-style steering experiment with GS-purified directions at L=16 for a full empirical closure.
- **Pattern P4**: Reviewer scores rose 4/10 → 5/10 → 6/10 across three review rounds. Two of the three suspicion clusters (procedural gaps + null power) were fully closed by real experiments; the third (conceptual C2-C3 alignment) remained narrative-only and remains open.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 2 / 6
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 4 (2 GPU experiments + 1 analytic LOPO + 1 layer-picker diagnostic; count reflects distinct output-producing dispatches).
- **Iteration GPU-hours**: gpu_hours_total = **0.883** (well under the ~3 GPU-h iteration budget; ~2.1 GPU-h headroom vs the 10-h HARD budget). No pin propagation failure — every dispatch used `CUDA_VISIBLE_DEVICES` within the allowlist {1,2,3,5,6}.

### Per-iteration breakdown (from `iteration_breakdown[]`)

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ② main-experiment augment (L=16 random-direction control + L=16 C4 4×4 selectivity) | C3, C4 | — | 2 | 0.678 | 5 | not ready |
| 2 | ① extended random null (n=3→10) + ⓪ C1 LOPO stress test + ⓪ layer-picker diagnostic | C1, C3 | — | 2 | 0.205 | 6 | almost |
| (3) | pure re-review, no back-edge (does NOT consume budget) | all | — | 0 | 0.000 | 6 | almost — **STOP** |

---

## Section 8 — Open Items for Human Reviewer

- **Still-FAIL claims**: none.
- **Still-INCONCLUSIVE claims**: none.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - `C1` [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run). [main-experiment integrity: warn; warn_source: experiment]
  - `C2` [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true`. [main-experiment integrity: warn; warn_source: experiment]
  - `C4` [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C4 — resume: true`. [main-experiment integrity: warn; warn_source: experiment+mechanism]
- **Legacy deferred claims**: none (empty under current verify architecture).
- **Recurring unresolved patterns**:
  - **P3 (C2 raw-vs-pure conceptual mismatch)**: causal steering at L=16 uses raw v̂_V while C2 claims "pure via decorrelation." Paper-scope fix: narrow C2 to "GS-decorrelated basis with off-diag 0.506 residual coupling"; or run a C3-style steering experiment at L=16 with GS-purified directions to bridge the two claims empirically.
- **Claim-reentry refusals** (③ requested but sub-budget exhausted): none.
- **Reviewer's final manuscript-scope recommendations** (all ⓪ narrative or paper-scope, not experimental fixes):
  - **CLAIM-SCOPE**: replace "bidirectional control" with "sign-asymmetric causal steering"; replace "selective" with "partially selective / entangled".
  - **C2-CONCEPT**: demote purity/disentanglement claims (or run purity intervention experiment at productive layer band as future work).
  - **C4-STATS**: p=0.128 is not enough for a selectivity headline; future work should expand the variable set or use a stronger selectivity statistic.
  - **MANUSCRIPT-INTEGRITY**: explicitly disclose original picker failure, LOPO failure at V=M ell=2, post-hoc L=16 discovery, asymmetry/collateral steering caveats.
