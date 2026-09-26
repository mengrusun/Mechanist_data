# Auto Iteration Final Report — Subliminal Learning on Qwen-Image (Denoising SFT)

- **Generated**: 2026-07-16T09:45:00
- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 6.5 / 10
- **Final canonical verdict**: almost
- **Termination reason**: positive_verdict (STOP rule three-dimensional check satisfied on iteration 1)
- **Cumulative iteration cost**: runs_total=0, gpu_hours_total=0.0 (no back-edge action fired — reviewer's minimum fix was type-⓪ narrative-only)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

Iteration 1 reached the three-dimensional STOP rule (score ≥ 6, canonical verdict ∈ {ready, almost}, no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS claims). The reviewer explicitly stated that "no new expensive runs are methodologically required for submission" — the minimum concrete fix is **narrative-only** (paper framing tightening for C1 scope caveats + C2 refutation-scoping caveats). The loop therefore terminates on the first iteration with **zero back-edge actions consumed**, leaving both the iteration budget (0/6) and the claim-reentry sub-budget (0/2) intact and the GPU budget untouched. C1's positive behavioral finding is verified and numerically consistent; C2's negative mechanism finding for the LoRA-SVD-additive-steering family is honestly reported and audit-clean, with the optional swap-test upgrade (`/auto-verify C2 — resume: true`) recorded in Open Items for potential follow-up.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 | 1 PASS (C1 held; narrative caveats logged) |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY           | 1 | 1 INTEGRITY_ONLY (C2 held; upgrade command recorded in Open Items) |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C1` — Subliminal transfer via denoising SFT (M0 gate)

- **Statement**: Subliminal transfer via denoising SFT on banana-filtered teacher-generated images: mean_gap ≥ 0.10, per-seed majority ≥ 4/7, cleaned-channel banana residue = 0.
- **Original robustness signal**: robustness=1.0, variants_passed=1/1 (rank-8 model-swap; mean_gap=0.150 at 89% of main-experiment magnitude; 3/3 seeds pass; integrity PASS)
- **Reviewer consistency check** (iteration 1): PASS. Claim wording matches on-disk numbers verbatim:
  - mean_gap = +0.169 (threshold ≥ 0.10) ✓
  - 95% bootstrap CI [0.110, 0.246] — lower bound above threshold ✓
  - per-seed majority = 6/7 (86%) — above ≥ 4/7 threshold ✓
  - Wilcoxon p_onesided = 0.008 (< 0.05) ✓
  - residues = 0 both arms ✓
  - Verify rank-8 variant: mean_gap = +0.150, CI [0.113, 0.175], 3/3 seeds positive, integrity PASS
- **Touched in iterations**: [1] — narrative-only (⓪), no scripts or data changes.
- **Final status**: **PASS (held)** — verify-robust to LoRA-rank 8 model-swap.

- **Notes for downstream (paper-writing) — MANDATORY narrative caveats from reviewer**:

  1. **Scope disclosure (abstract + conclusion)** — recommended sentence:
     > "The effect is established in a narrow proof-of-concept setting: a single concept (banana), a single model family (Qwen-Image), a single teacher/student adaptation recipe (LoRA denoising SFT), and a small matched post-filter dataset (N=53/arm), with robustness checked only along a model-axis variant (LoRA rank 16→8)."

  2. **Judge-dependence caveat (limitations section)** — recommended text:
     > "Both the residue-filtering step and the final P(banana) evaluation use gpt-4o. Independent-judge replication is left to follow-up work."

  3. **Under-N caveat (methods + limitations)** — must be foregrounded, not buried. The matched N=53/arm is induced by an extreme teacher LoRA (raw teacher P(banana)=0.903 on 600 generations → only 58 non-banana teacher survivors), which is a `task.md`-flagged risk realized.

  4. **Conditional-slice framing (methods)** — the surviving teacher-arm training images are the ~10% of the teacher's neutral-prompt generations that gpt-4o did NOT judge as banana, not a "clean neutral fruit set." Qualifies what "subliminal signal" means in this setup.

---

## Section 2 — FAIL Claims (full journey)

None. `verify_failed` bucket was empty.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

None. `verify_inconclusive` bucket was empty.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

None. `verify_zero_eligible_variants` bucket was empty.

---

## Section 4b — INTEGRITY_ONLY Claims (no-action bucket; recorded for potential follow-up)

### 4b.1 `C2` — Some internal DiT component causally carries the banana signal (mechanism, refuted for one family)

- **Statement**: Some internal component of the Qwen-Image DiT — layer set, attention heads, residual-stream direction, or low-rank LoRA-update component — carries the banana signal and is causally intervenable with sign, dose-response, and specificity. Depends on C1.
- **Original state**: INTEGRITY_ONLY. `stage2_skip_reason: max_verify_claims_cap` — Stage 1 Phase 2 audit PASSED for both `/experiment-audit` and `/mechanism-audit`; Stage 2 (swap-test) was intentionally skipped because `MAX_VERIFY_CLAIMS=1` picked C1 (the load-bearing positive) as the top-1 admitted claim.
- **Main-experiment verdict (unchanged by this loop)**: REFUTED for the LoRA-SVD-derived additive-steering family. Location (M1.1): top-3 blocks [2, 0, 8]; overlap-gap_u positive at these blocks (correlational evidence). Causal (M1.2): flat α-sweep in both single-block (spearman=-0.60) and 9-block-window (spearman=-0.26) configs; random-direction control matches — primary is not distinguishable from noise.
- **Reviewer judgment (iteration 1)**: NO back-edge action proposed (respected the INTEGRITY_ONLY no-action contract). Refutation is genuine and honestly reported (negative α included, random-direction control run, wider window tested); this is good science, not a bug.
- **Touched in iterations**: [1] — narrative-only (⓪) scope caveat; no back-edge action.
- **Final status**: **INTEGRITY_ONLY (held)** — swap-test upgrade recorded in Open Items.

- **Mandatory narrative caveat for paper text**:
  > "The mechanism section reports a **negative result scoped to the LoRA-SVD-derived additive-residual-stream steering family**, in two configurations (single top-block and 9-early-block window). This does NOT constitute evidence that no internal DiT component causally carries the banana signal — it refutes ONE mechanism family. Alternative mechanism families (activation patching, MLP-path steering, full-LoRA weight-space swap, per-block per-module intervention) remain open and are called out as follow-up work."

- **Upgrade instruction (Open Items)**:
  ```
  /auto-verify C2 — resume: true
  # single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 (swap variants + judge) run
  ```

---

## Section 5 — Legacy DEFERRED Claims

Empty under the current architecture (no legacy `## Deferred Claims` section in `verify/VERIFY_REPORT.md`).

---

## Section 6 — Cross-Cutting Patterns

Patterns flagged by the reviewer in iteration 1 that the paper / any follow-up round should watch for:

- **Same-judge coupling** (gpt-4o for both filter and evaluation) — a systemic evaluation-methodology concern that touches BOTH C1 (P(banana) evaluation) and C2 (steering-sweep P(banana) evaluation). Not resolved this round; recommendation is independent-judge replication in follow-up.
- **Effect-size fragility under post-filter N growth** — if a weaker teacher or broader prompt pool were used, post-filter N would grow and the surviving teacher-arm training slice would become less unusual. It is unknown whether the subliminal transfer effect persists at higher N (or, conversely, whether the effect is specifically an artifact of the extreme conditional slice that survives an aggressive filter).
- **Generality across concepts / model families / recipes** — banana / Qwen-Image / LoRA-denoising-SFT is a narrow proof-of-concept. All three axes remain untested.
- **Mechanism family diversity** — one family refuted (LoRA-SVD-additive-steering); several alternatives (activation patching, MLP-path, full-LoRA swap) untried. Pattern: mechanism refutations should be phrased as family-scoped, never universal.

Only touched C2 (mechanism scoping) and C1 (all four bullets) via narrative caveats in iteration 1. Fully resolving these patterns requires follow-up rounds (`/next-round`) and independent-judge replication, not iteration back-edges.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.0 (no back-edge — narrative-only fixes carried to paper-writing stage)

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative-only (no back-edge) | C1, C2 (both for narrative caveats) | — | 0 | 0.0 | 6.5 | almost |

---

## Section 8 — Open Items for Human Reviewer / Paper-Writing Stage

- **Still-FAIL claims** (after exhausting routing options): none.
- **Still-INCONCLUSIVE claims**: none.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - **C2** — `stage2_skip_reason: max_verify_claims_cap`; main_experiment_integrity: pass; warn_source: null. Upgrade instruction: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim). Not blocking for READY.

- **Legacy deferred claims (empty in new runs)**: none.

- **Recurring unresolved patterns** (see Section 6):
  1. Same-judge coupling (gpt-4o filter + eval).
  2. Effect-size fragility under post-filter N growth (untested).
  3. Generality across concepts / model families / recipes (untested).
  4. Mechanism family diversity (only LoRA-SVD-additive refuted).

- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): none.

- **Paper-writing to-do list** (from reviewer's minimum concrete fix — narrative-only, type-⓪):
  1. Make scope limitations explicit in abstract / conclusion (single-concept, single-model-family, N=53/arm, model-axis-only robustness).
  2. State that C2 is a refutation of ONE mechanism family, not evidence of no mechanism.
  3. Disclose judge-dependence and N=53 post-filter bottleneck prominently.
  4. Add "conditional-slice" framing to methods (surviving 10% of teacher generations, not a clean neutral set).
