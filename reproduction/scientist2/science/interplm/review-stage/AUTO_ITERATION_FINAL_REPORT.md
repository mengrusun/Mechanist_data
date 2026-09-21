# Auto Iteration Final Report — Reproduction of Five SAE-on-ESM-2 Interpretability Claims

- **Generated**: 2026-07-15T11:35:00 CST
- **Iterations consumed**: 2 / 6
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 6 / 10 (workshop framing; "Almost" for top-tier main-track)
- **Final canonical verdict**: **ready** (workshop / reproducibility track / negative-results venue)
- **Termination reason**: `positive_verdict` (three-dimensional STOP satisfied at iter 3)
- **Cumulative iteration cost**: runs_total = 2, gpu_hours_total = 3.0103 GPU-h
- **Aggregate pipeline cost**: ~5.14 of 10 GPU-h budget (experiment 2.14 + verify 0.07 + iteration 3.01)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

Two iterations addressed the two highest-leverage methodology concerns raised by the external LLM reviewer: (i) c5a's dead-code + under-fit probe was replaced with a well-converged, class-balanced log-loss probe, converting an integrity-compromised null into a **credible negative** (SAE 0.539 < neurons 0.548, p=0.516); (ii) c1's LLM-gate sample size was expanded 3× (100 → 300 features) with 2× more sequences at layer 9, **falsifying the "under-power" hypothesis**: the SAE gate pass rate is stable at 14.3% (±4% at 95%CI), so the absolute-count target of ~2548 sits credibly outside the 95% CI on SAE_interp = 1430 [1034, 1825], while the SAE≫neurons ratio remains at 12.9× (well above target ≥10×). The reviewer's iter-3 verdict flipped from "Almost" (iter 2) to "READY for workshop / reproducibility track" (iter 3, score 6/10) explicitly because c1 moved from ambiguous partial to well-characterized partial. Optional c3 characterization run (~1.5 GPU-h) was skipped: the STOP rule is satisfied and the reviewer explicitly said stop-and-write is the best-efficiency path. c4 (assay non-discriminative) and c5b (protocol too weak) remain unsupported — the reviewer confirmed neither is worth further compute under the remaining budget.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS (verify)                   | 1 | 1 PASS (held: c2) |
| FAIL (verify)                   | 0 | — |
| INCONCLUSIVE (verify)           | 0 | — |
| ZERO_ELIGIBLE_VARIANTS (verify) | 0 | — |
| INTEGRITY_ONLY (verify)         | 5 | 5 still INTEGRITY_ONLY (c1, c3, c4, c5a, c5b — Stage-2 swap-tests deferred by MAX_VERIFY_CLAIMS cap; iteration improved characterization of c1 + c5a but does not upgrade INTEGRITY_ONLY status itself, which requires re-running `/auto-verify <id> — resume: true` per the routing contract) |
| DEFERRED (legacy)               | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `c2` — SAE features cover substantially more Swiss-Prot concepts than raw neurons (scale-invariant under 650M→8M swap)
- **Original robustness signal**: robustness = 1.00; variants_passed = 1/1 (ESM-2-650M → ESM-2-8M swap; SAE covered 15→14, neurons 0→0, ratio ∞ preserved)
- **Reviewer consistency check** (iter 1, iter 2, iter 3): direction matches claim wording; absolute count target (~143) NOT hit at primary (SAE=15) but approached at τ=0.3 (SAE=65). Comparative gap replicated robustly; absolute count is not. **Paper-mandatory caveat**: state that the comparative headline reproduces but the absolute count does not, and cite likely-contributors (1500/10k SP-test residues used; 400 fine-grained concept universe more granular than reference).
- **Touched in iterations**: [1, 2, 3] (only as consistency check; no fix applied)
- **Final status**: **PASS (held)** — no iteration change
- **Notes for downstream (paper)**: This is the paper's strongest positive. Under-power caveat mandatory. Scale-invariance across 80× parameter reduction is a compelling secondary finding.

---

## Section 2 — FAIL Claims (full journey)

**No FAIL claims** — `verify_failed` bucket was empty from the start (no claim's Phase 9 swap variants both ran AND fell below robustness threshold).

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

**No INCONCLUSIVE claims** — `verify_inconclusive` bucket was empty from the start.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

**No ZERO_ELIGIBLE_VARIANTS claims** — `verify_zero_eligible_variants` bucket was empty from the start.

---

## Section 4b — INTEGRITY_ONLY Claims (Stage-2 swap-test skipped; main-experiment fixes carried in iteration)

Five claims (c1, c3, c4, c5a, c5b) reached iteration with state `INTEGRITY_ONLY` due to `stage2_skip_reason: max_verify_claims_cap`. Per the routing contract, iteration does NOT dispatch back-edge actions on INTEGRITY_ONLY claims — the upgrade is to re-run `/auto-verify <id> — resume: true` in a separate call. However, iteration **can** address the *content* of these claims' main experiments via type-② fixes (which is orthogonal to the swap-test status). Two claims (c5a, c1) received such fixes this round; the swap-test status of all five remains INTEGRITY_ONLY.

### 4b.1 `c5a` — SAE probes > neuron probes on Swiss-Prot annotation-filling (paired-Wilcoxon p<0.05)

**Original main-experiment result** (as-of-verify): mean_PR-AUC(SAE) = 0.5907, mean_PR-AUC(neurons) = 0.5906, paired-Wilcoxon p = 0.191. `verify_integrity_only` (max_verify_claims_cap); main-experiment integrity WARN (dead code in `per_concept_pr_auc` + SGDClassifier substitution + 25k/387k subsample).

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ② | Dead code (`per_concept_pr_auc` LogisticRegression defined but never called); active path was `SGDClassifier(max_iter=30, no class_weight)` — reported null did not test the intended probe | Wrote new function `per_concept_pr_auc_logreg()` — `SGDClassifier(loss='log_loss', max_iter=1500, tol=1e-4, class_weight='balanced')` — real convergence at ~43 iters via tol early-stop, with class-imbalance correction. Test subsample 25k → 50k. Reran m5 lightweight. | mean PR-AUC(SAE) = **0.5391**, mean PR-AUC(neurons) = **0.5476**, W=231, **p=0.516**; direction slight-neurons; **credible negative** at 1000 train seqs / 50k test residues / 30 scored concepts on L9 |

**Path taken (summary)**: main-experiment-script fix (type ②) — 1 iteration consumed; claim-reentry sub-budget 0 (given claim, not rewritable).

**Experiment & script modifications**

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `scripts/m5_annotation_filling.py:91-111` (dead code `per_concept_pr_auc`) | Defined `LogisticRegression(lbfgs, max_iter=200, StratifiedKFold)` — never called | Replaced with `per_concept_pr_auc_logreg()` (SGD-log-loss well-converged, class-balanced) — actively called in main loop |
| 1 | `scripts/m5_annotation_filling.py:236-238` (main probe call) | `SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30, tol=1e-4, random_state=seed)` — no class_weight, 30 iters | `per_concept_pr_auc_logreg(X_tr_sub, y_tr_sub, X_te, y_te, seed=seed, l2=1.0, max_iter=500)` — 1500 max_iter, class_weight='balanced', tol-early-stop |
| 1 | `scripts/m5_annotation_filling.py:178` (test subsample) | `TEST_SAMPLE = min(25000, N_test)` | `TEST_SAMPLE = min(50000, N_test)` |
| 1 | `refine-logs/EXPERIMENT_PLAN.md` M5 | (no iteration correction) | Added "Iteration-1 correction (type ②, c5a)" block documenting fix |

**Claim modifications**: none — c5a is `given` per task.md. The iteration verifies the claim as-stated; when the fair test yields not-supported, that is the correct disposition (reported as "credible negative under corrected fair test", not rewritten).

**Final experiment summary**
- New runs cited: `runs/iteration_round_1/m5_c5a_logreg_fix/{summary.md, pr_auc.parquet, wilcoxon.json, gpu_hours.txt}` (0.6616 GPU-h)
- Final mean PR-AUC(SAE) = 0.5391, mean PR-AUC(neurons) = 0.5476, p = 0.516
- Final status: **NOT SUPPORTED — credible negative** (was: NOT SUPPORTED — null-of-unknown-validity due to dead code)
- Reviewer confirmation (iter 2): "genuine methodological correction, not cosmetic"

**Reviewer memory thread (c5a)** — from `REVIEWER_MEMORY.md`:
- iter 1 New suspicion: "c5a dead-code: `per_concept_pr_auc()` defined but never called; active path uses `SGDClassifier(max_iter=30, no class_weight)`. Major integrity concern — reported null result is not testing the intended probe."
- iter 2 Previous-suspicion-addressed: **c5a dead-code / wrong active probe path: RESOLVED (genuine methodological correction).** Result: SAE 0.5391 < neurons 0.5476, p=0.516 — credible negative.
- iter 3 New pattern: "c5a is best described as **scope-limited negative**, not just 'under-powered null'. 20/50 dropped concepts create selection-on-evaluable-concepts caveat. Fine for scoped reporting."

### 4b.2 `c1` — SAE surfaces ~2548 interpretable features per layer, ≥10× the raw-neuron count

**Original main-experiment result** (as-of-verify): best layer L9: SAE_interp=1480 (target ~2548, 58%), ratio 19.3× (target ≥10×, MET). `verify_integrity_only` (max_verify_claims_cap); main-experiment integrity WARN (auto-interp proxy GT + scope under-power at 1500/10k seqs, 100/10240 LLM-gate).

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 2 | ② | Under-power on LLM-gate sample (100/10240 features, ±15% margin) and Swiss-Prot seqs (1500/10k). Reviewer hypothesis: bigger sample would tighten CI toward target 2548. | Focused L9-only rerun of `scripts/m1_feature_count.py` with n_seqs 1500→3000, n_auto_interp_features 100→300, n_auto_interp_neurons 100→300. | SAE gate pass rate stable: 15% → **14.33%**; SAE_interp = **1430** (was 1480); ratio SAE/neurons = **12.9×** (was 19.3×; neurons_interp went 77 → 111 due to 3× larger neuron gate sample); 95% CI SAE_interp = **[1034, 1825]** — target 2548 credibly OUTSIDE CI |

**Path taken (summary)**: main-experiment-script fix (type ②) — 1 iteration consumed; claim-reentry sub-budget 0 (given claim, not rewritable).

**Experiment & script modifications**

| Iter | Component | Before | After |
|---|---|---|---|
| 2 | `scripts/m1_feature_count.py` invocation (run.sh) | `--n-seqs 1500 --n-auto-interp-features 100 --n-auto-interp-neurons 100` (defaults from experiment stage) | `--n-seqs 3000 --n-auto-interp-features 300 --n-auto-interp-neurons 300` (via `runs/iteration_round_2/m1_c1_gate_expand_L9/run.sh`) |
| 2 | `runs/iteration_round_2/m1_c1_gate_expand_L9/run.sh` | (new file) | `export CUDA_VISIBLE_DEVICES=0,1,2,3; export DMX_API_KEY="$LLM_API_KEY"; python scripts/m1_feature_count.py --layer 9 ...` |

**Claim modifications**: none — c1 is `given` per task.md. After the under-power falsification, correct disposition is "well-characterized partial (direction met, absolute count credibly not met at ESM-2-650M L9)".

**Final experiment summary**
- New runs cited: `runs/iteration_round_2/m1_c1_gate_expand_L9/{layer9.json, gpu_hours_L9.txt, run.sh, logs/m1.log}` (2.3487 GPU-h)
- Final SAE_interp = 1430 (95% CI [1034, 1825]); ratio 12.9×
- Final status: **PARTIAL — direction MET (ratio 12.9× ≫ target 10×); absolute count NOT met (1430/2548 = 56%; target statistically outside 95% CI)** — the under-power hypothesis is falsified
- Reviewer verdict-shift trigger (iter 3): "c1 is now a well-characterized partial. That satisfies the real scientific criterion. Extra compute should not be spent trying to 'rescue' c1 by simple scaling."

**Reviewer memory thread (c1)** — from `REVIEWER_MEMORY.md`:
- iter 1 New suspicion: "c1 LLM-gate sample too small for layer-wide extrapolation: 100/10240 features/layer with ±15% margin. Increase to 300-500 minimum."
- iter 3 Previous-suspicion-addressed: **c1 under-power hypothesis: FALSIFIED.** Pass rate stable at ~14-15%; 95% CI on SAE_interp [1034, 1825] excludes target 2548.

### 4b.3 `c3` — SAE-vs-neuron gap is direct evidence of superposition (ordered ladder with Δ_SAE-PCA ≥ 20 at primary)

**Original main-experiment result** (as-of-verify): ladder holds direction (SAE=15, PCA=0, random-rot=0, neurons=0, shuffled-SAE=0 at primary); strict Δ_SAE-PCA=15 (target ≥20) near-miss; Δ=65 at τ=0.3. `verify_integrity_only` (max_verify_claims_cap); main-experiment integrity WARN.

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 3 | (none) | Reviewer marked as OPTIONAL characterization (~1.5 GPU-h). Given c1's under-power falsification, c3 is likely to behave similarly — direction robust, absolute-strict threshold near-miss. | Skipped: STOP rule satisfied in iter 3; per skill semantics, once STOP fires the loop terminates. c3 characterization would only add "polish". | Unchanged from baseline: direction MET, strict Δ=15 near-miss at primary, Δ=65 at τ=0.3 |

**Path taken (summary)**: no iteration fix applied — reviewer explicitly deemed optional; STOP rule fired before dispatch.

**Final experiment summary**
- No new runs (iteration action skipped).
- Final status: **PARTIAL — direction MET (ladder SAE ≫ PCA ≈ random-rot ≈ neurons ≈ shuffled-SAE at primary); strict Δ_SAE-PCA=15 near-miss (target ≥20); Δ=65 at τ=0.3 (WELL above)**.
- Paper framing: "partial directional support under current budget". If a c3 characterization run had been executed and behaved like c1, this would upgrade to "well-characterized partial". Reviewer explicitly said this delta is not required for submission.

**Reviewer memory thread (c3)**: no unique iter-flag; always paired with c1 as "under-powered near-miss".

### 4b.4 `c4` — ≥10% of Swiss-Prot-unaligned SAE features receive novel-concept-coherent LLM auto-interp labels

**Original main-experiment result** (as-of-verify): 0/100 novel on real arm, 0/50 novel on control arm — strict synonym-check-null gate has zero acceptance rate on BOTH arms. The metric cannot discriminate. `verify_integrity_only` (max_verify_claims_cap); main-experiment integrity WARN (criterion non-discriminative).

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| all | (none) | Criterion redesign (3-way adjudication: paraphrase / coherent-but-not-in-SP / incoherent) — reviewer said cheap in compute (~0.2-0.5 GPU-h) but "least strategic priority" and "review-optics risk" (redesigning after seeing zero/zero can look opportunistic). | Skipped in favor of higher-leverage c5a + c1 fixes. STOP rule fired in iter 3 before c4 became actionable. | Unchanged: NOT SUPPORTED (assay failure) |

**Final status**: **NOT SUPPORTED — assay failure (criterion non-discriminative)**. Paper framing: this is NOT evidence against the phenomenon; it is evidence that the test as instantiated cannot separate signal from control. LLM did emit coherent-sounding labels ("C2H2 zinc finger alpha-helix start", "Hydrophobic signal peptide core", "Gly-rich small-residue transmembrane helix motif") but every one was flagged as a paraphrase.

**Reviewer memory thread (c4)**: iter 1 flagged as "evaluation null, not negative result". Iter 2 reviewer explicitly categorized as "methodologically invalid current assay" and put lowest strategic priority.

### 4b.5 `c5b` — Clamping labeled SAE features steers ESM-2 generation with monotone dose-response

**Original main-experiment result** (as-of-verify): no_steer yield > every steered arm on every tested feature/dose. Doses α∈{0.5,1,2,4} span <1 OOM (target ≥3 OOM). Only 3/4 planned features tested. `verify_integrity_only` (max_verify_claims_cap); main-experiment integrity WARN (scope + mechanism-tuning gaps).

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| all | (none) | Widened α ladder {0,0.5,1,2,4,8,16,32} + restore batch 25 + 4th feature (~2-3 GPU-h). Reviewer marked as "moderate but uncertain" (rank 3 in iter 1, rank 4 in iter 2). Iter 3 explicit: "Do not spend on c5b — even after widening, upside is low." | Skipped: budget-conservation. Reviewer's iter-3 verdict does not require c5b fix. | Unchanged: NOT SUPPORTED (protocol too weak) |

**Final status**: **NOT SUPPORTED — protocol too weak to test claim fairly**. Not "steering demonstrably fails"; "current intervention protocol is not calibrated to reveal the claimed effect". Paper framing: "evidence limited by weak intervention sweep; cannot confidently reject nor confirm".

**Reviewer memory thread (c5b)**: iter 1 flagged as "under-scaled dose ladder + within-sample σ_f normalization suppresses intervention magnitude". Iter 2/3 both explicit: NOT worth ~3 GPU-h investment given remaining budget and low upside.

---

## Section 5 — Legacy DEFERRED Claims (empty under current architecture)

Empty. Legacy `deferred_claims` bucket is not populated by new verify runs; cap-cut claims now land in `verify_integrity_only` (Section 4b).

---

## Section 6 — Cross-Cutting Patterns

From `REVIEWER_MEMORY.md`'s per-iteration `- **Patterns**:` lines (deduped across iterations):

- **Under-power is a common thread across c1/c2/c3** — same 1500/10k SP-test subsample. Iter 2's c1 focused expansion falsified this hypothesis for c1 (pass rate stable, ratio slightly lower); by analogy, c3's absolute Δ_SAE-PCA at 15 is also likely a real limit rather than a compute artifact. **Resolved for c1; conjectured but not tested for c3.**
- **Two flavors of "not-supported"**: (a) methodology-broken (c5a dead code — fixable via surgical script change; c4 non-discriminative criterion — fixable via criterion redesign but "review optics risk"); (b) under-scaled protocol (c5b dose ladder). Iter 1 fixed (a) for c5a; (b) and c4's (a) were not fixed under budget constraints. **Partially resolved.**
- **Cleanest replicated claim is only the direction, not the absolute number** (c2 SAE≫neurons under 650M→8M swap). Paper must carry that caveat throughout.
- **Concept-granularity mismatch conjecture** — using 400 fine-grained concepts (each Pfam family separately, each PTM sub-class separately) may depress absolute counts vs reference. This affects both c2 (SAE=15/143) and c3 (Δ=15/20). **Unresolved but plausibly stable feature of this reproduction**; explicit paper caveat needed.
- **Iteration value ≠ iteration count** — iter 2's c1 rerun did NOT rescue the absolute-count target, but it DID move c1 from ambiguous "under-powered" to "well-characterized partial". Reviewer bumped score +1 (5→6) explicitly for this quality-of-outcome change, not for a claim rescue. **Resolved: this reproduction's value is in characterization, not in claim-count.**
- **Under-power vs criterion-design distinction is critical for review optics** — c4's zero-on-both-arms is fundamentally different from c5a's non-significant p — both are "not-supported" but for different reasons (assay failure vs credible negative). Paper must state this distinction clearly.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 2 / 6 (iter 3 was ⓪ narrative-only, does not consume budget)
- **Claim-reentries consumed**: 0 / 2 (all iterations were type ②, not ③)
- **Iteration `/run-experiment` calls**: runs_total = 2
- **Iteration GPU-hours**: gpu_hours_total = 3.0103 (of ~7.79 remaining post-experiment/verify; ~4.78 GPU-h left unused, deliberately preserved per reviewer recommendation)
- **Aggregate pipeline GPU-hours**: 5.14 of 10 (experiment 2.14 + verify 0.07 + iteration 3.01)

### Per-iteration breakdown (from `iteration_breakdown[]`)

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ② plan_script_rerun | c5a | — | 1 | 0.6616 | 4→(4-5) | almost |
| 2 | ② plan_script_rerun | c1 | — | 1 | 2.3487 | 5 | almost |
| 3 | ⓪ reviewer-narrative-only | (n/a) | — | 0 | 0.0 | **6** | **ready (workshop)** |

---

## Section 8 — Open Items for Human Reviewer

Items the loop could not close (or explicitly chose not to close under budget constraints — reviewer-approved skip):

- **Still-FAIL claims**: none (bucket was empty from start)
- **Still-INCONCLUSIVE claims**: none (bucket was empty from start)
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none (bucket was empty from start)
- **INTEGRITY_ONLY claims (Stage-2 swap-test skipped — not stress-tested via method/dataset/model variants)** — surfaced verbatim from `AUTO_REVIEW.md`'s per-iteration "Open Items — Unverified Under Swaps" subsections:
  - **c1** [`stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment`]: `/auto-verify c1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run). Post-iter-2 the L9 main experiment is at 3000 seqs / 300 gated features but other 5 layers remain at baseline 1500/100 — a future c1 swap test would need to decide whether to test the improved L9 or the original 6-layer suite.
  - **c3** [`stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment`]: `/auto-verify c3 — resume: true`.
  - **c4** [`stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment`]: `/auto-verify c4 — resume: true`. Note: the underlying criterion non-discriminativity would not be fixed by a swap-test; a criterion redesign in the main experiment is a prerequisite before c4 swap-test makes sense.
  - **c5a** [`stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment`]: `/auto-verify c5a — resume: true`. Post-iter-1 main experiment uses the fixed LogReg-substitute probe; swap-test would validate the credible-negative under model/data swaps.
  - **c5b** [`stage2_skip_reason: max_verify_claims_cap`; `main_experiment_integrity: warn`; `warn_source: experiment+mechanism`]: `/auto-verify c5b — resume: true`. Reviewer explicitly cautioned against rescue attempts — swap-test would be for completeness only.
- **Legacy deferred claims (empty in new runs)**: — none
- **Recurring unresolved patterns**: (a) concept-granularity mismatch conjecture (400 fine-grained concepts vs reference likely-coarser aggregation) — affects absolute-count targets of c2/c1/c3; (b) c5b intervention protocol — dose ladder <1 OOM is likely insufficient in principle; reviewer expressed low expectation even with widening. Neither requires action per reviewer.
- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): none (all claims are `given` per task.md; no claim rewrite was ever proposed).

### Compute-preserved rationale
- ~4.78 GPU-h intentionally left on the 10-h aggregate budget. Reviewer's explicit iter-3 recommendation was to STOP AND WRITE, with optional c3 characterization (~1.5 GPU-h) as the only worthwhile remaining spend. The STOP rule fired first (three-dimensional at iter 3); per skill semantics, the loop terminates and any remaining budget is preserved.
- The paper framing (Section 8 of `AUTO_REVIEW.md` iter 3): "Strongly reproduced: c2 / Directionally reproduced but quantitatively short of original: c1, c3 / Not reproduced due to invalid/non-discriminative evaluation setup: c4 / Not reproduced under corrected fair test: c5a / Not reproduced; evidence limited by weak intervention sweep: c5b" — is reviewer-approved as "crisp, honest, and reviewer-resistant".
