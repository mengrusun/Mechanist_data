# Auto Review — Iteration Log

Project: Reproduction of Five SAE-on-ESM-2 Interpretability Claims (`task.md`).
Reviewer: external LLM `gpt-5.4` via `https://www.dmxapi.cn/v1` (curl fallback — MCP not present).

---

## Iteration 1 (2026-07-15 10:25 → 10:49 CST)

### Assessment (Summary)
- Score: **4/10** (top venue, as-on-disk) / **6/10** (top-venue ceiling if remaining ~7.79 GPU-h spent well) / **6/10 → 7/10** (bio-interp workshop)
- Verdict: **almost** (canonical, mapped from "Almost, not fully ready.")
- Budget after this iteration: iterations **1/6**, claim-reentries **0/2**
- Key criticisms:
  * c1/c3 partial but under-powered on both sequence count (1500/10000 SP) and LLM-gate sample (100/10240).
  * c4 has non-discriminative criterion (0 acceptance on both real AND control) — this is an operationalization failure, not evidence.
  * c5a: **dead code integrity concern** — `per_concept_pr_auc()` (LogisticRegression) defined but never called; active path used `SGDClassifier(max_iter=30, no class_weight)`. Reported null did not test intended probe.
  * c5b steering ladder under-scaled (α∈{0.5,1,2,4} < 1 OOM; claim expects ≥3 OOM); within-sample σ_f normalization may suppress intervention magnitude further; only 3/4 planned features tested.
  * c2 the cleanest replicated claim — direction robust under 650M→8M swap — but ONLY the direction, not the absolute ~143 count. Paper must carry that caveat.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (14.4 KB)</summary>

Score: 4/10 top-venue as-on-disk, 6/10 top-venue post-fix ceiling, 6-7/10 bio-interp workshop. Verdict: Almost, not fully ready. Ranked ② fix list (leverage per remaining GPU-h):
1. c5a — fix dead-code path (call LogisticRegression) + score all feasible concepts; expected uplift: turns "null of unknown validity" into a real test (0.5–1 GPU-h).
2. c1 — increase LLM-gate sample from 100 to 300–500 per layer + increase Swiss-Prot seqs from 1500 to full 10k; tightens the ±15% margin (1–1.5 GPU-h).
3. c3 — full-sample rerun; controls are all zero already so SAE scaling should cross Δ≥20 threshold at primary (1–1.5 GPU-h).
4. c5b — widen dose ladder to {0,0.5,1,2,4,8,16,32} (~3 OOM), stop under-scaling by within-sample σ_f, restore batch size 25, test 4th feature (2–3 GPU-h).
5. c4 — replace strict synonym-check-null with a 3-way adjudication (paraphrase / coherent-but-not-in-SP / incoherent) — criterion redesign, not more compute (<0.5 GPU-h).

Suggested total spend 4.7–7.5 GPU-h. Cut c4 first if under budget, then c5b.

c2 consistency: directionally yes, absolutely no. Primary (SAE=15 vs neurons=0) supports the comparative headline; but "up to ~143 concepts" is 15 at primary / 65 at τ=0.3. Under-power + finer concept-granularity likely explain the shortfall. **Caveat mandatory for paper**: comparative gap replicated robustly; absolute count NOT.

READY? Almost. No FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS to block; but scientifically the story is "partial reproduction with one robust positive (c2), several provisional positives (c1, c3), two currently-blocked negatives (c4 criterion, c5a implementation, c5b protocol too weak)". Ready-ish for a workshop-style honest partial-reproduction paper.

Full text saved at `review-stage/_response_iter1.md`.

</details>

### Verify-Passed Claims (brief audit)
- **c2**: verify-PASS at robustness=1.0 (650M→8M swap: SAE=14 vs 15, neurons=0 in both, ratio ∞ preserved). Reviewer confirms directional match to claim wording. Caveat for paper: absolute ~143 target NOT hit (15 at primary; 65 at τ=0.3). "The comparative gap is replicated robustly; the absolute count is not."

### Actions Taken (per claim, per type)
- **c5a — type ② — fixed dead-code probe path + enlarged test subsample**
  - **Script Before/After** (`scripts/m5_annotation_filling.py`):
    - Before (probe): `SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=30, tol=1e-4, random_state=seed, n_jobs=1)` — no class_weight, only 30 iters, direct `.decision_function(X_te)`. Dead code: `per_concept_pr_auc()` (LogisticRegression lbfgs, max_iter=200, StratifiedKFold) defined but never called.
    - After (probe): new function `per_concept_pr_auc_logreg()` — `SGDClassifier(loss='log_loss', alpha=1e-4, max_iter=1500, tol=1e-4, class_weight='balanced', random_state=seed, n_jobs=1)`. Real convergence at ~43 iters (early-stop via tol), with class-imbalance correction. LogisticRegression(lbfgs/liblinear) was tried first but stalled at ≥ minutes per fit; SGD with same log-loss objective converges in ~8-17s per fit on 15k×10240 shape.
    - Before (test subsample): `TEST_SAMPLE = min(25000, N_test)` → 25k residues.
    - After (test subsample): `TEST_SAMPLE = min(50000, N_test)` → 50k residues (2× larger; still tractable).
  - **Plan Before/After** (`refine-logs/EXPERIMENT_PLAN.md` M5 section): appended "Iteration-1 correction (type ②, c5a)" block documenting the fix.
  - **Re-invoked**: `runs/iteration_round_1/m5_c5a_logreg_fix/run.sh` (single lightweight re-run, `CUDA_VISIBLE_DEVICES=0,1,2,3`). Not a full `/auto-experiment` re-dispatch — only the c5a-scoped m5 script was rerun.
  - **New result**: mean_PR-AUC(SAE) = 0.5391, mean_PR-AUC(neurons) = 0.5476, paired-Wilcoxon(SAE > neurons) W=231, **p=0.516**. Only 30/50 concepts still scored (the 50k subsample did not add coverage — the 20 dropped concepts have zero test positives anywhere in the 387k residues, not just in the subsample). Notable direction flip: mean neuron > mean SAE (previously nearly identical).
  - **Interpretation**: The dead-code + under-fit SGD fix confirms c5a is **genuinely not supported** at this scale — not an SGD-under-fitting artifact. Direction is neuron-slight-edge, p=0.52 well above any threshold. The absence of a SAE > neurons signal here is now a reliable negative rather than a null-of-unknown-validity. **The claim as stated (SAE probe > neuron probe with p<0.05) is not defensible at 1000 train seqs / 50k test residues / 30 scored concepts on ESM-2-650M layer 9.**

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none — c5a is `given` per task.md; per the routing contract, this loop verifies given claims and does not rewrite them. Its correct disposition after a fair-test null is "confirmed not supported at this scale".

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **c1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run). Main-experiment integrity: WARN; warn_source: experiment (auto-interp proxy GT + scope under-power).
- **c3** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c3 — resume: true`. Main-experiment integrity: WARN; warn_source: experiment (strict margin not met; direction supported).
- **c4** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c4 — resume: true`. Main-experiment integrity: WARN; warn_source: experiment (synonym-check GT proxy + scope + criterion non-discriminative).
- **c5a** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c5a — resume: true`. Main-experiment integrity: WARN; warn_source: experiment (post-iter-1: dead-code fixed, but the underlying null is now the informative result — reviewer should note WARN persists on scope: 30/50 concepts scored, 1000 train seqs).
- **c5b** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c5b — resume: true`. Main-experiment integrity: WARN; warn_source: experiment+mechanism (scope 3/4 features, 10/25 seqs; sweep <3 OOM / <5 grid points).

### Results
- `[run-experiment] iteration=1 runs_this_iteration=1 gpu_hours_this_iteration=0.6616 cumulative_gpu_hours=0.6616`
- c5a → lightweight m5 re-run → **still not-supported** (SAE 0.5391 < neurons 0.5476, p=0.516). Now a *fair* null, not a dead-code artifact.

### Status
continuing to iteration 2

---

## Iteration 2 (2026-07-15 10:50 → 11:33 CST)

### Assessment (Summary)
- Score: **5/10** (top venue as-on-disk) / **6/10** (top-venue ceiling) / **7/10** (workshop as-is) / **8/10** (workshop ceiling)
- Verdict: **almost** (canonical)
- Budget after this iteration: iterations **2/6**, claim-reentries **0/2**
- Key criticisms (from reviewer iter 2):
  * c5a resolution: **genuinely fixed** in iter 1; now a well-tested "credible negative" (not integrity-compromised null). Scope caveat remains (30/50 concepts scored, 1000 train seqs, L9 only).
  * c1 under-power still open (was scheduled for this iter — result below).
  * c3 under-power, c4 metric-broken, c5b protocol too weak still open.
  * Bifurcation of remaining not-supported: (1) c5a = well-tested negative; (2) c1/c3 = under-powered near-misses; (3) c4/c5b = methodology-broken.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (9.8 KB)</summary>

Score 5/10 top-venue as-is → 6/10 ceiling; 7/10 workshop as-is → 8/10 ceiling. Bumped from iter 1 by +1 because c5a moved from "methodologically compromised null" to "credible negative result" — that is real progress.

**c5a resolution**: genuine methodological correction, not cosmetic. Active probe path now matches intended objective; optimization no longer obviously under-fit; class imbalance addressed; test size increased; result flipped from ambiguous null to slight-neurons-advantage with p=0.516. At the tested setting, claim is not supported; direction is slightly against SAE. This is not universal falsification — scope caveats remain (1 layer, 1000 train seqs, 30/50 concepts). Distinction: not "obviously under-powered" anymore; remaining limitations are scope, not methodology invalidators. Do NOT prioritize more c5a.

**Remaining fixes (updated ranking, ~7.13 GPU-h available)**:
1. c1: expand LLM-gate 100→300-500 + 10k seqs; keep criterion unchanged; ~1.0-1.5 GPU-h; **high** chance PARTIAL→SUPPORTED.
2. c3: full-sample rerun at 10k; ~1.0-1.5 GPU-h; **moderate-to-high** chance strict Δ≥20 crosses at primary.
3. c5b: expand α ladder over ≥3 OOM, restore batch 25, 4 features; ~2.0-3.0 GPU-h; less optimistic.
4. c4: redesign criterion (3-way adjudication); ~0.2-0.5 GPU-h; lowest strategic priority, review-optics risk.

If 2 fixes: c1+c3. If 3: c1+c3+c5b. Do NOT spend more on c5a.

**READY**: Almost. Not top-venue-ready as five-claim reproduction; workshop-ready if honestly framed as "partial reproduction with important negative correction (c5a) + c2 robust positive + c1/c3 near-misses". Best remaining investment: powering up c1 and c3. If they don't move → stop, frame paper as workshop-style well-scoped partial reproduction.

Full text at `review-stage/_response_iter2.md`.

</details>

### Verify-Passed Claims (brief audit)
- **c2**: unchanged — still verify-PASS at robustness=1.0. Direction match confirmed; absolute-count caveat still mandatory. No new consistency issues raised in iter 2.

### Actions Taken (per claim, per type)
- **c1 — type ② — LLM-gate sample expansion + more sequences (L9 only, focused)**
  - **Script arg Before/After** (`scripts/m1_feature_count.py` invoked via `runs/iteration_round_2/m1_c1_gate_expand_L9/run.sh`):
    - Before (iter 0 baseline): `--layer 9 --n-seqs 1500 --n-auto-interp-features 100 --n-auto-interp-neurons 100`
    - After (iter 2): `--layer 9 --n-seqs 3000 --n-auto-interp-features 300 --n-auto-interp-neurons 300`
  - **Additional fix**: `runs/iteration_round_2/m1_c1_gate_expand_L9/run.sh` sets `export DMX_API_KEY="$LLM_API_KEY"` (script requires this env var; original iter 0 runs had it set by the caller).
  - **Re-invoked**: `runs/iteration_round_2/m1_c1_gate_expand_L9/run.sh`, `CUDA_VISIBLE_DEVICES=0,1,2,3` on 4× A800.
  - **New result at L9** (vs iter 0 for comparison):
    | Metric | iter 0 (n_seqs=1500, gate=100) | iter 2 (n_seqs=3000, gate=300) |
    |---|---|---|
    | Total residues | 387,195 | **765,123** (2×) |
    | SAE gate tested / passed / rate | 100 / 15 / 15.0% | **300 / 43 / 14.33%** |
    | Neuron gate tested / passed / rate | 100 / 6 / 6.0% | **300 / 26 / 8.67%** |
    | SAE estimated interpretable | 1480 | **1430** |
    | Neuron estimated interpretable | 77 | **111** |
    | Ratio SAE / Neurons | 19.3× | **12.9×** |
    | 95% CI on SAE gate pass rate (±half-width) | ±7% | **±4%** (tighter) |
    | 95% CI on SAE_interp | ~[1258–1702] | **[1034, 1825]** |
    | To hit target SAE_interp=2548 | pass rate needed = 25.5% (vs observed 15%) | pass rate needed = 25.5% (vs observed 14.3%) |
  - **Interpretation**: c1 under-power hypothesis (that a larger LLM-gate sample would push toward target) is **NOT confirmed**. With 3× more features gated + 2× more sequences, the SAE gate pass rate is *stable* at ~14-15%, and the 95% CI on SAE_interp is now [1034, 1825] — target 2548 sits well OUTSIDE the CI. The absolute-count claim is **statistically confirmed as not met at ESM-2-650M layer 9**. Direction (SAE≫neurons, ratio 12.9×) remains MET at target ≥10×.
  - **Cost**: 2.3487 GPU-h (LLM gate step took most of the wall-clock — API calls, not compute).

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none — claim c1 is `given` per task.md; iteration verifies, does not rewrite. The claim's absolute-count target is now statistically defensible as not met at 650M L9 — that is the correct disposition (report a well-characterized negative on the absolute count while directionally supporting the SAE≫neurons magnitude).

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **c1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c1 — resume: true`. Main-experiment integrity: WARN → **still WARN** post-iter-2 (scope now improved to 3000 seqs / 300 gated features, but scope: L9-only, other 5 layers still at 1500/100 baseline).
- **c3** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c3 — resume: true`. Main-experiment integrity: WARN (unchanged).
- **c4** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c4 — resume: true`. Main-experiment integrity: WARN (unchanged).
- **c5a** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c5a — resume: true`. Main-experiment integrity: WARN (iter-1 dead-code fix landed but scope still limited).
- **c5b** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c5b — resume: true`. Main-experiment integrity: WARN (unchanged).

### Results
- `[run-experiment] iteration=2 runs_this_iteration=1 gpu_hours_this_iteration=2.3487 cumulative_gpu_hours=3.0103`
- c1 → lightweight m1 rerun at L9 with expanded gate → **still partial**; direction MET (ratio 12.9×), absolute count NOT met (1430/2548 = 56%; 95% CI [1034, 1825] excludes target).
- The under-power hypothesis for c1 (that more compute would push toward target) is **falsified** — the SAE gate pass rate is remarkably stable, the true SAE interpretable count is ~1400-1500, well below the ~2548 target. Cause is likely NOT compute; more likely (a) differences in what counts as "interpretable" between our LLM-gate protocol and the reference's, or (b) SAE checkpoint properties differ (fewer features actually reach the "interpretable" bar at this operating point).

### Status
continuing to iteration 3

---

## Iteration 3 (2026-07-15 11:33 → 11:35 CST) — reviewer-only cycle (⓪ narrative)

### Assessment (Summary)
- Score: **6/10** (up +1 from iter 2's 5/10). Reviewer explicit: "READY, for a workshop / reproducibility track / negative results venue. If you insist on top-tier main-track standard: Almost, not ready."
- Verdict: **ready** (canonical, mapped from "READY, for workshop"). Note: reviewer also gives "Almost" for top-tier main-track — using workshop framing per the reviewer's own decision tree.
- Budget after this iteration: iterations **2/6** consumed (iter 3 is ⓪-narrative, does NOT consume budget); claim-reentries **0/2**
- Key criticisms: **none critical.** All prior suspicions resolved or explicitly categorized (see below).

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (8.2 KB)</summary>

Score bumped to 6/10 because c1 was converted from "maybe under-powered" to "well-characterized partial reproduction". Recommendation: **STOP AND WRITE.** Optional: spend ~1.5 GPU-h on c3 only for characterization polish. Do NOT spend on c5b (upside too low even with widened protocol). Do NOT touch c4 (assay design failure, not compute).

Paper framing:
- Strongly reproduced: c2
- Directionally reproduced but quantitatively short of original: c1, c3
- Not reproduced due to invalid/non-discriminative evaluation setup: c4
- Not reproduced under corrected fair test: c5a
- Not reproduced; evidence limited by weak intervention sweep: c5b

Ready for workshop / reproducibility track / negative results venue. Almost for top-tier main-track.

Reviewer's stopping heuristic ("if c1/c3 don't move → stop") — c1 has now "moved enough" (to clarity, not to success). That satisfies the real scientific criterion. c3 is optional characterization.

Full text at `review-stage/_response_iter3.md`.

</details>

### Verify-Passed Claims (brief audit)
- **c2**: no new consistency issues. Absolute-count caveat still mandatory for paper.

### Actions Taken (per claim, per type)
- **type ⓪ narrative-only** — none of the reviewer's recommendations require new script edits or new runs. The reviewer explicitly says: no critical fix is required for submission. This iteration is a reviewer-consultation with no back-edge action.
- Optional c3 characterization run was CONSIDERED but SKIPPED — the STOP rule is already satisfied (see Status below); running c3 would only add polish, and per the loop's semantics, once the STOP rule fires, the loop terminates.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **c1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c1 — resume: true`. Main-experiment integrity: WARN (improved scope at L9: 3000 seqs, 300 gated features; iter-2 additional characterization).
- **c3** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c3 — resume: true`. Main-experiment integrity: WARN.
- **c4** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c4 — resume: true`. Main-experiment integrity: WARN.
- **c5a** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c5a — resume: true`. Main-experiment integrity: WARN.
- **c5b** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify c5b — resume: true`. Main-experiment integrity: WARN.

### Results
- `[run-experiment] iteration=3 runs_this_iteration=0 gpu_hours_this_iteration=0.0 cumulative_gpu_hours=3.0103`
- No new experiment runs. Cumulative iteration GPU-h: 3.0103 of 7.79 remaining (aggregate pipeline: 5.14 of 10-h).

### Status
completed — termination_reason=positive_verdict (three-dimensional STOP satisfied: score 6 ≥ TARGET_SCORE 6, verdict ∈ {ready, almost}, all FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS buckets empty)
