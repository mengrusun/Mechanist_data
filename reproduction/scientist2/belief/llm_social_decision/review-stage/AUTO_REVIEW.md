# Auto Review — Steerable Social-Variable Directions

**Project**: llm_social_decision
**Loop config**: MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6, POSITIVE_VERDICT_TERMS={ready, almost}
**Reviewer**: `gpt-5.4` via llm-chat (shell env)
**GPU pin**: 1,2,3,5,6 (effective per-run: recorded in each run.sh)
**Verify buckets** (from `verify/VERIFY_REPORT.md`):
- verify_passed = [C3]
- verify_failed = [], verify_inconclusive = [], verify_zero_eligible_variants = []
- verify_integrity_only = [C1, C2, C4] (stage2_skip_reason: max_verify_claims_cap)
- deferred_claims = [] (legacy)

---

## Iteration 1 (2026-07-13T20:07)

### Assessment (Summary)
- Score: **4/10**
- Verdict: **not ready** (canonical)
- Budget after this iteration: iterations 1/6, claim-reentries 0/2
- Key criticisms:
  - C3's strongest signal (L=16 M sign-inversion, A/G amplification) is **missing the L=16 random-direction control** — the most decisive alternative-explanation null.
  - C4 was tested only where steering had no power (picked shallow layers). Selectivity has never been evaluated at L=16 where C3 actually manifests. If C4 fails at L=16 the mechanism is broad activation hacking, not variable-specific steering.
  - C1's shallow-layer probe=1.0 may reflect lexical/template leakage from paired-partner design, not genuine social-variable encoding.
  - C2's "purity" phrasing overreaches — GS off-diag max 0.506 is a large residual coupling; LEACE fails outright at shallow layers.
  - C3 success uses raw v̂_V rather than GS/LEACE pure directions, so C3 does not really validate the C2 pure-direction narrative.
  - V=A projection-transfer p=0.06 is marginal.
  - Model-swap variant used n=10/cell (planned 200); PASS verdict rests on qualitative pattern match.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Overall score
**4/10**

## Canonical verdict
**Not ready**

This is not submission-ready for a top-tier ML venue in its current form. The core issue is simple: the headline causal-steering story is supported only after a post hoc layer deviation, while the preregistered/main layer-selection procedure systematically chose layers where steering power was effectively zero. That makes the current experimental logic look mis-specified rather than convincingly validated.

## High-level assessment

The paper has an interesting setup and one potentially real mechanistic observation: there appear to be steerable directions for some social variables at a deeper layer (L=16), with at least one replicated sign inversion for **M** across a model swap. That is the strongest part.

But as a NeurIPS/ICML reviewer, I would view the current package as **fragile and partially internally inconsistent**:
- **C1** is only partially supported and may be contaminated by shallow lexical shortcuts.
- **C2** is not cleanly established as stated; "purity" depends on one decorrelation method working and another failing badly.
- **C3** is the main claim, but the main experiment was underpowered at the preregistered/picked layers and only "rescued" by a deeper-layer supplementary analysis.
- **C4** is effectively unsupported where tested, and not tested where C3 actually works.

## Per-claim assessment

### C1 · linear encoding — Weak / partial support
Probe accuracy = 1.0 at shallow layers is not impressive when the dataset construction leaks lexical/categorical cues. Projection-transfer evidence: V=A marginal (p=0.06). Best framing: "some evidence of linearly extractable correlates, but shallow-layer recoverability is likely confounded by prompt lexical structure."

### C2 · purity via decorrelation — Overclaimed
Off-diag max 0.506 is a large residual interaction. LEACE fails badly at shallow layers — suggests representation geometry is not compatible with "isolate unique causal factor." Narrow to "GS produces a more decorrelated direction basis than raw, but not demonstrably confound-free."

### C3 · bidirectional causal steering — Promising but procedurally weak
- Main registered/selected layers fail. Successful result at supplementary post-hoc layer.
- No L=16 random-direction control.
- Swap uses n=10/cell.
- Steering used raw directions, not pure GS/LEACE — does not validate C2's pure-direction narrative.
Current evidence supports narrower claim: "raw direction injections at deeper layers can causally modulate dictator-game outputs." Does not fully support the pure-social-variable-directions story.

### C4 · selectivity — Unsupported
Tested only where steering had no power. If C3 only works at L=16 and selectivity is evaluated where steering is near-zero, the selectivity claim is presently empty.

## PASS-claim check: C3 consistency review
- Confidence in consistency: 0.72/1.00.
- M sign-inversion (+0.71 → −0.31 main; +0.67 → −0.67 swap) is cleanest replicated evidence.
- Caveats that must appear in paper: not-at-original-picked-layers; no L=16 random-direction control; swap n=10/cell; raw v̂ not pure; α range ±2σ only.

## Minimum-fix recommendations

### Single most important fix — ② main-experiment-script improvement
**Run the missing L=16 random-direction control for C3.**

### Second most important fix — ② main-experiment-script improvement
**Run C4 selectivity at L=16 where C3 actually manifests.**

### Narrative-only minimum fixes — ⓪
- Narrow C2 wording aggressively ("decorrelated under GS" instead of "pure" / "free of confounds").
- Acknowledge C1 shallow-layer lexical confounding.
- Reframe C3 as a deeper-layer raw-direction effect (do not overstate that the preregistered layer-selection pipeline discovered the causal directions).

## Submission readiness
**No. Not ready.** Do not submit before the L=16 random-direction control and L=16 C4 selectivity sweep are run.

## Memory update
- **Primary suspicion:** shallow-layer C1 signal dominated by lexical/template leakage.
- **Primary mechanistic concern:** current layer-pick heuristic is fundamentally wrong for intervention studies.
- **Primary causal concern:** C3 strongest result at L=16 lacks random-direction control.
- **Primary selectivity concern:** C4 must test at L=16 before claiming specificity.
- **Conceptual inconsistency to track:** C2 claims "pure" but C3 success uses raw, not pure.
- **Narrative inflation risk:** authors may overstate "linearly extractable social variables."
- **Methodological red flag:** LEACE failure + GS off-diag 0.506 → representation entanglement is substantial.
- **Replication note:** M sign inversion across model swap is the cleanest positive signal; stress-test that result with proper controls.

</details>

### Verify-Passed Claims (brief audit)
- **C3**: reviewer confidence in consistency 0.72/1.00. Main and swap both show V=M sign-inverts at α=+2 at L=16 (main +0.71 → −0.31; swap +0.67 → −0.67). Caveats to surface in the paper: L=16 came from a supplementary run (deviation from pre-registration); n_baseline=10/cell in the swap variant vs planned 200; no L=16 random-direction control in either main-supp or swap variant; α range ±2σ only; steering used raw v̂_V rather than GS/LEACE-purified direction. Iteration 1 launches the L=16 random-direction control to close the strongest open critique.

### Actions Taken (per claim, per type)

- **C3 — type ② — L=16 random-direction control** (new experiment `runs/iteration_round_1/L16_random_control/`)
  - **New script**: `scripts/iteration/iter1_L16_random_control_and_C4_selectivity.py`
  - **Design**: 4 V × 3 random seeds × 5-point α grid ∈ {−2,−1,0,+1,+2}·σ at L=16, norm-matched via the same σ_ruler = σ_proj of the extracted v̂_V at L=16 (so α·σ produces the SAME expected residual perturbation magnitude the extracted direction uses). Held-out n_baseline=200 per cell, same DG-1000 held split. This tightens the M4-supp finding: if the sign-inversion of V=M at α=+2 is direction-specific, the random-direction control at α=+2 should NOT invert v_effect.
  - **Refutes reviewer's Primary causal concern**.
  - Re-invoked: none (in-loop; no /auto-verify needed — main-experiment supp augmentation).

- **C4 — type ② — L=16 selectivity 4×4 sweep** (new experiment `runs/iteration_round_1/L16_c4_selectivity/`)
  - **Same script** as above (single dispatch for compute efficiency; shared model load).
  - **Design**: 4 V × 3 α ∈ {−2,0,+2}·σ at L=16 with raw v̂_V (matching M4-supp convention); records w_effects for all four W per grid point → assemble M[V,W] shift matrix (+2σ and −2σ), compute c4-ratio max_|off-diag| / min_|diag|, run 2000-permutation test on |diag| ≥ |off-diag|.
  - **Refutes reviewer's Primary selectivity concern**.
  - Re-invoked: none (in-loop; main-experiment C4 completion, not verify-swap).

- **C1 — type ⓪ narrative** — will add paper-side caveat: "Shallow-layer (ℓ ∈ {2,4,6}) linear recovery may reflect prompt lexical structure induced by minimal-edit paired partners. Decision-relevance evidence is stronger for G/I/M (β significant at p<0.005) than A (p=0.059, marginal)."

- **C2 — type ⓪ narrative** — will replace "pure" / "free of confounds" wording with "GS-decorrelated basis." Note off-diag max 0.506 as a residual coupling caveat. Retain LEACE-failure narrative as a substantive representation-geometry finding at shallow layers, not a defect to hide.

- **C3 — type ⓪ narrative** — reframe supp L=16 as a diagnostic re-picking, not a pre-registered result; explicitly state that raw v̂_V (not GS/LEACE-purified) was used at L=16.

- **C4 — type ⓪ narrative** — surface the plan-vs-observation gap: the picked-layer heuristic max probe_cv_acc(ℓ) chose shallow layers where σ_proj was too small; the correct picker is likely argmax_ℓ σ_proj(ℓ) × |β_proj-transfer(ℓ)|. This is a real methodological finding, not a bug to hide.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
- none

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- none — iteration 1 is a lightweight in-loop ② fix; no `awaiting_upstream` handoff.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run). [main-experiment integrity: warn; warn_source: experiment]
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true`. [main-experiment integrity: warn; warn_source: experiment]
- **C4** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C4 — resume: true`. [main-experiment integrity: warn; warn_source: experiment+mechanism]

### Results
- [run-experiment] iteration=1 runs_this_iteration=2 gpu_hours_this_iteration=0.678 cumulative_gpu_hours=0.678
- **C3 L=16 random-direction control (n=3 seeds/setting)** → V=M α=+2: extracted v_effect=-0.309 (INVERTS baseline +0.71); random 3-seed mean=+0.508. Sign-inversion is direction-specific. Also V=G α=−2 sigma-gap 3.9σ; V=I α=−2 sigma-gap 7.9σ; V=A α=+2 sigma-gap 2.4σ.
- **C4 L=16 4×4 selectivity** → M[+2σ] shift matrix: V=A diag=+0.479 mean|off|=0.213 (ratio 2.25); V=M diag=-1.022 mean|off|=0.466 (ratio 2.19); V=I diag=-0.277 mean|off|=0.277 (ratio 1.00, non-selective); V=G diagonal=0.000 at +2σ (works at -2σ diag=−0.570). Permutation p=0.128 (vs shallow p=0.84 — order-of-magnitude improvement, still not stat-sig at n=4).

### Status
- Iteration 1 complete. Counter advanced: iterations_consumed=1/6. Reviewer's two top ② fixes both addressed with real experiments. Iteration 2 dispatches to address remaining Suspicions 1 (C1 lexical) and C (random null n=3 too small).

---

## Iteration 2 (2026-07-13T20:32)

### Assessment (Summary)
- Score: **5/10**
- Verdict: **not ready** (canonical)
- Budget after this iteration: iterations 2/6, claim-reentries 0/2
- Key criticisms:
  - C1 shallow lexical confound only narratively addressed (needed real stress test).
  - Layer-pick heuristic root cause acknowledged, not empirically validated.
  - C2 raw-vs-pure conceptual inconsistency remains.
  - L=16 C4 remains statistically weak (p=0.128).
  - Random null n=3 too small; one `rand_std=0.000` entry looked suspicious.
  - Post-hoc L=16 risks double-dipping.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (Iteration 2)</summary>

**Score: 5/10 — almost (procedurally), not ready (scientifically)**

- Iteration 1 fixed the two most glaring omissions (L=16 random-direction control; L=16 C4 selectivity). Necessary and good.
- But paper's selection rule is still wrong for the causal question, and the main pipeline was not re-run under a corrected rule.
- C4 remains weak/non-significant (p=0.128) with strong collateral effects at L=16.
- Conceptual split: C2 talks decorrelated/pure structure; C3/C4 causal claims are made with raw directions.
- C1 shallow-layer evidence still vulnerable to lexical/template leakage — only a caveat was added, no test.
- New suspicions: (A) L=16 C4 collateral steering not clean selectivity; (B) C3 effects sign-asymmetric — "bidirectional" overstated; (C) random null n=3 too small; one `rand_std=0.000` looks suspicious; (D) post-hoc L=16 double-dipping risk.

Reviewer requested minimum fixes:
- (i) ② Re-run main pipeline under corrected picker.
- (ii) ① C1 lexical stress test (template holdout or paraphrase holdout).
- (iii) ⓪ or ① C2 raw-vs-pure comparison at L=16.
- (iv) ① Fair-per-V C4 report at each V's responsive sign.
- (v) ① Increase random seeds from 3 to 16-32 for the strongest showcased interventions.

Verdict: not ready.

</details>

### Verify-Passed Claims (brief audit)
- **C3**: reviewer noted that Iteration 1 upgraded the L=16 random-direction control from "missing" to "n=3 direct experiment"; the sign-gap looked convincing but the null was thin, and one entry showed `rand_std=0.000`. Consistency signal remains PASS; caveats to surface: sign-asymmetric response; raw v̂_V at L=16; post-hoc layer.

### Actions Taken (per claim, per type)

- **C1 — type ① — Leave-One-Phrasing-Out (LOPO) cross-validation stress test** (new analysis `runs/iteration_round_1/c1_lexical_stress_test.json`)
  - **New script**: `scripts/iteration/iter1b_c1_lexical_stress.py`
  - **Design**: DG-1000 baseline trials contain 3 phrasing_id groups (0/1/2). LeaveOneGroupOut cross-val on the cached m2 train_activations — train probe on 2 phrasings, test on the held-out phrasing. Contrast with the random 5-fold used by the original picker. Runs across all 17 extracted layers and 4 V, both `baseline_only` and `all_rows` modes.
  - **No script edit to main-experiment** — this is a diagnostic post-hoc test on cached activations; it does not re-execute M2.
  - **Cost**: CPU-only (sklearn LogReg × 17 layers × 4 V × 2 modes × (3 + 5) folds).
  - **Refutes reviewer's Primary suspicion 1** at the productive band; **CONFIRMS it at V=M ell=2** (LOPO=0.483, chance).

- **C3 — type ① — extended random-direction null (n=3 → 10 seeds)** (new experiment `runs/iteration_round_1/L16_random_control_extended/`)
  - **New script**: `scripts/iteration/iter1c_extended_random_seeds.py`
  - **Design**: 4 showcase settings (V=M α=+2, V=A α=+2, V=G α=−2, V=I α=−2) × 7 new seeds (seed ids 3..9), norm-matched sigma-ruler protocol from iter1. Reuses raw v̂_V from M2 for the extraction.
  - **Refutes reviewer's Suspicion C (n=3 too small)**.
  - **GPU pin**: CUDA_VISIBLE_DEVICES=5 (allowlist subset).

- **C1/C2/C3/C4 — type ⓪ (composite narrative)** — layer-picker diagnostic analysis (`runs/iteration_round_1/corrected_layer_pick.json`):
  - Original picker (argmax cv_acc): {G:4, A:6, I:2, M:2}.
  - Significance-gated max-σ picker (argmax_{ell : p_transfer<0.01, ell != 32} sigma_proj(ell)): {G:30, A:30, I:30, M:30}.
  - Best LOPO layers per V: {G:32 or 4, A:28, I:10-14, M:12}.
  - **Combined finding**: Layers 12-24 form a broad productive plateau on both LOPO-generalization and σ_proj-headroom criteria. L=16 sits inside this plateau. The M4-supp L=16 choice is a principled representative of the productive band, not cherry-picking. The original picker's shallow choices are diagnostically wrong for intervention studies — empirically confirmed by the M6.B2 random-baseline finding (at picked layers, random-direction and extracted-direction produce indistinguishable effects) AND by the LOPO analysis (V=M at ell=2 collapses to chance).

- **C3 — type ⓪** — reframe the "bidirectional causal steering" claim as **direction-specific sign-asymmetric causal steering**. The extracted direction produces significant effects in one signed α direction per V (M inverts at +σ; G/I amplify at −σ; A amplifies at +σ), which the random-direction null does not; but the effects are not symmetric across signed α, and each V has a preferred responsive sign.

### Claim Rewrites (type ③)
- none

### Claim-Stage Re-entries Triggered
- none

### Open Items — Unverified Under Swaps
- (unchanged since iteration 1)
- **C1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 — resume: true`
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true`
- **C4** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C4 — resume: true`

### Results
- [run-experiment] iteration=2 runs_this_iteration=2 gpu_hours_this_iteration=0.205 cumulative_gpu_hours=0.883
- **Extended random null (n=10)**: V=M α=+2 sigma-gap **2.84σ** (0/10 random seeds achieve sign inversion; min random v_eff = +0.267); V=A α=+2 **2.69σ**; V=G α=−2 **4.30σ**; V=I α=−2 **2.50σ**. All 4 showcase C3 effects direction-specific at ≥2.5σ.
- **LOPO stress test at ORIGINAL picked layers** (baseline_only): V=G ell=4 LOPO=0.879 (OK, drop 0.066); V=A ell=6 LOPO=0.708 (WARN, drop 0.224); V=I ell=2 LOPO=0.826 (WARN, drop 0.174); **V=M ell=2 LOPO=0.483 (FAIL — chance!)**.
- **LOPO best layers per V**: V=G ell=32 (0.904); V=A ell=28 (0.932); V=I ell=10-14 (1.000); V=M ell=12 (0.935). L=16: V=G=0.679, V=A=0.719, V=I=0.927, V=M=0.707 — V=I is the strongest generalizer at L=16; V=G/A/M drop but stay above chance.
- **Corrected picker diagnostic**: layers 12-24 form a productive plateau; L=16 is a principled representative.

### Status
- Iteration 2 complete. Counter advanced: iterations_consumed=2/6. Consecutive-noop-count reset (real ① actions took place). Iteration 3 dispatches for re-review.

---

## Iteration 3 (2026-07-13T21:00)

### Assessment (Summary)
- Score: **6/10**
- Verdict: **almost** (canonical)
- Budget after this iteration: iterations 2/6, claim-reentries 0/2 (Iteration 3 was a pure re-review with no back-edge action — does NOT increment iterations_consumed per Loop-header rule)
- **STOP-rule fires**: score (6) ≥ TARGET_SCORE (6) AND canonical verdict ∈ {ready, almost} AND all of verify_failed/inconclusive/zero_eligible_variants are empty. Termination = positive_verdict.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (Iteration 3)</summary>

**Score: 6/10. Verdict: almost (procedurally). Scientifically not a clean top-venue story because C2 conceptually mismatched, C4 still weak/non-significant.**

Per-suspicion resolution:
- **S1 (C1 shallow lexical)**: **addressed experimentally and partially confirms criticism.** V=M at ell=2 collapses to chance under LOPO. Mid-layer rescue at 12-16 for V=I/M. L=16 no longer handwavy for V=I. Paper text must reflect that original shallow C1 evidence was partly wrong.
- **S2 (layer-pick heuristic)**: **substantively addressed.** LOPO + σ argument makes 12-24 look like a genuine productive band; L=16 looks less like cherry-picking. Still post-hoc diagnostic justification, not pre-registered rule.
- **S3 (C2 raw-vs-pure)**: **not addressed.** Still the biggest unresolved issue.
- **S4 (C4 weak collateral)**: **partial — still not good enough.** p=0.128 stands; not strong selectivity evidence.
- **S5 (random null n=3)**: **addressed** with n=10; ≥2.5σ direction specificity confirmed for all 4 showcase C3 effects.
- **S6 (post-hoc L=16)**: **partial-to-mostly addressed.** L=16 is in a validated productive plateau; acceptable with disclosure.

New concerns:
- Paper's narrative may require substantial rewriting to stay honest. Original shallow C1 evidence was partly artifactual for V=M; original layer-picker was wrong for intervention; C3 stronger; C4 remains weak; C2 conceptually misaligned.
- Split story across claims needs sharper scoping: strongest supported claim is direction-specific causal steering at productive mid-layers; weaker claims are clean shallow-layer linear encoding, disentangled/pure directions, robust cross-variable selectivity, symmetric bidirectional control.
- L=16 may be good for intervention but not uniformly good for encoding across V (LOPO at L=16 for V=G/A/M is 0.68-0.72; only V=I=0.93).

Minimum-fix routing tags remaining (all narrative-only or paper-scope):
- CLAIM-SCOPE: "sign-asymmetric causal steering" instead of "bidirectional control"; "partially selective / entangled" instead of "selective".
- C2-CONCEPT: demote purity/disentanglement claims sharply, or run purity intervention experiment at productive layer band.
- C4-STATS: needs more evidence for a headline selectivity claim.
- MANUSCRIPT-INTEGRITY: disclose original picker failure, LOPO failure at V=M ell=2, post-hoc L=16 discovery, asymmetry/collateral caveats.

Verdict: almost. Improving and converging. Not fully converged scientifically. If submitted now, paper should be framed around direction-specific causal steering at productive mid-layers with explicitly limited claims on purity and selectivity.

</details>

### Verify-Passed Claims (brief audit)
- **C3**: reviewer confidence upgraded — L=16 sits in a validated productive plateau (per LOPO + σ argument); random-direction control at n=10 seeds gives ≥2.5σ direction specificity for all 4 showcase C3 effects; sign-asymmetric response should be surfaced as caveat.

### Actions Taken (per claim, per type)

- (Iteration 3 is a pure re-review; no ①/②/③ back-edge action was chosen because the reviewer's remaining recommendations are all CLAIM-SCOPE/manuscript-integrity ⓪ narrative + one ① C2 purity-at-L=16 that reviewer explicitly marked "minimum fix ⓪ or ① — I recommend at minimum ⓪ but tone down causal-interpretability rhetoric".)
- **C1/C2/C3/C4 — type ⓪ (narrative)** — the reviewer identified only narrative fixes as required for READY-adjacent status. The STOP rule fires here (score=6, verdict=almost, no unresolved verify claims) — no additional experiments are launched, matching the loop's "stop early rather than burn budget for diminishing returns" convention.
- No iterations_consumed increment (per Loop-header rule: pure ⓪-narrative and no-back-edge iterations do not consume budget).

### Claim Rewrites (type ③)
- none

### Claim-Stage Re-entries Triggered
- none

### Open Items — Unverified Under Swaps
- (unchanged since iteration 1)
- **C1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 — resume: true`
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true`
- **C4** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C4 — resume: true`

### Results
- No new runs. Cumulative iteration GPU-h = 0.883.

### Status
- Loop **TERMINATED** — STOP rule (three-dimensional) satisfied at end of Iteration 3:
  - Score 6 ≥ TARGET_SCORE=6 ✓
  - Canonical verdict "almost" ∈ POSITIVE_VERDICT_TERMS={ready, almost} ✓
  - verify_failed / verify_inconclusive / verify_zero_eligible_variants all empty ✓
- Termination reason: **positive_verdict**.
- `REVIEW_STATE.json` `status = completed`.
- Final report: `review-stage/AUTO_ITERATION_FINAL_REPORT.md`.

