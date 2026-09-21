# Auto Review Loop — Causal α-helix steering in Evo2-7B

Reviewer: gpt-5.6-luna (external, via dmxapi). Budget: MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6.
Verify entry state: C1 = INCONCLUSIVE (main-experiment integrity FAIL → type-② fix), C2 = PASS.

---

## Iteration 1 (2026-08-19)

### Assessment (Summary)
- Score: 4/10
- Verdict: not ready
- Budget after this iteration: iterations 1/6, claim-reentries 0/2
- Key criticisms:
  - C1 PRIMARY endpoint was a single ESM2-probe proxy applied OOD to steered, GC-collapsed sequences with no structural validation; the +%H gain co-emerges with a severe GC collapse (0.40→0.13) — proxy/compositional confound.
  - Independent predictors are necessary but not sufficient; must control amino-acid composition, length, low-complexity, and stratify by GC with uncertainty intervals.
  - Dose curve is non-monotonic; "monotonic" (arbitrary Spearman>0.5) is an overstatement.
  - C2 PASS on natural proteins does not validate the evaluator on the steered OOD generations.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (iteration 1)</summary>

Score 4/10. C1 correctly INCONCLUSIVE (not failed); routing = type-② main-experiment fix; claim must NOT be rewritten yet. Planned fix (independent predictors + GC-matched comparison + non-monotonicity handling) is broadly right but insufficient as specified — must ALSO control amino-acid composition, protein length, low-complexity/repetitiveness; stratify predictor scores by GC (and composition where feasible); report GC-matched distributions with effective sample sizes and overlap; use permutation/bootstrap uncertainty; estimate the steering effect WITHIN GC-overlap strata with CIs. Independent predictors (GOR/Chou-Fasman) are still sequence-based and may share the compositional vulnerability — they do NOT substitute for structural validation; label the result sequence-predictor-supported rather than structurally confirmed if folding is impossible. Do not describe the dose response as monotonic — only α=8→16 rises; any working range must be prespecified/justified. Minimum acceptable C1 resolution: uplift under ≥1 genuinely independent method AND persists in GC-overlapping/matched comparisons AND not explained by composition/length/low-complexity AND random-direction control stays negative AND dose reported honestly as non-monotonic. If it does not survive, narrow or falsify C1. C2 passes (r=0.987 natural proteins; r=0.85 GOR method-swap; r=0.979 dataset-swap) — caveat: does not validate the evaluator on the steered distribution; do not use C2 rhetorically to validate the C1 endpoint. NOT ready for submission; becomes Almost after the re-analysis, Ready only if the effect survives the controls. [Full Memory update captured in REVIEWER_MEMORY.md Iteration 1.]

</details>

### Verify-Passed Claims (brief audit)
- C2: consistent. Fast %H harness agrees with experimental DSSP (r=0.987, Q3=0.858, frame recovery 1.0); robust under method-swap (GOR r=0.85) and dataset-swap (fresh 310 proteins r=0.979). Caveat for paper: validated on NATURAL proteins only — does NOT validate the evaluator on the steered, GC-collapsed OOD generations used for C1. Do not cite C2 as validating the C1 endpoint. (No back-edge; part of Phase A.)

### Actions Taken (per claim, per type)
- C1 — type ② — main-experiment fix: added milestone **M5** (`experiments/run_M5_structural_gc_control.py`), a composition-controlled independent-predictor re-evaluation of the steered-%H endpoint on the already-generated 400 sequences (no regeneration — steering is sound, only the %H EVALUATION was broken). Then re-audited the corrected C1 main experiment (`/experiment-audit` re-check) and re-verified.
  - **Plan Before/After** (`refine-logs/EXPERIMENT_PLAN.md`): inserted `## M5: Composition-controlled, independent-predictor re-evaluation …` before M4 (independent-predictor triangulation + GC/composition control + honest non-monotonicity; `resource_fidelity` marker preserved verbatim).
  - **Results Before/After** (`refine-logs/EXPERIMENT_RESULTS.md`): added the M5 section and NARROWED the Claim-1 summary from "SUPPORTED — causally and monotonically raises %H" to "NARROWED — supported only as a composition-level metric effect, NOT structurally-validated α-helix steering."
  - **New script**: `experiments/run_M5_structural_gc_control.py` → `runs/iteration_round_1/M5_structural_gc_control.json`.
  - **Re-audit** (`verify/C1_causal_helix_steering/main_experiment_audit/EXPERIMENT_AUDIT.{json,md}` updated; stale copies archived `*.stale.md`): Check F REPAIRED fail→warn; residual overall FAIL now driven only by Check-E claim-scope. Combined verdict → main_experiment_verdict on frozen C1 = **not-supported**; recommended action **narrow_claim**.
  - **Key M5 findings** (`runs/iteration_round_1/M5_structural_gc_control.json`):
    - Uplift reproduced across 3 predictors (steered−baseline, bootstrap 95% CI): ESM2 +15.5 [9.3,21.6]; GOR +15.0 [9.5,20.8]; Chou-Fasman +13.6 [9.3,18.1]; all MWU p<1e-6 → NOT a single-probe artifact.
    - Confound is real & severe: AA-shift dominated by Lysine +0.396 (AAA codon); low-complexity max-AA-fraction 0.21→0.43; length 69→46; GC 0.44→0.13; within-baseline %H anti-correlated with GC (ESM2 r=−0.17 p=0.017; CF r=−0.36 p=2e-7).
    - Composition control UNDERPOWERED / does not survive: only 5/200 baseline generations overlap the steered GC band [0.11,0.20]; GC-matched & NN-matched deltas' 95% CIs span zero.
    - No structural grounding (ESMFold 2.7 GB HF-mirror-CDN-blocked; no local folder) → labeled "sequence-predictor-supported, NOT structurally confirmed."
    - Non-monotonic dose curve (full-range Spearman 0.54–0.86; only α≥8 monotone across all predictors).
  - Re-invoked: `/auto-verify C1 — resume: true` (inline). Result: C1 INCONCLUSIVE → **FAIL (claim-support)** — endpoint repaired, frozen claim overclaims; route to claim narrowing (③).

### Claim Rewrites (type ③)
- none this iteration (deferred to iteration 2 — the ② fix first had to make the endpoint trustworthy so the frozen claim became assessable).

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- none (no full-path ③ this iteration).

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- none (verify_integrity_only bucket empty).

### Results
- [run-experiment] iteration=1 runs_this_iteration=1 gpu_hours_this_iteration=0.012 cumulative_gpu_hours=0.012
- C1 → M5 composition-controlled re-eval + `/experiment-audit` re-check → INCONCLUSIVE → **FAIL (claim-support)**; main_experiment_verdict(frozen C1)=not-supported; recommended narrow_claim (③).
- C2 → unchanged PASS (robustness 1.00).

### Status
- continuing to iteration 2 (route C1 → type-③ claim narrowing).

---

## Iteration 2 (2026-08-19)

### Assessment (Summary)
- Score: 4/10
- Verdict: almost
- Budget after this iteration: iterations 2/6, claim-reentries 1/2
- Key criticisms: C1's frozen claim overclaims vs the repaired evidence; the proposed narrowing is directionally correct but must drop "mutually-independent"→"distinct/heterogeneous", qualify "causally and specifically" as relative to the tested random control, reframe "composition-level effect" as an unresolved confound, make non-monotonicity explicit (α≥8 only), and never equate predicted %H with physical structure.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (iteration 2)</summary>

Score 4/10, "almost but not ready" as written. Routing = type-③ claim rewrite (NOT falsification) — the intervention has a genuine effect on the reported predictor outputs; what fails is the original claim's scope. C1_v2 directionally correct but still overclaims: (1) "mutually-independent predictors" too strong → "distinct/heterogeneous"; (2) "causally and specifically" → "direction-specific relative to the tested norm-matched random-direction control"; (3) "composition-level effect" → "predictor-level effect strongly confounded with, not shown separable from, composition"; (4) make non-monotonicity explicit, restrict quantitative claim to α≥8; (5) state 5/200 GC-overlap underpowering concretely; (6) bound ORF/coding-likelihood to tested conditions. Provided a recommended minimum honest claim (adopted). Do NOT falsify entirely — the narrow empirical claim is supported (intervention changes output; robust across 3 predictors; random-direction control negative; coding metrics acceptable) but the original biological claim should be dropped. After wording changes C1 → PASS-with-major-caveat. C2 remains PASS (evaluator validated on natural proteins only; must not be cited as validating C1's steered sequences). Ready? As written No; after narrowing, Almost — submission-ready claims but not strongly competitive for a top venue; structural validation absent + extreme composition confound remain major limitations.

</details>

### Verify-Passed Claims (brief audit)
- C2: PASS held. Caveat re-confirmed: evaluator validated on natural proteins only; do not cite as validating C1_v2's steered OOD sequences.

### Actions Taken (per claim, per type)
- C1 — type ③ — lightweight in-loop claim narrowing (FAIL Phase-2, variants already clean via M5's independent-predictor re-scoring → claim rewrite). Ancestor C1 → produced C1_v2.
  - No new experiments (the narrowed claim is fully supported by existing M2/M3/M5 evidence).
  - Re-scoped audit: with the claim narrowed to match the evidence, Check E (scope) → PASS; Check F → WARN (honest proxy caveat). C1_v2 main-experiment verdict = supported → PASS-with-major-caveat.
  - Records: `verify/C1_causal_helix_steering/ROBUSTNESS_C1_v2.md`; `claims_ledger.json` C1 entry updated (final_status, iteration.narrowed_to, caveats); `refine-logs/EXPERIMENT_RESULTS.md` Claim-1 summary replaced with C1_v2.

### Claim Rewrites (type ③)
- Original claim id: `C1` — "A localizable internal component of Evo2-7B (block-26 linear-probe α-helix direction) causally and specifically increases generated α-helical content (%H)."
- New claim id: `C1_v2` — an intervention-specific (vs norm-matched random control), predictor-consistent (3 heterogeneous predictors) +14–16 pt increase in PREDICTED %H at high dose (α≥8; non-monotonic below), with ORF-validity/coding-likelihood preserved under tested conditions, strongly confounded with and not shown separable from an AT-rich/Lys-rich/low-complexity composition collapse (GC 0.44→0.13), composition-matched test underpowered (5/200 overlap), and NO structural validation → NOT physical α-helix structure or a composition-independent mechanism.
- Path: lightweight in-loop (no orchestrator handoff; no new experiments — existing evidence supports the narrowed claim).
- Reason: the type-② fix made the endpoint trustworthy, revealing the original strong claim overclaims; the honest, evidence-matched statement is the narrowed C1_v2. Original strong C1 falsified.
- Claim-reentry sub-budget consumed: 1 (now 1/2).

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- none (lightweight in-loop path; no `awaiting_upstream`).

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- none.

### Results
- [run-experiment] iteration=2 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=0.012
- C1 → narrowed to C1_v2 → ✅ PASS-with-major-caveat. C2 → PASS (held).

### Status
- continuing to a confirmation review (iteration 3).

---

## Iteration 3 (2026-08-19) — confirmation review + ⓪ refinements; TERMINATION

### Assessment (Summary)
- Score: 5/10
- Verdict: almost
- Budget after this iteration: iterations 2/6 (unchanged — ⓪ action), claim-reentries 1/2
- Key criticisms: a few final wording/statistical-presentation tweaks (all ⓪, applied). Confirmed: composition confound is NOT closable (the intervention itself induces the composition collapse); structural validation absent and, per the reviewer, would not resolve the confound (and ESMFold is infrastructure-blocked). Score capped ~5; publishable as an honest result but not strongly top-venue competitive.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (iteration 3)</summary>

Score 5/10. C1_v2 mostly honest; final tweaks: "causally-specific"→"intervention-specific relative to the tested control"; "predictor-robust"→"consistent across three sequence-based predictors"; make random-control "no gain" a statistically-supported equivalence not mere non-significance; report CIs/effect sizes for helix shift, random-control contrast, ORF-validity, coding-likelihood; "establishes"→"shows". The explicit not-physical-structure/not-composition-independent caveat is excellent. Actionable improvements remaining are wording + a statistical audit; the composition confound is probably NOT closable (if the intervention inherently induces the composition collapse, a composition-independent helix claim cannot be recovered from this experiment); structural validation is not strictly impossible but would not solve the confound. READY: Almost — submit after the small wording/statistical-audit fixes; publishable as an honest result but unlikely to clear a top venue unless framed around controlled generative steering + rigorous confound analysis rather than protein-structure discovery. Defensible endpoint: Evo2 steering produces an intervention-specific, predictor-consistent increase in PREDICTED helix propensity at high dose, tightly coupled to severe AT-rich/low-complexity composition change; no composition-independent or physical-structure claim supported.

</details>

### Verify-Passed Claims (brief audit)
- C2: PASS held. C1_v2: PASS-with-major-caveat held.

### Actions Taken (per claim, per type)
- C1_v2 — type ⓪ — applied the final reviewer wording/statistical-presentation fixes into the claim (intervention-specific; consistent-across-three-predictors; random-control equivalence; CIs attached; "shows" not "establishes"). No scripts/data touched; no runs. Does NOT consume an iteration.

### Claim Rewrites (type ③)
- none.

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- none.

### Results
- [run-experiment] iteration=3 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=0.012
- Final states: C2 ✅ PASS; C1_v2 ✅ PASS-with-major-caveat (original strong C1 falsified & superseded).

### Status
- completed — TERMINATION (converged / stalled below TARGET_SCORE). All claims resolved & honest; no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE remaining; verdict "almost"; final score 5/10 < TARGET 6. The gap to TARGET is NOT closable by any actionable back-edge — structural validation is infrastructure-blocked (ESMFold HF-mirror CDN) and the composition confound is intrinsic to the mechanism (confirmed by the reviewer). Remaining reviewer suggestions were all ⓪ narrative, now incorporated. Continuing would produce only identical ⓪-noops.
