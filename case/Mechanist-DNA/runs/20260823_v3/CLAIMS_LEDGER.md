# Claim Ledger — Constrained property-steering of Evo2-7B toward high α-helical content

**Direction**: Steering Evo2-7B to generate DNA sequences whose encoded protein has high α-helical content, above the unintervened baseline
**Date**: 2026-08-23 → 2026-08-24
**Pipeline**: completed | **Iteration**: 7/10 "almost" (0/6)
**Models**: claim=claude-opus-4-8 (session), experiment=claude-opus-4-8 (session), verify=claude-opus-4-8 (session), iteration=claude-opus-4-8 (session); external reviewer gpt-5.4
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 helix gain (dose-response) | **SUPPORTED** (ρ=0.92, q=1.5e-4; +0.068→+0.16) | **PASS** robustness 1.00 (model-swap reproduces) | PASS held (score 7/10) | ✓ holds — supported, robust, reviewer PASS |
| C2 validity preserved (non-inferior) | **SUPPORTED** (0.968 vs 0.991, LB −0.035 > −0.05) | ⚪ INTEGRITY_ONLY (WARN; cap-deferred) | narrowed (⓪ wording fix) | ✓ supported, narrowed — GC/length off-targets documented |

---
## C1 — intervention raises α-helix fraction above baseline
- **Statement**: Under a targeted internal intervention during decoding, Evo2-7B generates DNA whose encoded protein has a mean α-helix fraction significantly higher than the unintervened baseline, with a monotone dose-response.
- **Origin**: task.md behavior (given) — helix-gain predicate of the bundled success criterion
- **Data**: Evo2-7B generations (baseline + steered) scored via ESMFold→DSSP; high/low-helix contrast set back-translated from labeled-SS proteins — provenance=constructed; used=1500 baseline (M1) + 13,500 steered (M3: 3 sites×6 coef×3 seeds×250) + 3,000 M4 controls; DEV/TEST split enforced
- **Models**: Evo2-7B
- **Method**: Committed — Steering Vectors / contrastive activation addition (CAA) at probe-selected residual **site 28**, coefficient grid [0,0.5,1,2,4,8]; ESMFold→DSSP α-helix; primary = permutation dose-response trend test on all-generations helix endpoint (TEST, invalid→0), BH-FDR; matched-control + same-norm sham for specificity; secondary SAE-clamp (L26) = negative
- **Main experiment**: **SUPPORTED** — site-28 CAA dose-response ρ=0.922, p=5.0e-5, BH-FDR q=1.5e-4; winning coef 1.0 helix **0.482 vs baseline 0.414 (+0.068, 95% CI [+0.040,+0.098])**, Cliff's δ=0.135, Hedges g=0.237; up to +0.161 at coef 4; matched-control −0.024 (n.s.) & sham −0.005 (n.s.) → CAA exceeds both (+0.093 p=9e-10; +0.073 p=1e-6); survives length-matching (+0.089 in [40,90]); no aa-composition bias
- **Verify**: robustness=**1.00** — method n/a / dataset n/a / **model PASS**; integrity=PASS (Phase 2 C1 PASS, Phase 9 variant PASS); verdict=**PASS**. Model-swap to `evo2_7b_262k` reproduced the dose-response (ρ=0.878, perm p=0.0057, winning-coef gain +0.094, δ=0.201); the length/GC off-target signature also transfers → robust family-level property, not a one-checkpoint artifact
- **Iteration**: PASS held — reviewer consistency check clean (dose ladder, ρ/q, winning gain+CI, specificity, length-matched gain, and model-swap robustness all reconcile with no numeric drift); no back-edge needed
- **Final**: ✓ holds — supported, robust (model-swap reproduces), reviewer PASS (round score 7/10 'almost'); no back-edge needed
- **Caveats**: Off-target — winning setting shortens protein length (−36.6 aa, δ −0.45) and lowers GC (−0.139), CAA-specific & documented (transfers to swapped model; C1 survives length-matching, no composition bias); per-gen effect small (δ=0.135) but CI excludes 0; SAE-clamp secondary arm negative
- **Artifacts**: refine-logs/EXPERIMENT_RESULTS.md, refine-logs/MECHANISM_ROUTING.md, results/m3/s28_caa.json, results/m4/specificity.json, results/m4/validity_frontier.json, experiments/steer_helix/
- **Figures**:
  - ![α-helix fraction vs steering coefficient for CAA at sites 28/30 and the SAE-clamp arm; site-28 CAA rises monotonically above the 0.414 baseline.](figures/C1/c1_dose_response.png) — vector: `figures/C1/c1_dose_response.pdf`
  - ![Direction-specificity at the winning setting: CAA raises helix (+0.068) while matched-control and same-norm sham show no gain.](figures/C1/c1_specificity.png) — vector: `figures/C1/c1_specificity.pdf`

---
## C2 — intervention preserves sequence validity
- **Statement**: At the C1-winning intervention setting, Evo2-7B's generated DNA sequences remain valid (well-formed / biologically plausible ORFs) at a rate non-inferior to baseline (pre-registered margin −0.05). Validity is preserved within the margin, but not all other sequence properties are: the intervention substantially shifts GC content (−0.139) and inferred protein length (−37%), which are documented, CAA-specific off-target effects. _(tightened in iteration ⓪ — supersedes the prior "off-target properties not degraded" overclaim flagged by the verify Phase-2 WARN.)_
- **Origin**: task.md HARD validity requirement — validity-preservation predicate of the bundled success criterion
- **Data**: Evo2-7B full generated set (baseline vs winning CAA setting) — per-sequence validity + off-target (β-sheet, GC, Evo2 NLL, length, aa-composition) — provenance=constructed; used=750 gens/condition at winning coef 1.0 vs baseline (M4), full set incl. invalid
- **Models**: Evo2-7B
- **Method**: Non-inferiority test on validity rate over the full generated set (pre-registered margin −0.05, one-sided 95% LB) at the C1-winning setting; off-target comparison vs baseline; validity frontier across the coefficient sweep
- **Main experiment**: **SUPPORTED (non-inferior)** — validity **0.968** (winning coef 1.0) vs **0.991** baseline, Δ=−0.023, one-sided 95% LB −0.035 > margin −0.05 → non-inferior; β-sheet & aa-helixfav unchanged, NLL +0.061 (< 1.5 cap); off-target GC −0.139 & length −36.6 aa documented. Frontier: coef ≤1 pass NI, coef ≥2 fail (dose–validity Pareto)
- **Verify**: ⚪ **INTEGRITY_ONLY** — integrity=WARN (Phase-2 audit: the "off-target not degraded" wording overclaims vs the documented −37% length / −0.139 GC shift; the non-inferiority test itself is clean); verdict=INTEGRITY_ONLY, swap-test deferred (max_verify_claims cap=1). Upgrade via `/auto-verify C2 — resume: true`
- **Iteration**: INTEGRITY_ONLY held — NI test clean; the only issue was the "off-target not degraded" overclaim wording, resolved by a ⓪ narrative-only fix (0 GPU-h). Narrowed to: validity (well-formedness) preserved within the NI margin, but GC and protein length are explicitly NOT preserved (documented off-targets)
- **Final**: ✓ supported, narrowed — validity non-inferior at winning setting; GC/length off-targets now explicitly documented (wording corrected in iteration); Stage-2 swap-test deferred (max_verify_claims cap)
- **Caveats**: Non-inferiority holds only at conservative coefficients (≤1.0); coef ≥2 fails the margin — reported success is the frontier knee, not max-helix; off-target GC/length shifts documented (non-disqualifying)
- **Artifacts**: refine-logs/EXPERIMENT_RESULTS.md, results/m4/validity_frontier.json, verify/C2_validity_preserved_noninferior/ROBUSTNESS.md, verify/C2_validity_preserved_noninferior/main_experiment_audit/
- **Figures**:
  - ![Helix-vs-validity frontier for site-28 CAA: helix gain rises with coefficient while validity falls; winning coef 1.0 is the max gain with validity non-inferior (LB > −0.05 margin).](figures/C2/c2_validity_frontier.png) — vector: `figures/C2/c2_validity_frontier.pdf`

---
## Journey Summary
- **Claim**: given behavior faithfully split into C1 (helix gain) ∧ C2 (validity preserved); top idea = constrained property-steering of Evo2-7B
- **Mechanism strategy**: Location → Causal Intervention (locate α-helix representation via probes + Evo2 SAE features, then steer via contrastive activation addition / SAE-feature clamp with a dose grid)
- **Mechanism routing**: family=Representation & Parameter Analysis / Steering Vectors (CAA) [primary, carries the result]; SAE-clamp (Goodfire Evo-2 L26) secondary = negative; localization=Probing / Residual Stream States; committed:true
- **Experiment**: 12 runs, ~27.3 GPU-h, headline POSITIVE — CAA steering along a site-28 residual direction gives a specific, dose-dependent α-helix increase in Evo2-7B's DNA-encoded proteins; validity non-inferior at the conservative winning coefficient
- **Verify**: 2 target claims: 1 PASS (C1, robustness 1.00 — model-swap to evo2_7b_262k reproduces the dose-response ρ=0.878, gain +0.094) / 0 FAIL / 1 INTEGRITY_ONLY (C2, deferred by max_verify_claims cap); integrity Phase2 WARN (C2 off-target wording) / Phase9 PASS; ~1.5 GPU-h
- **Iteration**: 0/6 iterations (converged at entry — no blocking claims), claim-reentries 0/2, score 7/10 verdict 'almost', termination=positive_verdict; 1 ⓪ narrative fix (tightened C2 off-target wording), 0 GPU-h; external reviewer gpt-5.4
- **Figures**: 3 across 2 claims (C1: dose-response line + specificity bar; C2: validity-frontier line); 0 judgment-skipped; 0 render-skipped, 0 errored

## Open Items
- Off-target shifts at the winning setting are CAA-specific and documented (protein length −36.6 aa, GC −0.139); C1 survives length-matching (+0.089 within [40,90]) and shows no aa-composition bias, and C2 validity stays non-inferior. Verify found the length/GC signature ALSO transfers to the swapped model (evo2_7b_262k) → a robust family-level property of the CAA lever, not a one-checkpoint artifact (C1 helix gain still holds).
- C2 swap-test DEFERRED (INTEGRITY_ONLY, max_verify_claims cap=1): audit passed but Phase-2 flagged a WARN — the "off-target not degraded" wording overclaimed vs the documented shifts (now corrected in the C2 statement). The validity non-inferiority test itself is clean. Optional upgrade via `/auto-verify C2 — resume: true` (swap-test the length-regressed endpoint), fits the remaining budget.
- SAE-clamp secondary arm is an honest negative — helix DECREASES with clamp strength (ρ=−0.95), driven by SAE recon_rel_error≈1.0; C1 rests on the CAA arm, not the SAE arm.
- Paper-side caveats (reviewer Section 8, score 7/10 'almost'): the structural endpoint is proxy-heavy (ESMFold→DSSP on translated ORFs, not experimental structures); generalization is demonstrated within the Evo2-7B family only (7B + 262k-context checkpoint), not across architectures. Per-generation effect size is modest (δ≈0.135) despite the strong, robust trend.
- Budget: ~28.8 / 40 GPU-h used this round (iteration spent 0) → ~11.2 GPU-h unused and available for the optional C2 upgrade or a next round.
- External reviewer ran via the gpt-5.4 HTTP fallback (MCP llm-chat tool not exposed to subagents, but the endpoint was reachable — genuine external review). cost.json not emitted by sanity/M1/M2 (GPU-hours estimated; M3/M4/verify reconstructed) — noted in tracker, not fabricated.
