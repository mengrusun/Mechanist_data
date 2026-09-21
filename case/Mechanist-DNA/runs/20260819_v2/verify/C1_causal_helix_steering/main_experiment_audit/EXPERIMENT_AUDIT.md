# Experiment Audit — C1 (main experiment, M1–M5) — RE-AUDIT after iteration-loop type-② fix

**overall_verdict: FAIL** (fail_driver = **E_scope**, a claim-scope overclaim — NOT an evaluation-integrity break) · reviewer: gpt-5.6-luna-2026-07-09

| Check | Verdict | Note |
|---|---|---|
| A. GT provenance | PASS | SS-probe labels from real experimental PDB/DSSP. |
| B. Score normalization | PASS | No self-referential normalization; random control norm-matched. |
| C. Result-file existence | PASS | M1–M5 values on disk (incl. `runs/iteration_round_1/M5_structural_gc_control.json`). |
| D. Dead code | WARN | M5 re-executes the ORF→translate→predict path on the stored generations. |
| E. Scope | **FAIL** | The frozen C1 wording ("causally and specifically increases generated α-helical content") overclaims vs the corrected evidence: a composition-level metric effect, confounded with a Lys/low-complexity/GC collapse, not surviving (underpowered) GC control, and not structurally validated. **Fix = narrow the claim (③), not the experiment.** |
| F. Eval type | **synthetic_proxy → WARN (REPAIRED from FAIL)** | +%H uplift now triangulated across two ESM2-independent predictors (GOR +15.0, Chou-Fasman +13.6 vs ESM2 +15.5), composition-controlled, and labeled non-structural. Residual WARN: all estimators are sequence proxies on composition-shifted OOD sequences; GC-matched analysis underpowered; no structural folding obtained (ESMFold CDN-blocked). |

**What changed vs the prior audit:** the type-② fix (M5) repaired Check F (FAIL→WARN) — the primary endpoint is no longer a single unvalidated proxy. With the anchor repaired, a main-experiment verdict on the frozen claim is now **computable** and is **not-supported**. The residual overall FAIL is driven entirely by **Check E** (the frozen claim text is broader than the evidence). This is the intended INCONCLUSIVE→FAIL transition: evaluation integrity is admissible (WARN), and the remaining problem is a claim-scope overclaim.

**main_experiment_verdict_on_frozen_C1:** not-supported
**Recommended iteration action:** narrow_claim (③) — narrow C1 to the composition-level, causally-specific-vs-random %H-metric effect it actually supports; drop the unvalidated structural assertion; disclose the confound and the absence of structural validation.
