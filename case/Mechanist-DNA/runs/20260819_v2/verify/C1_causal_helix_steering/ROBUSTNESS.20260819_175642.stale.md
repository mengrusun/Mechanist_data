## C1: robustness = — (threshold = 0.5, eligible = —)  →  🟡 INCONCLUSIVE

- verdict: INCONCLUSIVE
- swap_variants_run: false
- Main-experiment verdict on C1: supported (main experiment); but Phase 2 integrity gate FAILED
- inconclusive_reason: main-experiment integrity broken — see verify/C1_causal_helix_steering/main_experiment_audit/EXPERIMENT_AUDIT.md
- Phase 2 combined: FAIL (experiment-audit FAIL, mechanism-audit WARN)
- Variants: none (Phases 3–10 short-circuited for C1)

Interpretation: The PRIMARY causal-steering claim was NOT stress-tested, because its main-experiment evaluation methodology did not pass the Phase 2 integrity gate. The block-26 probe steering direction genuinely raises the *proxy* %H (ESM2-650M + linear probe) and the mechanism sweep + norm-matched random control are adequate (mechanism-audit = WARN, not FAIL), but the primary endpoint is a synthetic proxy validated only on natural proteins and then applied to heavily-steered, GC-collapsed, out-of-distribution generated sequences with no structural validation — and the largest %H gains coincide exactly with a severe GC-composition collapse (0.40→0.13), leaving helix-propensity vs. compositional-drift unresolved. Running method/dataset swaps on top would only compute robustness around a broken anchor.

Iteration instruction (route to main experiment, NOT the claim): rerun /auto-experiment to (1) structurally validate the %H of a steered-sequence subset (fold a subset and DSSP it, or use an independent SS predictor that does not share the ESM2 backbone) so the endpoint is grounded on the actual out-of-distribution generations, and (2) add a GC-matched / composition-controlled analysis that separates α-helix propensity from the AT-rich-codon compositional shift. Then re-invoke `/auto-verify C1 — resume: true` (Phase 2 will re-audit the corrected main experiment).
