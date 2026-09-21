## C2: robustness = 1.00 (threshold = 0.5, eligible = 1/1)  →  ✅ PASS

- swap_variants_run: true
- Main-experiment verdict on C2: supported
- Main-experiment integrity (Phase 2): WARN (exp WARN / mech WARN — see
  `main_experiment_audit/EXPERIMENT_AUDIT.md` + `MECHANISM_AUDIT.md`)
- Variant integrity (Phase 9): WARN (exp WARN / mech WARN — see `variant_audit/EXPERIMENT_AUDIT.md`
  + `MECHANISM_AUDIT.md`). WARN, not FAIL → variant counts in both numerator and denominator.
- Variant counts (over `consistent_with_main_experiment`): 1 pass, 0 fail (of 1 eligible; 0 excluded
  for integrity reasons)
- Method dimension (SS-assignment algorithm: mkdssp → pydssp, on identical ESMFold-predicted
  structures): **matches the main experiment** (consistent=pass, claim_supported=pass). Pydssp-based
  helix fraction rises from 0.468 (α=0) to 0.795 (α=32), effect +0.327, Spearman ρ=0.886 p=6.4e-4 —
  closely tracking this same variant's own mkdssp-path readout (ρ=0.935, p=7.0e-5, effect +0.332)
  within 0.01–0.02 absolute helix fraction at every dose, and consistent with the main experiment's
  own mkdssp numbers at the overlapping doses (main: 0.475→0.834, effect +0.359).

**Scope of this PASS (read carefully — do not over-generalize):** this variant stress-tests
**only the SS-assignment-algorithm axis** — whether DSSP's specific 8-state HGI grouping rule vs
pydssp's independently-implemented hydrogen-bond-map 3-state algorithm agree on the same predicted
structures. They agree closely, so C2's dose-response conclusion is **robust to the choice of
SS-assignment tool**.

This PASS does **NOT** test the **structure-predictor axis** — whether ESMFold's own folding/
confidence behavior could be inflating the effect specifically on degenerate/low-complexity
high-α sequences. That concern was raised independently in this claim's own main-experiment
`EXPERIMENT_AUDIT.md` (Check E: at α=16/32, sheet content collapses to near-zero while ESMFold's own
pLDDT confidence rises above baseline, with no raw sequences preserved by the main experiment to
rule out degenerate-sequence collapse) and is the same axis flagged by C3's mechanism-audit FAIL.
An ESMFold→OmegaFold structure-predictor swap was attempted first (code written, reviewed, and
fixed) and abandoned as infeasible within this verify pass's network/time budget — see
`variants/method-swap-omegafold/PIVOT_NOTE.md` for full detail. The existing quality guardrails in
the main experiment (valid-ORF rate, pLDDT gating) are a partial mitigation for this untested axis,
not a substitute for an independent structure-predictor comparison.

Interpretation: C2's causal dose-response claim is robust to a genuine, independently-implemented
swap of the secondary-structure definition/tool. The structure-predictor axis remains a flagged,
untested robustness gap for a future verify pass (recommended: retry the OmegaFold or ColabFold swap
with a longer time budget or a pre-staged model-weight cache to avoid the network bottleneck hit
this pass).
