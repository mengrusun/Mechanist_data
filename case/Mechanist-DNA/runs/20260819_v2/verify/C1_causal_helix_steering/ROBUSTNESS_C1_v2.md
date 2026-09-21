## C1_v2 (narrowed, iteration-loop type-③ rewrite): robustness = 1.00 (method dimension; eligible = 3/3 predictors)  →  ✅ PASS (with major caveat)

- Produced by: iteration 2, type-③ lightweight in-loop claim narrowing (ancestor: C1).
- swap_variants_run: inline (no new GPU) — the method dimension is covered by M5's three-predictor re-scoring of the existing generations; model axis UNAVAILABLE (Evo2-7B pinned); dataset axis not separately run (composition confound is mechanism-intrinsic, not prompt-specific).
- Main-experiment integrity for C1_v2: experiment-audit Check E (scope) now **PASS** — the narrowed claim scope matches the evidence; Check F **WARN** (sequence-proxy, honestly caveated, no structural validation). Combined = WARN → admitted.
- Main-experiment verdict on C1_v2: **supported**.
- Robustness evidence (method dimension): the +14–16 pt predicted-%H uplift is reproduced by three distinct, heterogeneously-implemented predictors independent in construction of the primary ESM2 probe (ESM2-probe +15.5, GOR-windowed +15.0, Chou-Fasman +13.6; all MWU p<1e-6; bootstrap 95% CIs exclude 0). 3/3 predictors agree in direction and magnitude → method-robustness = 1.00.

### Final status
**PASS with a major caveat baked into the claim wording.** C1_v2 asserts only a causally-specific (vs norm-matched random), predictor-robust bias in *predicted* helix-propensity/composition — explicitly NOT physical α-helical structure. It is fully supported by the existing M2/M3/M5 evidence.

### Caveats (must travel with the claim in the paper)
- Composition confound is severe and NOT shown separable (Lys +0.396; low-complexity 0.21→0.43; GC 0.44→0.13); composition-matched test underpowered (5/200 GC overlap).
- No structural/folding validation (ESMFold CDN-blocked) — predicted %H ≠ verified tertiary structure.
- Dose response non-monotonic; quantitative claim restricted to the high-dose α≥8 regime.
- C2's evaluator validation is on NATURAL proteins only and does NOT validate the evaluator on these steered OOD sequences.

### Open scientific limitation (not closable in-loop)
Establishing genuine physical α-helix steering would require (a) structural folding of steered sequences (blocked at experiment time by the HF-mirror CDN for ESMFold 2.7 GB) and (b) a composition-decoupled steering protocol producing enough GC-overlap for a powered composition-controlled test — the current mechanism collapses composition at exactly the doses where the effect appears, so no such regime exists in the present data. Flagged as an Open Item.
