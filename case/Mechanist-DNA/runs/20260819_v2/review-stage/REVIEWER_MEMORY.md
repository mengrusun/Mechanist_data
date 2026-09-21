# Reviewer Memory

## Iteration 1 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - Primary-endpoint **proxy/compositional confounding**: the largest %H gains coincide with a GC collapse 0.40→0.13; the ESM2-probe %H may be reacting to amino-acid composition, not true helical propensity.
  - Independent sequence predictors (GOR, Chou-Fasman) are useful but **may share the same compositional vulnerability** as ESM2 — agreement among them is NOT structural proof.
  - GC-matching alone is **necessary but not sufficient**: must also control amino-acid composition, protein length, and low-complexity/repetitiveness, and stratify predictor scores by GC (and by composition where feasible).
  - Dose response is **non-monotonic** (α=4 dip 33.1); "monotonic" via an arbitrary Spearman>0.5 threshold is not acceptable. Only the high-dose range (α=8→16) appears consistently increasing; any "working range" must be prespecified/justified with uncertainty.
  - C2's natural-protein PASS does **not** validate the evaluator on steered, OOD sequences — must not be used rhetorically to prop up the C1 endpoint.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**: all of the above — pending the type-② re-analysis of the existing 400 generations.
- **Patterns**: primary metric evaluated far outside its validated (natural-protein) domain, exactly where the apparent effect co-emerges with a compositional shift — a proxy-validity/OOD-generalization gap.

## Iteration 2 — Score: 4/10, Verdict: almost

- **New suspicions**:
  - "mutually-independent predictors" overstates M5 — GOR/Chou-Fasman also use AA-level propensities and may share the composition sensitivity; call them "distinct/heterogeneous," not "independent."
  - "causally and specifically" risks reading as α-helix-specific mechanism; evidence supports only direction-specificity vs the ONE tested norm-matched random control.
  - "composition-level effect" risks implying a demonstrated mechanism rather than an unresolved composition confound; say "predictor-level effect strongly confounded with, not shown separable from, composition."
- **Previous suspicions addressed?**:
  - Single-probe-artifact concern: SUBSTANTIALLY RESOLVED (3 distinct predictors reproduce the effect).
  - Evaluator OOD/domain concern: addressed by separating C2's natural-protein validation from the steered-sequence interpretation.
  - Non-monotonicity: RESOLVED (now reported honestly; claim restricted to α≥8).
- **Unresolved (carried forward)**: composition matching underpowered (5/200 overlap); NO structural validation; predictor increase not separable from AT-rich/Lys/low-complexity shift; physical α-helix structure unverified.
- **Patterns**: evidence supports a robust sequence-predictor / output-steering phenomenon, but every stronger biological interpretation still rests on an unclosed OOD + composition-confounding gap. Routing = type-③ narrow (not falsify); after wording fixes, claim-support FAIL removed → C1 PASS-with-major-caveat, verdict "almost", but work is scientifically modest for a top venue (score capped ~4).

## Iteration 3 — Score: 5/10, Verdict: almost (confirmation review; ⓪-only refinements)

- **New suspicions**: minor residual wording — "causally-specific"→"intervention-specific relative to the tested control"; "predictor-robust"→"consistent across three sequence-based predictors"; "no gain" for random control should be an equivalence/statistically-supported statement, not mere non-significance; attach CIs to the helix shift, random-control contrast, ORF-validity, coding-likelihood. ALL applied (⓪).
- **Previous suspicions addressed?**: yes — the explicit "not physical structure / not composition-independent" caveat prevents the major overclaim; the endpoint repair (3 distinct predictors) is accepted.
- **Unresolved (carried forward — NOT closable in-loop)**: composition confound is intrinsic (the intervention itself induces the composition collapse, so a composition-independent helix claim cannot be recovered from this experiment); no structural validation (ESMFold CDN-blocked; local folders/biophysical assays could help but would NOT resolve the confound). Score capped ~5; unlikely to clear a top venue unless framed around controlled generative steering + rigorous confound analysis rather than protein-structure discovery.
- **Patterns**: converged honest ceiling — the defensible endpoint is "Evo2 steering produces an intervention-specific, predictor-consistent increase in PREDICTED helix propensity at high dose, tightly coupled to severe AT-rich/low-complexity composition change; no composition-independent or physical-structure claim is supported." No actionable back-edge remains.
