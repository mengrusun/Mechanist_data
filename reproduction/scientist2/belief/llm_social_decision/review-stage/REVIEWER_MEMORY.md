# Reviewer Memory

Persistent adversarial-review notes across iterations. Append-only; prior iterations are never rewritten.

## Iteration 1 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - Shallow-layer C1 signal is dominated by lexical/template leakage from paired-partner prompt design, not genuine social-variable encoding at those layers.
  - Layer-pick heuristic (argmax probe_cv_acc) is likely fundamentally wrong for intervention studies; probe accuracy selects easily decodable but low-leverage layers.
  - C3 strongest result at L=16 lacks the decisive random-direction control; could still be generic activation perturbation.
  - C4 was tested only where steering power was absent; must test at L=16 before claiming specificity.
  - Conceptual inconsistency: C2 claims "pure" directions but C3 success uses raw v̂_V rather than GS/LEACE pure directions.
  - "Purity" of GS direction is overclaimed — off-diag max 0.506 is a large residual coupling for a "free of confounds" statement.
  - LEACE failure at shallow layers points to representation entanglement — watch for overclaiming confound removal in the paper.
  - V=A projection-transfer p=0.06 is marginal — cannot be soft-inflated to "significant."
  - Model-swap variant used n=10/cell (planned 200); PASS verdict rests on qualitative pattern match — quantitative claims need caveat.
- **Previous suspicions addressed?**: n/a (first iteration).
- **Unresolved (carried forward)**: all of the above.
- **Patterns**:
  - The pipeline systematically selected shallow layers for its main analysis, then found the real effects at a supplementary deeper layer without the matching controls (random-direction, LEACE-purified) redone there.
  - Post-hoc L=16 results (M4-supp) create a narrative-vs-methodology gap: the pre-registered protocol failed; the successful protocol wasn't pre-registered.

## Iteration 2 — Score: 5/10, Verdict: not ready

- **New suspicions**:
  - A. L=16 C4 shows collateral steering / coupled latent subspace rather than clean selectivity (V=I shifts G/A/M by similar magnitudes; V=M shifts I strongly).
  - B. C3 effects are sign-asymmetric; "bidirectional causal steering" is overstated.
  - C. Random-direction null with only 3 seeds may be unstable/underpowered; one `rand_std=0.000` entry looks suspicious.
  - D. L=16 is post-hoc exploratory; manuscript must not oversell as confirmatory validation of the original pipeline.
- **Previous suspicions addressed?**:
  - Layer-pick heuristic: partial (acknowledged + alternative proposed, but no re-run under corrected picker).
  - L=16 random-direction control: addressed (n=3 direct experiment).
  - C4 at L=16: addressed as experiment; result partial/weak (p=0.128).
  - C1 lexical: sidestepped — narrative caveat only.
  - C2 raw-vs-pure: partial — narrative admits inconsistency, empirical link not made.
  - C2 "purity" wording: addressed rhetorically ("GS-decorrelated basis").
  - LEACE entanglement: unresolved.
  - V=A p=0.06: unresolved.
  - Swap n=10: unresolved.
- **Unresolved (carried forward)**: C1 lexical (still narrative only), layer-pick empirical validation, C2 raw-vs-pure conceptual link, C4 statistical strength, sign-asymmetry framing, post-hoc discovery framing.
- **Patterns**:
  - Iteration 2 fixed the two most glaring procedural omissions from Iteration 1 with real experiments. Continued pattern: authors would rather add narrative caveats than run stress tests unless pushed.

## Iteration 3 — Score: 6/10, Verdict: almost

- **New concerns**:
  - Paper's narrative may require substantial rewriting to stay honest with the new results (original shallow C1 evidence was partly artifactual for V=M; original layer-picker was wrong for intervention; C3 stronger; C4 remains weak; C2 conceptually misaligned).
  - Split story across claims needs sharper scoping: strongest supported claim is now direction-specific causal steering at productive mid-layers; weaker/unsupported claims are clean linear encoding at original shallow layers, disentangled/pure directions, robust cross-variable selectivity, symmetric bidirectional control.
  - L=16 may be good for intervention but not uniformly good for encoding across all V (LOPO at L=16 for V=G/A/M is 0.68-0.72; only V=I=0.93).
- **Previous suspicions addressed?**:
  - S1 (C1 lexical): **addressed experimentally** — LOPO stress test confirms the criticism at ell=2 for V=M (LOPO=0.483, chance) and rescues mid-layers 12-16 for V=I/M. Original shallow C1 evidence for V=M is essentially a phrasing detector.
  - S2 (layer-pick root cause): **substantively addressed** — LOPO scan + σ-headroom argument makes layers 12-24 the productive band; L=16 is a principled representative, not cherry-picking.
  - S3 (C2 raw-vs-pure): **not addressed** — biggest unresolved conceptual issue.
  - S4 (L=16 C4 collateral / non-significant): **partial** — no new C4 experiment; p=0.128 stands.
  - S5 (random null n=3 too small): **addressed** — n=10 extension; all 4 showcase C3 effects at ≥2.5σ.
  - S6 (post-hoc L=16): **partial-to-mostly addressed** — L=16 is in a validated productive plateau, acceptable with disclosure.
- **Patterns**:
  - Iteration converged on: **direction-specific causal steering at productive mid-layers** as the strongest defensible claim; original shallow-layer C1 encoding for V=M is artifactual; C4 selectivity remains weak; C2 purity remains conceptually misaligned unless narrowed to descriptive.
- **Bottom line at termination**:
  - Publishable if reframed around direction-specific causal steering at productive mid-layers, with narrowed claims on purity and selectivity, and explicit disclosure of the layer-pick correction and post-hoc L=16 discovery. Not publishable as originally framed.
