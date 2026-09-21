## C1: robustness = — (main-experiment endpoint REPAIRED; frozen claim not-supported)  →  ❌ FAIL (claim-support; was 🟡 INCONCLUSIVE)

- verdict: FAIL  (transitioned from INCONCLUSIVE after the iteration-loop type-② main-experiment fix)
- swap_variants_run: false  (model axis UNAVAILABLE — Evo2-7B HARD-pinned; method axis covered inline by M5's ESM2-independent GOR + Chou-Fasman re-scoring; dataset axis not run — the composition confound is mechanism-intrinsic, not prompt-specific, so a swap would reproduce the same confounded metric)
- Main-experiment integrity (Phase 2 RE-AUDIT): experiment-audit = FAIL→ driven ONLY by Check-E claim-scope; mechanism-audit = WARN (unchanged). Check F (eval type) REPAIRED fail→warn.
- Main-experiment verdict on frozen C1: **not-supported**

### Why the state changed (INCONCLUSIVE → FAIL)
The prior INCONCLUSIVE was set because Check F was FAIL — the %H endpoint was a single ESM2 proxy applied out-of-distribution with no structural validation, so **no verdict was computable** (robustness around a broken anchor is meaningless). The type-② fix (milestone M5, `runs/iteration_round_1/M5_structural_gc_control.json`) repaired that anchor:
- The +%H uplift is reproduced by two predictors independent of ESM2 (GOR +15.0 [9.5,20.8]; Chou-Fasman +13.6 [9.3,18.1]; ESM2 +15.5 [9.3,21.6]; all MWU p<1e-6) → not a single-probe artifact.
- The GC/composition confound is now quantified and disclosed (Lys +0.396; low-complexity 0.21→0.43; length 69→46; GC 0.44→0.13; within-baseline %H anti-correlated with GC).
- The composition-controlled comparison is honestly reported as **underpowered and non-surviving** (only 5/200 baseline generations overlap the steered GC band; matched deltas' 95% CIs span zero).
- Non-monotonicity reported honestly (α=4 dip; only α≥8 monotone).
- Grounding explicitly labeled "sequence-predictor-supported, NOT structurally confirmed" (ESMFold CDN-blocked).

With the anchor repaired (Check F warn), a main-experiment verdict is now **computable** — and it is **not-supported**: the trustworthy endpoint reveals a **composition-level metric effect**, not the "causal, specific, structurally-real α-helix steering" the frozen claim asserts. The only residual audit failure is Check-E scope: **the frozen claim text overclaims relative to the corrected evidence.** That is a claim-narrowing problem, not an evaluation-methodology problem — hence a claim-support **FAIL**, not a continued INCONCLUSIVE.

### fail_reason
frozen claim overclaims vs the repaired, trustworthy endpoint — the evidence supports only a composition-level, causally-specific-vs-random %H-metric shift, not structurally-validated α-helix steering. See `main_experiment_audit/EXPERIMENT_AUDIT.json` (Check E) and `../../runs/iteration_round_1/M5_structural_gc_control.json`.

### Iteration instruction (route to CLAIM narrowing, type ③)
The main experiment is repaired; do NOT do more ② work. Narrow C1 to the claim the evidence actually supports (a causally-specific — vs norm-matched random — composition-level shift toward helix-propensity-scoring, AT-rich-codon sequences that raises the predicted %H metric across independent predictors without loss of ORF-validity/coding-likelihood), explicitly dropping the unvalidated structural assertion and disclosing the Lys/low-complexity/GC-collapse confound and the absence of structural validation. Reviewer-recommended action: **narrow_claim**.
