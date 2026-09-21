# Mechanism Audit Report — Claim C2

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C2 — Amplifying the C1 α-helix feature set during Evo2-7B autoregressive DNA generation increases the encoded-protein α-helix fraction, monotonically with amplification strength up to an optimum.
**Linked milestones**: M1, M2

## Overall Verdict: WARN
*C2's mechanism-rigor verdict — whether the steering intervention backing this claim was tuned with
the necessary controls.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: WARN
- Triggered: yes — `code/mechanism.py`'s `Steerer._hook` (`x + alpha*base_dir`) and
  `code/m2_dose_response.py`'s `--alpha` grid dispatch.
- Intervention type: fixed additive steering direction at one hook site
  (`blocks.26.post_norm`); `base_dir` is precomputed once from the frozen C1 feature set S and is
  NOT re-derived per α.
- Sweep grid: α ∈ {-2, 0, 1, 2, 4, 8, 16, 32} × seed ∈ {42,200,201} = 24 runs. Includes α=0
  baseline and a negative sign-check (α=-2). **Does not span ≥3 orders of magnitude** — positive
  doses run 1→32 (32×), well under the catalogue's 1000× benchmark.
- σ_proj scaling used: **no**. `s_f` (the per-feature scale) is the mean nonzero activation
  magnitude of each feature from the M0 discovery data — a raw empirical-magnitude scale, not a
  projection standard deviation. No σ_proj is computed anywhere in the codebase.
- Capability metric logged: yes, at every dose — valid_orf_rate, gated_pass_rate (pLDDT≥50 gate),
  pLDDT_mean, sheet_mean.
- Plateau range: reviewer-identified clean region ≈ [4, 8] — helix rises from 0.475→0.571 with
  valid-ORF at/above baseline and pLDDT not collapsed; α≥16 breaks the ~10%-degradation capability
  tolerance (valid-ORF drops >20% relative to baseline at α=16).
- Locked α: the reported headline "optimal α*=32" does **not** qualify as a valid plateau lock under
  this check (capability tolerance already violated by α=16); the reviewer instead evaluated the
  separately-reported "conservative clean optimum" α=8 as the only defensible lock candidate.
  Position in plateau: **edge** (α=8 sits at the upper edge of the ≈[4,8] clean region, not the
  middle).
- Random-direction control: **none within M2's own dispatched runs.** M3 (a separate milestone
  backing claim C3) runs a matched-control feature arm at α∈{0,4,8,16}, cross-referenced in the plan
  text, but this does not cover M2's reported α*=32 and is not part of M2's own sweep.
- Sign pattern: n/a (M2 is a single-direction positive-effect test with a negative-α sanity check,
  not an asymmetric dual-group protocol).
- Output-case spot-check: **not available.** `mechanism.py`'s `generate_and_readout()` explicitly
  discards the generated/translated protein sequence text (`rec.pop('prot', None)`) after computing
  metrics; no raw sequence-level output was ever saved, so no coherence/degeneracy spot-check is
  possible at any α.
- Evidence: `code/mechanism.py` (`Steerer.__init__`, `Steerer._hook`, `compute_feature_scales`,
  `generate_and_readout`'s `rec.pop('prot', None)`); `code/m2_dose_response.py` (`--alpha` grid);
  `results/m2_dose_response_curve.json` (per-dose table).
- Verdict reason: a real sweep with both target and capability metrics logged at every dose is a
  genuine strength, but the sweep neither spans the catalogue's recommended order-of-magnitude range
  nor uses σ_proj units, no random-direction control exists within M2's own scope, the only
  capability-preserving "clean" operating point (α=8) sits at the edge (not middle) of its plateau,
  and — most importantly — no raw generated-sequence text survives for spot-checking whether the
  high-α "helix-rich" readout reflects genuine diverse sequences or degenerate/repetitive artifacts
  that a structure predictor might over-confidently fold. Not a FAIL (no single-hardcoded-α, no
  capability blindness, no asymmetric-sign flattening), but a strict WARN.

### B–F. Reserved (not_implemented)
Status: not yet implemented.

## Action Items
- Extend the α grid to a wider order-of-magnitude range (or express α in σ_proj units) so the
  reported plateau/optimum is not confounded with an untested top end.
- Preserve a sample of raw generated sequences per dose (this is the single highest-value fix — see
  the paired EXPERIMENT_AUDIT.md's identical recommendation).
- Run a random/matched-control arm within M2's own α grid (not only cross-referenced from M3) at the
  doses actually used for the headline claim, including α=32 if that remains the reported optimum.
- If α*=32 remains the headline "optimum," explicitly reconcile this with the capability-tolerance
  violation observed at α=16, or adopt α=8 as the reported/locked operating point instead.
