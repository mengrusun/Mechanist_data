# Reviewer Memory

Persistent, append-only reviewer suspicion log across iterations of `/auto-iteration-loop`.
Prepended to every Phase A reviewer prompt from iteration 2 onward so the reviewer can
check whether prior suspicions were genuinely addressed or sidestepped.

## Iteration 1 — Score: 6.5/10, Verdict: almost

- **New suspicions**:
  1. **Predictor-mediated causality.** All causal outcomes are read through ESMFold/OmegaFold — the steering direction could exploit sequence statistics the predictors are sensitive to rather than reflect a true internal causal variable for protein helix propensity. This is the single biggest scientific vulnerability; two predictors help but do not eliminate it. Would want additional orthogonal readouts (secondary-structure propensity predictors beyond ESMFold/OmegaFold; simple sequence-level helix propensity; MD or wet-lab if feasible).
  2. **C1 train/test leakage risk not audited to reviewer taste.** Need explicit documentation of PDB redundancy filtering, chain-level and sequence-identity homology-family splits, codon-to-residue mapping details, and whether homologs can cross splits. The current AUROC could be modestly inflated if homology-family splits were not enforced.
  3. **Capability preservation is thin.** valid-ORF ≥ 0.95×baseline alone does not establish biological plausibility. Should also report: Evo2 LM likelihood/perplexity, amino-acid composition drift, repeat/low-complexity content, length/truncation, codon usage drift, disorder predictions.
  4. **C3 finite-null resolution.** With 48 nulls, empirical one-sided p is floored at ~1/(48+1)=0.0204 — the z-score (5.3–5.8) uses null mean/variance and is much stronger than the empirical p suggests. Paper must explain this cleanly. If affordable, increase null count or add principled null families (matched sparse-feature sets, same-cardinality non-helix-enriched sets, activation-matched controls).
  5. **Matched-control arm construction must be audit-proof.** Any ambiguity in what statistics were matched invites cherry-picking accusations.
  6. **β-arm negative must not drift.** Do NOT slip back into "secondary-structure axis" language — the supported claim is helix-axis specificity, NOT symmetric helix-vs-sheet disentanglement.
  7. **Mechanistic depth is limited.** Where in sequence do the 19 features fire? What codon/amino-acid motifs does steering enrich? Does steering shift amino-acid composition toward helix-favoring residues? Is the effect mediated by residue identity, periodicity, hydrophobic patterns, signal peptides, transmembrane helices, coiled-coils?
  8. **Trivial biological confound not fully excluded.** Rule out that S is steering toward membrane proteins / signal peptides / coiled-coils / low-complexity helical repeats.
  9. **H-only per-sample storage omission is a minor but real sloppiness signal.** Fix in future runs.
  10. **Title/abstract language must stay narrow.** "Causally manipulable knob" is defensible; "mechanism of α-helix formation" or anything implying biological ground-truth control is not yet defensible.

- **Previous suspicions addressed?**: n/a (first iteration)

- **Unresolved (carried forward)**: all 10 of the above. These are paper-narrative and future-work items; the current work does not need to close them to be publishable-at-a-strong-venue, but they must be either (a) explicitly caveated in the paper text or (b) resolved in future rounds. None warrant a back-edge in this iteration.

- **Patterns**: The dominant systemic pattern is **predictor-mediated causality** — every downstream claim (C2 dose-response, C3 specificity) ultimately depends on a computational structure predictor's response to steered sequences. Round 2 hardened this axis in the strongest way currently feasible (dual predictors, pLDDT-weighted endpoint, gate-sweep sensitivity check, pLDDT-confound explicitly ruled out) but the axis remains the primary reviewer attack surface for a top ML venue submission.
