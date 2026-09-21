# Auto Iteration Final Report — Hardening the α-Helix Feature-Steering Knob in Evo2-7B (Round 2)

- **Generated**: 2026-07-20
- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: **6.5 / 10** (top ML mech-interp track); **7 / 10** (strong genomics/protein-methods venue with narrow framing)
- **Final canonical verdict**: **almost**
- **Termination reason**: `positive_verdict` — three-dimensional STOP rule satisfied on iteration 1 (score ≥ 6, verdict ∈ {ready, almost}, zero claims in FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS)
- **Cumulative cost**: `runs_total = 0`, `gpu_hours_total = 0.0` — zero back-edges consumed; every reviewer concern is a paper-narrative or future-work item, not a script/plan/claim fix.
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)
- **Reviewer LLM**: `gpt-5.4` via `https://www.dmxapi.cn/v1` (resolved from shell env)

---

## Executive Summary

Round 2 delivered all six planned rigor upgrades over round 1 (pLDDT-weighted primary endpoint, OmegaFold as an independent second structure predictor, σ_proj-unit dosing with a demonstrated interior optimum c*=21.48, 48-direction norm-matched random null with cluster-robust CIs, swap-robustness on all three claims, β-arm resolved as documented negative → helix-axis specificity framing). All three claims — C1 existence, C2 dose-response, C3 specificity — are experiment-supported and verify-PASS at robustness 1.00 with clean integrity at both Phase-2 (main-experiment audit) and Phase-9 (variant audit). The external reviewer scored the work 6.5/10 for a top ML mech-interp venue and 7/10 for a strong genomics/protein-methods venue with narrow framing, calling it "much better than round 1... now looks like a real paper rather than an intriguing but under-hardened claim" and issuing a canonical "almost" verdict. The three-dimensional STOP rule fired on iteration 1 with **zero back-edges consumed**: every remaining reviewer concern is paper-narrative discipline (predictor-mediated causality caveat, C1 label-provenance documentation, capability-preservation breadth, C3 empirical-p vs z-score explanation, matched-control transparency, β-arm framing discipline) or future-work mechanistic depth — none warranted a variant-integrity fix (①), a main-experiment repair (②), or a claim rewrite (③). Ten reviewer suspicions are recorded in `REVIEWER_MEMORY.md` for future rounds.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 3 | 3 PASS (held) — C1, C2, C3 |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY           | 0 | — |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C1_helix_feature_set_exists` — an α-helix-selective feature set exists (M0 existence gate)
- **Original robustness signal**: robustness = 1.00, variants passed = 1/1 (DSSP helix-def HGI → H-only swap; prokaryote H-only AUROC 0.906 ± 0.002 vs HGI 0.897; eukaryote H-only 0.868 vs HGI 0.865)
- **Reviewer consistency check**: numerically coherent (prok HGI 0.897 / H-only 0.906; euk HGI 0.865 / H-only 0.868; confound-only 0.506; shuffle-null 0.53); confound gap +0.39 is large; H-only swap not degrading is a good robustness sign; distributed set-level encoding (no single feature clears 0.75) is mechanistically plausible and actually more interesting than one hero feature; "cleanest of the three claims".
- **Touched in iterations**: [1] (brief consistency check only; no action)
- **Final status**: **PASS (held)**
- **Notes for downstream**:
  - AUROC is an encoding/existence signal, not causality — C1 must be written as an **existence claim**, not a mechanism claim.
  - Label-provenance details (PDB selection criteria, redundancy filtering, homology-family split policy, codon-to-residue mapping, unresolved-residue handling) must be documented explicitly in the paper's supplementary — currently in code only. Reviewer flagged this as the single largest documentation risk for C1.
  - Cross-organism (E. coli + H. sapiens) is meaningful but not broad enough to imply universality.

### 1.2 `C2_dose_response_helix` — amplification raises α-helix fraction (dose-response)
- **Original robustness signal**: robustness = 1.00, variants passed = 1/1 (H-only swap; ESMFold ρ=0.867 p=0.0012, Δ=+0.144; OmegaFold ρ=0.903 p=0.0003, Δ=+0.147 — Δ slightly larger than under HGI)
- **Reviewer consistency check**: "the claim that improved the most". σ_proj-unit dosing + held-out trend + capability-preserved interior optimum "is exactly what the paper needed"; corrected c*-selection rule (contiguous-preserved-region argmax, transparent correction after too-lenient tolerance took a global argmax at c=85.92) is the right one and actually strengthens the paper. ESMFold dose curve shape (0.454 → 0.593 at c*=21.48 through capability-preserved region, then degradation at c≥32) matches the expected "real effect + out-of-regime collapse at large steering" pattern. Dual-predictor agreement lowers artifact-of-one-fold-model risk. H-only slightly stronger is reassuring.
- **Touched in iterations**: [1] (brief consistency check only; no action)
- **Final status**: **PASS (held)**
- **Notes for downstream**:
  - Monotonicity language in the paper must be scoped to the pre-specified capability-preserved segment; the full curve is non-monotone under degradation. Do not claim monotonicity over the entire dose axis.
  - Capability preservation metric (valid-ORF ≥ 0.95×baseline) is weakly specified for biological plausibility — add composition drift, LM perplexity, disorder prediction, length distribution as supplementary quality diagnostics. Deferred to future work.
  - Selection-on-one-seed / evaluate-on-two is thin but acceptable; reviewer will not kill the paper over it.
  - Structure-predictor dependence remains the central paper-narrative vulnerability. Two predictors is the strongest robustness currently feasible without wet-lab or MD.

### 1.3 `C3_specificity_causal_knob` — the rise is specific to S (helix-axis specificity)
- **Original robustness signal**: robustness = 1.00, variants passed = 1/1 (H-only swap; 0/48 null ≥ S under both predictors, ESMFold z=5.83, OmegaFold z=6.13 — specificity strengthens under the swap; empirical p=0.020 unchanged; caveat: per-sample H-only helix_h_w not stored → no cluster-bootstrap CI for H-only variant, empirical p+z remain valid)
- **Reviewer consistency check**: "strongest hardening component". 48-direction norm-matched null + matched control + dual predictors + 0/48 null ≥ S under both is "exactly the right response to skepticism". Numeric consistency clean (ESMFold Δ=+0.129 [0.100, 0.159], matched -0.031, null mean=-0.004, null max=+0.056, p=0.020, z=5.30; OmegaFold Δ=+0.135, matched -0.025, null max=+0.059, z=5.80). pLDDT-confound check is directionally favorable (S mean pLDDT LOWER than baseline while helix rises) — rebuts the simplest confidence-artifact story. β-arm resolution is "honest and correct — do not try to rescue rhetorically".
- **Touched in iterations**: [1] (brief consistency check only; no action)
- **Final status**: **PASS (held)**
- **Notes for downstream**:
  - **Finite-null resolution**: with n_null = 48, empirical one-sided p is floored at ~1/(48+1) = 0.0204. The z-score (which uses null distribution mean & variance) is much stronger and reflects the true effect. Both must be reported together with a clean explanation of the difference.
  - **Matched-control construction must be transparent** in the paper writeup: exactly how the matched-control features were chosen and what statistics were matched. Any ambiguity invites cherry-picking accusations.
  - **β-arm negative** limits the conceptual scope: no clean axis-of-secondary-structure control story. The supported claim is a helix-promoting knob, NOT evidence of disentangled orthogonal helix vs sheet latent axes. Paper text must not slip back to "secondary-structure axis" language.
  - **H-only per-sample storage omission** is a minor but real credibility nick — fix in future runs so every swap endpoint has full bootstrapable sample-level outputs.
  - **ESMFold hard-gated pLDDT-threshold sweep returned null** (bin populations too thin at ESMFold's calibration); OmegaFold gate-sweep populated and stable (S 0.65–0.67 vs c0 0.52–0.55 across pLDDT 50–80); primary pLDDT-weighted endpoint unaffected.

---

## Section 2 — FAIL Claims (full journey)

**None.** No claim entered the iteration loop in FAIL state.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

**None.** No claim entered the iteration loop in INCONCLUSIVE state.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

**None.** No claim entered the iteration loop in ZERO_ELIGIBLE_VARIANTS state.

---

## Section 4b — INTEGRITY_ONLY Claims (no-action bucket)

**None.** All 3 admitted claims were also picked at Stage 2 (`MAX_VERIFY_CLAIMS = 3 = |ADMITTED|`); no claim was Stage-2-skipped. No INTEGRITY_ONLY entries.

---

## Section 5 — Legacy DEFERRED Claims

**None.** New verify runs never populate this bucket; `VERIFY_REPORT.md` carries no legacy deferred section.

---

## Section 6 — Cross-Cutting Patterns

Reviewer flagged one dominant systemic pattern:

- **Predictor-mediated causality.** Every downstream causal claim (C2 dose-response, C3 specificity) ultimately depends on a computational structure predictor's response to steered sequences. Round 2 hardened this axis in the strongest way currently feasible (dual predictors, pLDDT-weighted endpoint, gate-sweep sensitivity check, pLDDT-confound explicitly ruled out via S pLDDT < baseline) but the axis remains the primary reviewer attack surface for a top ML venue submission.
  - **Touches**: C2, C3
  - **Resolved at termination**: partially — dual predictors + pLDDT-weighted endpoint + pLDDT-confound ruled out is the currently-feasible mitigation; full resolution requires wet-lab / MD / orthogonal sequence-level helix-propensity metrics (Chou-Fasman, PROSS) — deferred to future rounds.
  - **Reviewer recommendation for the paper**: caveat this explicitly; keep title/abstract language narrow ("causally manipulable knob" defensible; "mechanism of α-helix formation" NOT yet defensible).

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: `runs_total = 0`
- **Iteration GPU-hours**: `gpu_hours_total = 0.0`

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | (no back-edge — three-dimensional STOP fired) | — | — | 0 | 0.0 | 6.5 | almost |

---

## Section 8 — Open Items for Human Reviewer

> Items the loop could not close. These need paper-writing discipline or future-round work — not a re-run of this pipeline.

- **Still-FAIL claims**: none
- **Still-INCONCLUSIVE claims**: none
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**: none
- **Legacy deferred claims**: none
- **Claim-reentry refusals** (③ requested but sub-budget exhausted): none

- **Recurring unresolved reviewer suspicions (paper-narrative + future-work)**:
  1. **Predictor-mediated causality** (C2, C3) — hardened via dual predictors but not eliminated; caveat in paper; future work: orthogonal sequence-level helix-propensity metrics (Chou-Fasman, PROSS) or MD-snapshot DSSP.
  2. **C1 label-provenance documentation** — PDB redundancy filtering, homology-family split policy, codon-to-residue mapping, unresolved-residue handling; currently in code only, must be documented in paper supplementary.
  3. **Capability preservation breadth** (C2) — valid-ORF alone is thin; future round should add LM perplexity, amino-acid composition drift, repeat/low-complexity content, length/truncation, codon usage drift, disorder-prediction.
  4. **C3 finite-null resolution** — n_null=48 floors empirical p at ~0.02; both p and z (5.30/5.80) must be reported together with a clean explanation. Future round can add principled null families (matched sparse-feature sets, same-cardinality non-helix-enriched sets, activation-matched controls).
  5. **C3 H-only variant lacks cluster-bootstrap CI** — per-sample H-only values not stored; empirical p+z remain valid; fix in future runs so every swap endpoint has full sample-level outputs.
  6. **C3 ESMFold hard-gated pLDDT-threshold sweep null** — bin populations too thin at ESMFold's calibration; OmegaFold gate-sweep populated and stable; primary pLDDT-weighted endpoint unaffected.
  7. **Matched-control arm construction transparency** (C3) — must be explicitly specified in paper writeup to preempt cherry-picking accusations.
  8. **β-arm framing discipline** (C3) — supported claim is helix-axis specificity, NOT symmetric helix-vs-sheet disentanglement. Paper text must not drift back to "secondary-structure axis" language.
  9. **Mechanistic depth** — where do the 19 features fire in sequence? Codon/amino-acid motifs enriched by steering? Composition shift toward helix-favoring residues? Effect mediated by residue identity, periodicity, hydrophobic patterns, signal peptides, transmembrane helices, coiled-coils? Rule out trivial biological confounds (membrane proteins, signal peptides, coiled-coils, low-complexity helical repeats). Deferred to future rounds under untried mechanism directions.
  10. **Title/abstract language discipline** — "causally manipulable knob" defensible; "mechanism of α-helix formation" or anything implying biological ground-truth control not yet defensible.

- **Recommended next-round options** (from `research_memory.json.behaviors[0].untried_mechanism_directions`, plus reviewer-driven additions):
  - **Tuning & Editing** direction on B1 — targeted fine-tuning to enhance/suppress the S features and observe secondary-structure shift.
  - **Formation Tracing** direction on B1 — layer sweep for where the α-helix code emerges in Evo2's stack.
  - **Unit Interpretation** direction on B1 — per-feature motif enrichment, codon patterns, hydrophobic periodicity, position-in-CDS bias.
  - **Decision Auditing** direction on B1 — attribution paths from S to the specific residues Evo2 emits when steered.
  - **Predictor-independence** (reviewer-driven) — non-fold-predictor helix-propensity metrics (Chou-Fasman / PROSS) or MD-snapshot DSSP to reduce the predictor-mediated-causality concern.
