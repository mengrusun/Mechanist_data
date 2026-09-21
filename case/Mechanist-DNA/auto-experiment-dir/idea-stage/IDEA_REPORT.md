# Captured-Behavior Report — Round 2 (Rigor Hardening)

**Direction**: (empty — behavior/claims sourced authoritatively from `task.md`; supplementary scope only)
**Behavior-source**: given-validation
**Mechanism**: given (SAE feature steering / feature amplification)
**Claim source**: task.md (faithful capture)
**Date**: 2026-07-19
**Pipeline**: research-lit → faithful behavior capture → research-refine-pipeline

## Executive Summary

`task.md` (round 2) pins the **same** two-part behavior established in round 1 and asks to make its causal claim **publication-solid**, not to change it: (a) within Evo2-7B's pre-trained Layer-26 SAE a set of features *selectively responds* to protein α-helix, and (b) *amplifying* those features during autoregressive DNA generation *causally raises* the α-helical content of the encoded protein, increasing with amplification strength up to an optimum — proving the feature is a causally manipulable knob. Because this run is `given-validation`, the plan opens with a hard **M0 phenomenon-validation gate** that re-confirms the existence/selectivity half (C1) with the mandated natural-CDS + real-DSSP-label methodology — **reusing the frozen round-1 feature set S** (`rounds/round_1/results/m0_feature_set.json`, 19 helix features) as the starting point but still emitting the four-state M0 verdict — before the causal milestones (C2 dose-response, C3 specificity) run. The scientific content of C1/C2/C3 is unchanged from round 1; the round-2 work is entirely in the **testing method**: pLDDT folded into the statistics (not just a gate), a second independent structure predictor, σ_proj-unit dosing with a mapped interior plateau, tighter CIs on the modest C3 effect, an explicit β-sheet-arm resolution, and swap-robustness authored for all three claims. All claims are pinned to **Evo2-7B + its Layer-26 SAE** (model fidelity is non-negotiable).

## Why this is a legitimate hardening retry (not concluded work)

Per `research_memory.json`, this (behavior, mechanism) attempt is **NOT settled**: round 1 deferred the C1 and C3 swap-tests (the 1-claim `MAX_VERIFY_CLAIMS` cap left them audit-only), left C3 with a mechanism-audit residual WARN (α not in σ_proj units; α=8 at the capability edge), and never tested the structure-predictor robustness axis for C2. Round 2 honors the pin (same behavior + mechanism) and closes exactly these open items. This is hardening, not re-doing concluded work.

## Resources (pinned by task.md)

- **Model (pinned — model-fidelity HARD CONSTRAINT, do not substitute):** Evo2-7B (StripedHyena2, 7B) at `/data1/share_model/evo2/evo2_7b/evo2_7b.pt`.
- **SAE (pinned — do not substitute):** pre-trained Layer-26 mixed prokaryote/eukaryote **BatchTopK** SAE, expansion 8, k=64, ~32,768 features, at `/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt`, site `blocks.26.post_norm`.
- **Feature-discovery data (hard rule):** natural **coding sequences (CDS)** from databases + **real secondary-structure (DSSP) labels** from experimental protein structures, attached to codons after proper **CDS↔protein alignment**. **Reverse-translation of proteins into DNA is forbidden.** (M0 may reuse the frozen round-1 feature set + data caches under `data/`.)
- **Structural verification tooling:** **two independent** structure predictors — **ESMFold** (round-1 default) **plus a second, independent predictor** (OmegaFold or ColabFold/AlphaFold2, pre-staged in setup) — each followed by **DSSP** (mkdssp primary, pydssp swap) for α-helix-fraction scoring, with **pLDDT** used both as a gate and as a statistical covariate/weight.
- **Compute:** 8×A800-80GB machine; **≤4 GPUs simultaneously**, pinned to **GPUs 3,4,5,6** (0,2 busy with other users). `max_parallel: 4`.
- **Resource-fidelity marker:** NOT `strict` (this is given-validation + given, not the given+given reproduction combo). Model & SAE are still pinned by the model-fidelity constraint; data sizes are cost-aware but planned at **full scale** because the 4-GPU budget is generous and this is an explicit rigor round (no downscaling).

## Claims to Verify

Each claim below is authored to be **swap-testable** (a clear method / dataset / model axis the verify stage can swap under `MAX_VERIFY_CLAIMS=3`); the swap axes are listed per claim.

### Claim 1 (C1): α-helix-selective SAE feature set exists  *(← M0-validated)*

**Original (verbatim excerpt from task.md):**
> "Within Evo2-7B's pre-trained Layer-26 SAE, a set of features selectively responds to protein α-helix … natural coding sequences (CDS) from databases + real DSSP labels from experimental PDB structures after CDS↔protein alignment. Reverse-translation of proteins into DNA is FORBIDDEN." (round-2 task.md, "Behavior to investigate" + "Feature-discovery data rule")

**Extracted statement**: In Evo2-7B's Layer-26 SAE there exists a non-empty set of features S whose activation is selectively elevated on codons that (via CDS↔protein alignment) correspond to α-helix residues, relative to non-helix codons and to matched controls, quantified by a per-codon selectivity metric (set-level AUROC / F1) that survives confound control (length, GC, position, coding, label identity) and multiple-testing/FDR control on a held-out CDS/DSSP test split.
**Hypothesis**: H1 — one or more Layer-26 SAE features (set-level) are α-helix-selective detectors, not artifacts of codon frequency / GC / position / length.
**Measurable predicate**: ∃ feature set S s.t. its set-level α-helix-vs-non-helix discrimination (test AUROC ≥ τ, with margin over β-sheet/coil) survives confound + FDR control on a homolog-disjoint held-out split, replicating across organism groups and split seeds.
**Expected direction**: threshold (selectivity metric above chance/controls).
**Status**: pending verification — **this is the M0 gate** (`kind: phenomenon-validation`). Reuses frozen round-1 S as the starting point; still emits the four-state verdict.
**Swap axes (for verify)**: (i) DSSP assignment algorithm (mkdssp ↔ pydssp); (ii) organism dataset (prokaryote ↔ eukaryote CDS/DSSP); (iii) selectivity metric (AUROC ↔ F1 ↔ balanced-accuracy) and helix definition (HGI ↔ H-only).
**Notes**: Round-1 verdict was `established` set-level (set test-AUROC 0.86–0.89), but "no single feature clears AUROC 0.75" → C1 is a **set-level** claim, not a single-detector claim; the phrasing above preserves that. Its four-state M0 verdict governs whether C2/C3 run.

### Claim 2 (C2): Feature amplification causally raises α-helical content (dose-response)

**Original (verbatim excerpt from task.md):**
> "amplifying those features during autoregressive DNA generation causally raises the α-helix content of the encoded protein, increasing with amplification strength up to an optimum — a causally manipulable knob." (round-2 task.md, "Behavior to investigate")

**Extracted statement**: Amplifying the C1 feature set S at the Layer-26 SAE site during Evo2-7B autoregressive DNA generation increases the α-helix fraction (DSSP on the predicted structure of the translated protein) of the generated sequences, and this α-helix fraction increases with amplification strength across a swept coefficient range (monotone dose-response), with an identifiable **interior optimum** α\* beyond which sequence quality degrades — and the effect holds under **two independent structure predictors** and under **pLDDT-weighted** as well as hard-gated readouts.
**Hypothesis**: H2 — the steering coefficient α (in σ_proj units) has a positive, monotone (up to an interior optimum) causal effect on encoded-protein α-helix fraction, not confounded by pLDDT shifting with dose.
**Measurable predicate**: mean α-helix fraction is monotonically non-decreasing in α over the swept range (positive Spearman trend, significant vs α=0) up to an interior α\*; the rise survives (a) a second structure predictor, (b) pLDDT-weighting and pLDDT-conditioning, and (c) a pLDDT-threshold sensitivity sweep.
**Expected direction**: up (α-helix fraction rises with amplification strength) to an interior maximum.
**Status**: pending verification (`depends_on: [M0]`).
**Swap axes (for verify)**: (i) structure predictor (ESMFold ↔ OmegaFold/ColabFold); (ii) DSSP assignment (mkdssp ↔ pydssp); (iii) trend statistic / helix definition; (iv) generation seed set.
**Notes**: Core causal claim. Round-2 tightenings: σ_proj-unit dose grid refined so α\* is an **interior** maximum; pLDDT folded into stats; second predictor closes the round-1 open item.

### Claim 3 (C3): The effect is specific — the feature is a causally manipulable knob

**Original (verbatim excerpt from task.md):**
> "thereby proving that the feature is a causally manipulable knob" + round-2 §Required-improvements 5–6 (keep the ≥30-direction norm-matched random-null as the PRIMARY specificity statistic; resolve the β-arm as a stronger β set or as documented helix-axis specificity).

**Extracted statement**: The amplification-induced rise in α-helix content at the capability-preserved interior dose is **specific** to the C1 α-helix feature set S — a norm-matched **random-direction null** (≥30 independent directions) and a matched-control feature steered at the same σ_proj-unit dose do **not** reproduce the α-helix increase — establishing S as a causally manipulable, specific knob for α-helical content. The β-sheet arm is resolved either as a symmetric double dissociation (if a stronger β-selective set raises sheet) or explicitly as **helix-axis specificity** (documented β-negative).
**Hypothesis**: H3 — the dose-response effect is attributable to the α-helix feature identity (specificity), not to generic activation perturbation; the effect size (with CIs) exceeds the norm-matched random-null distribution.
**Measurable predicate**: at the locked capability-preserved interior dose (σ_proj units), S's α-helix gain is a significant outlier vs a ≥30-direction norm-matched random-null (empirical one-sided p<0.05, reported with effect size + CI) AND ≫ matched-control; generation-quality metrics (valid-ORF, perplexity, pLDDT) stay within tolerance across arms; result stable across both predictors and across the pLDDT-threshold sweep.
**Expected direction**: up for S (target) vs ≈equal for random-null/matched controls (specificity contrast).
**Status**: pending verification (`depends_on: [M0]`).
**Swap axes (for verify)**: (i) structure predictor (ESMFold ↔ OmegaFold/ColabFold); (ii) random-null construction (feature-set draw ↔ raw-direction draw, norm-match convention); (iii) DSSP assignment; (iv) locked-dose choice within the plateau.
**Notes**: Round-2 tightenings: ≥30-direction norm-matched random-null kept as the **PRIMARY** statistic with narrower CIs (more samples/seeds per dose); dose locked at a **demonstrated interior plateau** in σ_proj units (not the grid edge); pLDDT-stratified and second-predictor confirmation; explicit β-arm resolution. This is the "causally manipulable knob" assertion made rigorous.

## Claims → milestones (see EXPERIMENT_PLAN.md)
- **C1** → **M0** (phenomenon-validation gate): natural-CDS + DSSP feature-selectivity re-confirmation (reuses frozen S) + β-set re-identification attempt.
- **C2** → **M1** (harness: σ_proj calibration + dual-predictor + pLDDT-weighted readout) + **M2** (σ_proj-unit dose-response sweep, dual predictor, pLDDT-in-stats).
- **C3** → **M3** (specificity at the locked interior plateau: ≥30-direction norm-matched null primary, dual predictor, pLDDT-stratified, β-arm resolution).

## Literature Landscape

*(Populated in Phase 5 from `idea-stage/LANDSCAPE.md` — pLDDT-confidence-aware evaluation and multi-predictor robustness for generated-protein secondary-structure claims.)*

## Refined Proposal
- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering all three claims + the six required improvements)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies; opens with M0; stamps `chosen_mechanism`; `max_parallel: 4` on GPUs 3,4,5,6)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps
- [ ] /auto-experiment to implement + run the verification suite (mechanism family already committed: `chosen_mechanism` = SAE feature steering / amplification; no routing).
- [ ] /auto-verify to stress-test C1, C2, C3 under method/dataset/model swaps (`MAX_VERIFY_CLAIMS=3`).
- [ ] /auto-iteration-loop to iterate until reviewer-ready.
- [ ] Or invoke /auto for the autonomous claim → experiments → verify → review chain.
