# Captured-Behavior Report

**Direction**: (empty — behavior/claims sourced authoritatively from `task.md`)
**Behavior-source**: given-validation
**Mechanism**: given (SAE feature steering / feature amplification)
**Claim source**: task.md (faithful capture)
**Date**: 2026-07-18
**Pipeline**: research-lit → faithful behavior capture → research-refine-pipeline

## Executive Summary

`task.md` asserts a two-part behavior about Evo2-7B's pre-trained Layer-26 SAE: (a) a set of SAE features *selectively responds* to protein α-helix, and (b) *amplifying* those features during autoregressive DNA generation *causally raises* the α-helical content of the encoded protein, increasing with amplification strength up to an optimum — proving the feature is a causally manipulable knob. Because the run is `given-validation`, the plan opens with a hard **M0 phenomenon-validation gate** that first establishes the *existence/selectivity* half (C1) with the mandated natural-CDS + real-DSSP-label methodology, and only then runs the causal steering milestones (C2 dose-response, C3 specificity). All claims are pinned to **Evo2-7B + its Layer-26 SAE** (model fidelity is non-negotiable).

## Resources (pinned by task.md)

- **Model (pinned — model fidelity HARD CONSTRAINT, do not substitute):** Evo2-7B (StripedHyena2, 7B) at `/data1/share_model/evo2/evo2_7b/evo2_7b.pt`.
- **SAE (pinned — do not substitute):** pre-trained Layer-26 mixed prokaryote/eukaryote **BatchTopK** SAE, expansion factor 8, k=64, at `/data1/share_model/evo2/evo2_sae_layer26_mixed/sae-layer26-mixed-expansion_8-k_64.pt`. (Evo2-7B hidden dim 4096 → SAE dictionary ≈ 32,768 features.)
- **Feature-discovery data (hard rule):** natural **coding sequences (CDS)** from databases + **real secondary-structure (DSSP) labels** from experimental protein structures, attached to codons after proper **CDS↔protein alignment**. **Reverse-translation of proteins into DNA is forbidden.**
- **Structural verification tooling:** protein structure prediction (ESMFold / AlphaFold — pick per papers/tooling availability) + **DSSP** for α-helix-fraction scoring of generated sequences.
- **Compute:** 8×A800-80GB machine; **≤4 GPUs simultaneously**, pinned to GPUs 2,3,4,5 (0,1 busy). `max_parallel: 4`.
- **Resource-fidelity marker:** NOT `strict` (this is given-validation + given, not the given+given reproduction combo). Model & SAE are still pinned by the model-fidelity constraint; data sizes are cost-aware but planned at full scale because the 4-GPU budget is generous.

## Claims to Verify

### Claim 1 (C1): α-helix-selective SAE feature set exists  *(← M0-validated)*

**Original (verbatim excerpt from task.md):**
> "Within the pre-trained SAE of Evo2-7B, first confirm the existence of a set of features that selectively respond to 'α-helix' … When finding the set of features that selectively respond to 'α-helix', Do NOT generate artificial DNA sequences by reverse-translating protein sequences … Instead, directly obtain natural coding sequences (CDS) from databases and annotate their codons with real secondary structure labels from experimental protein structures after proper sequence alignment."

**Extracted statement**: In Evo2-7B's Layer-26 SAE, there exists a non-empty set of features whose activation is selectively elevated on codons that (via CDS↔protein alignment) correspond to α-helix residues, relative to non-helix codons and to matched controls, quantified by a per-codon selectivity metric (precision/recall/F1 or AUROC) with confound control and multiple-testing/FDR control.
**Hypothesis**: H1 — one or more Layer-26 SAE features are α-helix-selective detectors, not artifacts of codon frequency / GC / position / length.
**Measurable predicate**: ∃ feature set S s.t. each f∈S has α-helix-vs-non-helix discrimination (e.g., F1 ≥ threshold or AUROC ≫ 0.5) that survives confound controls and FDR control on a held-out CDS/DSSP test split.
**Expected direction**: threshold (selectivity metric above chance/controls).
**Status**: pending verification — **this is the M0 gate** (`kind: phenomenon-validation`).
**Notes**: Upstream half of the claim chain. Its four-state M0 verdict governs whether C2/C3 run. Data methodology fixed by the reverse-translation prohibition.

### Claim 2 (C2): Feature amplification causally raises α-helical content (dose-response)

**Original (verbatim excerpt from task.md):**
> "then autoregressively generate DNA by amplifying those features, determine the optimal amplification strength, and use protein structure prediction tools to verify whether the α-helical content of the protein encoded by the generated sequences increases with the amplification strength"

**Extracted statement**: Amplifying the C1 feature set at the Layer-26 SAE site during Evo2-7B autoregressive DNA generation increases the α-helix fraction (DSSP on the predicted structure of the translated protein) of the generated sequences, and this α-helix fraction increases with amplification strength across a swept coefficient range (monotone dose-response), with an identifiable optimal strength beyond which sequence quality degrades.
**Hypothesis**: H2 — steering coefficient α (amplification strength) has a positive, monotone (up to an optimum) causal effect on encoded-protein α-helix fraction.
**Measurable predicate**: mean α-helix fraction is monotonically non-decreasing in the amplification coefficient over the swept range (positive Spearman trend, significant vs α=0 baseline) up to an optimum; optimal α reported.
**Expected direction**: up (α-helix fraction rises with amplification strength).
**Status**: pending verification (`depends_on: [M0]`).
**Notes**: Core causal claim. Requires ORF/validity filtering + structure-prediction confidence gating + sufficient samples per dose for statistical power.

### Claim 3 (C3): The effect is specific — the feature is a causally manipulable knob

**Original (verbatim excerpt from task.md):**
> "thereby proving that the feature is a causally manipulable knob."

**Extracted statement**: The amplification-induced rise in α-helix content is specific to the C1 α-helix feature set — matched random/unrelated features and an off-target (β-sheet) feature steered at the same coefficients do not reproduce the α-helix increase (and the β-sheet feature instead raises β-sheet content), while generation quality does not collapse — establishing the α-helix feature set as a causally manipulable, specific knob for α-helical content.
**Hypothesis**: H3 — the dose-response effect is attributable to the α-helix feature identity (specificity), not to generic activation perturbation.
**Measurable predicate**: α-helix-fraction increase for the α-helix feature set is significantly larger than for matched-control/off-target features at matched coefficients; a β-sheet off-target raises β-sheet not α-helix (double dissociation); generation-quality metrics (coding fraction, valid-ORF rate, perplexity) stay within tolerance.
**Expected direction**: up for α-helix feature (target) vs ≈equal for controls (specificity contrast).
**Status**: pending verification (`depends_on: [M0]`).
**Notes**: Turns C2's correlation-with-dose into a controlled causal-specificity claim; this is the "causally manipulable knob" assertion made rigorous.

## Claims → milestones (see EXPERIMENT_PLAN.md)
- C1 → **M0** (phenomenon-validation gate): natural-CDS + DSSP feature-selectivity identification.
- C2 → **M2** (dose-response steering sweep) + M1 (generation/verification harness).
- C3 → **M3** (specificity / matched-control + off-target double-dissociation).

## Refined Proposal
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps
- [ ] /auto-experiment to implement + run the verification suite (mechanism family already committed: `chosen_mechanism` = SAE feature steering / amplification; no routing).
- [ ] /auto-verify to stress-test each verified claim under method/dataset/model swaps.
- [ ] /auto-iteration-loop to iterate until reviewer-ready.
- [ ] Or invoke /auto for the autonomous claim → experiments → verify → review chain.
