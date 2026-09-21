# Captured-Behavior Report

**Direction**: Generate DNA sequences with higher α-helical content using Evo2-7B (task.md is the authoritative direction source)
**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: task.md (faithful capture)
**Date**: 2026-08-19
**Pipeline**: research-lit → faithful behavior capture → research-refine-pipeline

## Executive Summary
The given behavior — Evo2-7B can generate DNA whose encoded protein has higher α-helical content — is captured as one primary mechanism claim plus one supporting evaluation claim. Because `mechanism: discovery`, the plan is shaped by `/mechanism-explore` toward a **Location → Causal Intervention → Tuning & Editing** strategy: locate an internal component/direction in Evo2-7B that carries α-helical propensity, causally confirm it via a steering dose-response with specificity controls, then use it to bias generation toward higher α-helix content. The concrete mechanism family (SAE-feature clamp on the public Layer-26 SAE vs contrastive steering vector vs linear-probe direction) is left to the experiment-stage `/mechanism-skills` routing. First runs: build the α-helix evaluation harness, then extract candidate directions.

## Literature Landscape
See `idea-stage/LANDSCAPE.md` (full) and `idea-stage/RESEARCH_LIT.md` (raw dump). Highlights used as context (never to alter the claim):
- **Evo 2** (`arcinstitute/evo2_7b`) is an autoregressive StripedHyena-2 DNA model with standard temperature/top-k/top-p sampling; inference-time guidance of its generation is established.
- **Goodfire "Interpreting Evo 2"** located **α-helix / β-sheet SAE features on Layer 26** (validated vs AlphaFold3; public artifact `Goodfire/Evo-2-Layer-26-Mixed`), but reports causal **steering of Evo 2 as underdeveloped** ("further research needed") — the mechanism-discovery opening.
- **ARCADE** and **Steering Protein Language Models** establish the training-free contrastive-steering-vector recipe for this model class.
- α-helix content of generated DNA is evaluated by ORF→translate→secondary-structure-predict→fraction-helix (%H), with **ESMFold+DSSP** as structure-grounded confirmation.

## Claims to Verify

### Claim 1: Causally steerable internal α-helix control in Evo2-7B (PRIMARY)

**Original (verbatim excerpt from task.md):**
> Generate DNA sequences with higher α-helical content using Evo2-7B.
> (Task title: "Generate DNA sequences with higher α-helical content using Evo2-7B"; Task Overview: "Generate DNA sequences with high α-helical content using Evo2-7B.")

**Extracted statement**: There exists a localizable internal component of Evo2-7B (a feature / activation-direction / subspace at one or more layers) such that intervening on it during autoregressive generation causally **increases the α-helical content** (fraction of residues assigned α-helix, %H, in the translated protein of the generated coding sequence) relative to an unsteered Evo2-7B baseline, and the increase is **specific** (achieved without collapsing sequence validity / coding-ness and without being a trivial by-product of decoding temperature or a random direction).
**Hypothesis**: H1 — a low-dimensional, causally-effective "α-helical propensity" direction is represented in Evo2-7B's activations; adding/clamping it during generation raises the %H of generated sequences monotonically over a working dose range.
**Measurable predicate**: mean %H of proteins translated from Evo2-7B sequences generated **with** the intervention exceeds mean %H of the **unsteered baseline** by a statistically significant margin (one-sided, corrected, n ≥ ~100 generations per condition), with a **monotonic dose-response** in the steering coefficient and a **specificity check** (matched random/control direction yields no significant %H gain; ORF validity and coding-likelihood preserved; β-sheet %E not the sole driver).
**Expected direction**: up (higher α-helical content under intervention).
**Resources (preferred; model pinned by HARD constraint)**: model: **Evo2-7B** (`arcinstitute/evo2_7b`) — binding, no substitution; candidate direction source: public `Goodfire/Evo-2-Layer-26-Mixed` SAE and/or contrastive activations; dataset: coding sequences labeled by secondary-structure class (all-α vs all-β / low-helix), sourced from PDB/DSSP-annotated proteins mapped to their CDS (e.g. CATH/SCOP structural classes); used_n: ≥ ~500 contrastive coding windows for direction extraction, ≥ ~100 generations/condition for evaluation (resolve exact counts in Phase 4.5 / at mechanism routing).
**Status**: pending verification
**Notes**: This is the task's single behavior framed as a mechanism claim under `mechanism: discovery`. Altitude kept at the *kind* of component (a direction/feature at some layer), not a specific layer — the exact locus (e.g. Layer 26) is what the Location work discovers. No M0 gate (behavior-source: given).

### Claim 2: α-helical-content evaluation is faithful (SUPPORTING)

**Original (verbatim excerpt from task.md):**
> Generate DNA sequences with higher α-helical content using Evo2-7B.
> (Implied by "α-helical content" being the measured target of the task.)

**Extracted statement**: α-helical content of a generated DNA sequence can be measured reproducibly by extracting its ORF, translating to protein, and predicting per-residue secondary structure, yielding %H that **agrees with a structure-based reference** (ESMFold/AlphaFold2 → DSSP) on held-out sequences.
**Hypothesis**: H2 — the sequence-based %H metric used as the online steering objective correlates strongly with the structure-based DSSP %H (so gains reported under Claim 1 reflect real helical content, not metric artifact).
**Measurable predicate**: on a held-out set of translated proteins, correlation between fast sequence-based %H and ESMFold+DSSP %H is high (e.g. Pearson r ≥ 0.7; exact threshold set in Phase 4.5), and the ORF-finding/translation pipeline recovers the intended reading frame on validation cases.
**Expected direction**: threshold (agreement above a set bar).
**Resources**: sequence-based SS predictor (e.g. NetSurfP-3.0 / S4PRED); ESMFold + DSSP for the confirmation subset; validation proteins with known DSSP labels.
**Notes**: Supporting/evaluation claim — makes the Claim-1 metric trustworthy. Derived strictly from task.md's "α-helical content" being the measured quantity; adds no new scientific assertion.

## Refined Proposal
- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering both claims)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged with the claim(s) each verifies; mechanism_strategy stamped; no M0 — behavior-source: given)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Next Steps
- [ ] /mechanism-skills to route the testing approach to a concrete mechanism family + submethod (Workflow 1.25) — candidates: SAE-feature clamping (Layer-26 SAE), contrastive steering vector, linear-probe direction
- [ ] /auto-experiment to implement and run the verification suite (Workflow 1.5)
- [ ] /auto-verify to stress-test each verified claim under method/dataset/model swaps (Workflow 1.75)
- [ ] /auto-iteration-loop to iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke /auto for the autonomous claim → routing → experiments → verify → review chain
