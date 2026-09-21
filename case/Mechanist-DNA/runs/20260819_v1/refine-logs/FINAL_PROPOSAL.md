# Final Proposal — α-Helix-Directed DNA Generation by SAE Feature Amplification in Evo2-7B

**Date**: 2026-08-19
**Behavior-source**: given
**Mechanism**: given
**resource_fidelity**: strict
**chosen_mechanism**: SAE feature identification + amplification (Evo-2 Layer-26 Mixed SAE)
**mechanism_strategy**: n/a (MECHANISM=given — family fixed, no /mechanism-explore routing)
**Final verdict**: READY

## Problem Anchor (fixed — do not move)

Reproduce and quantify the behavior asserted in `task.md`: **Evo2-7B can be steered to generate DNA whose translated protein product has higher α-helical content, by amplifying SAE features that encode α-helix.** The mechanism is user-given: the released **Evo-2 Layer-26 Mixed SAE** (tied-weight TopK SAE, d_model=4096, d_sae=32768, expansion 8, k=64, hooked on the layer-26 residual stream). This is a reproduction/quantification of a capability the Evo2 paper (Nature 2026) demonstrated qualitatively — its SAEs recover α-helix/β-sheet/tRNA features and it frames features as usable for "steering of sequence generations."

## Method Thesis (one sentence)

Identify the SAE latents whose activation is *selective for α-helix residue positions* on labeled coding DNA, then amplify exactly those latents during Evo2's autoregressive decoding, and measure the resulting increase in predicted α-helical content of the generated sequences' translated ORFs, sweeping amplification strength to locate the optimum under a fixed sequence-validity constraint.

## Dominant Contribution

A **dose-controlled, quality-controlled** demonstration that a specific, interpretable SAE feature set causally raises a *structural phenotype* (α-helix) of a DNA language model's generations — the first quantitative α-helix steering on Evo2-7B via its own released SAE, with an explicit optimal-strength (α*) determination and specificity controls.

## The three fixed claims (verbatim from `idea-stage/IDEA_REPORT.md`)

- **C1** — a small, numerically-defined subset of the 32768 SAE latents is α-helix-selective on held-out labeled coding DNA (selectivity above a class/gene-preserving null and above matched β-sheet/coil controls).
- **C2** — amplifying that feature set during decoding raises mean **predicted** %-helix (ESMFold→DSSP "H" fraction of translated ORFs) of generated DNA vs. an identically-decoded unsteered baseline, on a treatment-independent quality-controlled set.
- **C3** — sweeping the amplification coefficient α yields a dose-response with an identifiable optimum α* (argmax %-helix subject to a preregistered validity floor), α* chosen on validation and confirmed on held-out seeds.

## Complexity intentionally rejected

- **No SAE retraining / no new SAE.** The given Layer-26 Mixed SAE is used as released (strict fidelity).
- **No model fine-tuning or weight editing.** Pure inference-time activation steering.
- **No learned/optimized steering vector** (e.g., gradient-over-features). The primary intervention is a fixed, interpretable residual-add of the selected features' decoder directions — the simplest mechanism that tests the claim. Gradient/optimized steering is explicitly out of scope.
- **No smaller model / no data subsetting to save cost** — forbidden by `resource_fidelity: strict`.

## Preregistered definitions (make every claim falsifiable)

These numeric criteria are frozen **before** confirmatory generation; locked code + split assignments archived first.

1. **Steering intervention (primary, one definition).** Residual add on the layer-26 residual stream at every decoding step: `x' = x + α · Σ_{i∈S} s_i · d̂_i`, where `S` = selected α-helix feature set, `d̂_i = W[:,i]/‖W[:,i]‖` the unit decoder direction of latent i (tied-weight SAE, decoder = W), and `s_i = median(z_i | z_i>0)` its median positive activation on **train** (so α is in units of typical feature activation and is portable). Sign positive = amplify. Secondary robustness variant (reported, not primary): pre-TopK activation add (add `α·s_i` to feature i's pre-activation, re-run TopK, add only the induced reconstruction delta `W(z'−z)`).
2. **α=0 identity.** The baseline runs through the identical steering code path at α=0 and is verified numerically equal to unsteered generation.
3. **Feature set size ("small subset").** Candidate latents = those with train α-helix AUROC ≥ 0.70, BH-FDR significant across all 32768 (q<0.01) under a **gene- and codon-position-preserving** shuffle null, and helix:control activation ratio exceeding matched β-sheet/coil controls. From candidates, take top-K by **validation** AUROC, K selected on validation (grid K∈{1,5,10,20}, cap 32). K and threshold are frozen before test.
4. **Selectivity readout unit.** Primary statistical unit = gene / homology-cluster (not nucleotide); codon activation = mean over the codon's 3 nucleotide positions (frozen; last-nucleotide variant reported as sensitivity). Confidence via gene/cluster blocked bootstrap.
5. **Sequence validity (treatment-independent QC, identical for baseline & steered).** Valid in-frame start(ATG)/stop, no internal stop, translated length ≥ 50 aa, ≥95% canonical amino acids, Evo2 perplexity ≤ baseline 95th percentile, low-complexity/entropy within bounds. **Never** filter on the DSSP helix outcome or on a treatment-dependent pLDDT cutoff.
6. **Primary %-helix readout.** ESMFold (fixed version/settings) → DSSP (3-state, H); report helix fraction over **all** residues and over **pLDDT≥70** residues separately; phrase results as **predicted** helix. pLDDT/PAE reported as diagnostics, not as a post-hoc filter defining the comparison set.
7. **Reporting.** Both intent-to-treat (all generated, failures assigned prespecified value) and conditional-on-valid means; off-target metrics (β-sheet/coil, AA/GC/length/entropy composition) with CIs; effect = absolute percentage-point Δhelix with gene/sequence-cluster bootstrap CI.
8. **α* selection hygiene.** α grid, seeds, and validity floor preregistered; α* chosen on validation seeds, performance at α* estimated on **disjoint** held-out seeds; dose-response reported with simultaneous CI band; α* replicated across seeds.
9. **Specificity controls.** (i) random-feature steering (K random latents matched for decoder norm & activation) must not raise helix; (ii) β-sheet-feature steering raises sheet not helix; (iii) helix gain must persist after adjusting for length/composition/perplexity.

## Key risks and mitigations

- **Degenerate optimum** (helix maximized by wrecking coding validity) → treatment-independent QC + ITT reporting + validity floor in α* definition + composition-adjusted effect.
- **Feature-selection leakage** (homologs across splits, nucleotide pseudo-replication) → mmseqs2 homology-cluster split, gene/cluster-level stats, matched nulls, FDR over 32768.
- **Under-defined steering** for TopK SAE → single frozen primary intervention with a portable α scale + sanity telemetry (target-latent activation, TopK support overlap, residual-norm/KL change).
- **Readout is predicted, not experimental** → dual all-residue/high-confidence reporting, "predicted" phrasing, random/β-sheet steering controls, optional second predictor on a subset.
- **ESMFold weights availability offline** → primary path is ESMFold→DSSP (DSSP present in env); if ESMFold weights are unreachable, the experiment stage substitutes an equivalent local per-residue secondary-structure predictor as the readout *tool* (an eval-tool choice, not a model/SAE downscale) and phrases results accordingly — flagged for the experiment stage, does not violate strict fidelity.

## Binding resources (strict)

Evo2-7B `/mnt/quarkfs/share_models/evo2_7b_262k`; SAE `/mnt/quarkfs/share_model/Evo-2-Layer-26-Mixed/sae-layer26-mixed-expansion_8-k_64.pt`; 8×A800-80GB, ≤4 concurrent, ≤8h; env `scientist`. Full scale, no substitution.
