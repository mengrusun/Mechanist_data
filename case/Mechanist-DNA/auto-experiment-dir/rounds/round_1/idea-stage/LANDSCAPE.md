# Research Landscape — SAE Feature Steering to Causally Control Protein Secondary Structure in a Genomic LM

**Date**: 2026-07-18
**Scope note**: This landscape supports a `given-validation` behavior capture — it is context for baselines, datasets, and metric definitions only. It never alters the behavior/claims taken from `task.md`.

## 1. Where the field is

Sparse autoencoders (SAEs) have become the dominant tool for decomposing the internal activations of biological sequence models into monosemantic, concept-aligned features:

- **Protein LMs (ESM-2).** InterPLM (ref paper) shows SAEs recover up to ~2,548 interpretable features/layer aligned to ~143 biological concepts (vs ≤46 for raw neurons), quantified by per-residue **concept↔feature correlation / F1**. Several 2025 works (ProtSAE, "Interpreting and Steering PLMs through SAEs", "SAEs for Low-N Protein Design", automated-neuron-labelling) then move from *interpreting* features to **steering** them: intervening on a latent during generation to bias output toward a structural/functional motif (transmembrane, binding site, zinc-finger, secondary/tertiary motifs).
- **Genomic LMs (Evo 2).** The Evo 2 paper (ref paper; Arc Institute × Goodfire) trains SAEs on Evo 2 residual activations with **no biological labels** and recovers features for exon/intron boundaries, TF motifs, CDS/UTR structure, and — crucially for this task — **protein secondary structure (α-helices, β-sheets)** read off directly from *nucleotide* activations. The released **Layer-26 mixed prokaryote/eukaryote BatchTopK SAE** (expansion 8, k 64) is the exact artifact pinned by `task.md`.
- **RNA LMs (SAE-RNA)** and **small gene LMs** replicate the pattern: SAE features localize to structurally meaningful regions.

## 2. The specific gap this project targets

Two things are established only *correlationally* or *in-modality* in the literature:

1. **Correlational only in Evo 2.** The Evo 2 paper shows α-helix features *activate on* helix-coding regions (an activation map), and explicitly states that **steering these features to guide DNA generation is "still in its early stages."** Whether amplifying an α-helix feature *causally* raises helical content — and whether the effect is **dose-responsive** and **specific** — is not established.
2. **In-modality steering only.** Prior causal steering (Interpreting-and-Steering-PLMs, FoldSAE, automated-neuron-labelling) operates on **protein** LMs in **amino-acid space**, where the steered feature and the measured property live in the same sequence. This project steers a feature in a **DNA/nucleotide** model and measures the effect on the **downstream protein's** secondary structure — a **cross-modal causal claim** (genomic-LM feature → translated-protein structure) that has to survive the DNA→codon→protein→fold→DSSP pipeline.

→ The project's contribution is to convert the Evo 2 α-helix feature from a *correlational annotation* into a **causally manipulable, dose-responsive, specific knob**, verified end-to-end with structure prediction + DSSP.

## 3. Methodological anchors (adopted into the plan)

- **Feature-concept alignment metric** (InterPLM, CorrSteer, Model-X-knockoffs paper): assign a feature to "α-helix" by a per-token/per-codon **precision / recall / F1** (or AUROC) of the feature's activation against a real α-helix label track, with a selectivity margin over non-helix and over other secondary-structure classes. Use FDR-style control so the "α-helix feature set" is not an artifact of multiple testing.
- **Data construction (hard rule from `task.md`, corroborated by the field):** do **NOT** reverse-translate proteins to DNA (codon degeneracy → noise + unnatural DNA). Instead take **natural CDS** from databases and attach **real DSSP secondary-structure labels** from experimental PDB structures after proper **CDS↔protein alignment** (codon *i* ↔ residue *i*). This is the InterPLM-style "real annotation track" discipline, adapted to the nucleotide domain.
- **Steering / dose-response** (CorrSteer, Interpreting-and-Steering-PLMs): amplify the identified feature(s) at the SAE site during autoregressive generation, **sweep the amplification coefficient** (dose), and report a monotone dose-response curve plus an **optimal** strength before sequence degeneration.
- **Specificity controls** (CorrSteer side-effect analysis; general mechanism rigor): matched random/unrelated features and a β-sheet feature as off-target controls; check that generation quality (perplexity, valid ORF, coding fraction) does not collapse.
- **Structural verification**: predict the structure of the protein encoded by each generated DNA sequence (ESMFold / AlphaFold per tooling availability) and compute α-helix fraction with **DSSP**; compare across doses and against unsteered baselines.

## 4. Baselines the plan should include
- **Unsteered Evo 2 generation** (coefficient = 0) — the null dose.
- **Matched-control feature steering** — a random/unrelated SAE feature at the same coefficient (specificity).
- **Off-target structural feature** (β-sheet feature) steering — should raise β-sheet, not α-helix (double dissociation).
- **Naïve sequence baseline** (optional): natural helix-rich vs helix-poor CDS as reference distributions for the α-helix-fraction metric.

## 5. Open problems / risks the plan must handle
- Cross-modal noise: translation frame, premature stop codons, non-foldable generations → need ORF/validity filters before structure prediction.
- Structure-prediction reliability on short/de-novo sequences → confidence (pLDDT) gating and enough samples per dose for statistical power.
- Feature-set identification robustness → confounds (codon frequency, GC content, position, length) controlled in M0.
