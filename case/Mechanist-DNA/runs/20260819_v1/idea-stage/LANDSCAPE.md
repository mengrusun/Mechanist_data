# Landscape: SAE feature amplification steering of Evo2 to raise α-helical content of generated DNA

**Date**: 2026-08-19
**Scope**: Interpreted as — using the released Goodfire Evo-2 Layer-26 Mixed SAE to (1) identify SAE latents selective for protein α-helix, (2) amplify them during autoregressive DNA generation, and (3) verify raised α-helical content of translated ORFs (via ESMFold→DSSP) vs. an unsteered baseline, finding the optimal amplification strength. Method-and-application, genomic-LM interpretability, 2023–2026.
**Based on**: ~12 retrieved papers + Evo2 primary paper — see `RESEARCH_LIT.md` for the raw dump.

---

## 1. Structured Paper Table

| Paper | Venue | Method | Key Result | Relevance to Us | Source |
|-------|-------|--------|------------|-----------------|--------|
| Evo 2 (Brixi et al.) | Nature 2026 | 40B/7B StripedHyena DNA LM + BatchTopK SAEs; feature discovery + steering | SAE features for exon boundaries, promoters, **α-helices/β-sheets/tRNAs** (thrT/tufB loci); features transfer across species; generates + scores sequences | **Direct precedent** — our behavior extends the paper's own α-helix SAE feature to *quantitative steering of generated helix content* | local PDF + web |
| Goodfire Evo-2-Layer-26-Mixed | HF model card | Tied TopK SAE, d=4096→32768 (×8), k=64, layer 26, mixed pro/euk | Released SAE we must use | **The exact mechanism** (task.md) | web + ckpt |
| Interpreting & Steering PLMs via SAEs (2502.09135) | arXiv 2025 | SAE on ESM-2; stat-associate latents to annotations, then steer | Latents for transmembrane/binding/motifs; steer generation | **Methodological blueprint** for select-then-amplify, in protein space | mech-db + web |
| InterPLM (2024) | arXiv/ICLR | SAE on ESM-2; feature↔concept correlation | 2,548 interpretable features/layer vs ≤46 neurons | Justifies SAE (not raw neurons) + F1/correlation feature-selection | mech-db |
| SAEs in Small Gene LMs (2507.07486) | arXiv 2025 | SAE on small GLMs | Confirms Evo2-SAE feature discovery as prior art | Situates our work as reproduction-plus on the full 7B | mech-db + web |
| Decode-gLM (2025.10.31.685860) | bioRxiv 2025 | Interpret/audit/**steer** genomic LMs via latents | Latents for resistance genes/enhancers steer generation | Closest concurrent "steer a genomic LM" work | web |
| CB-pLM (2024) | arXiv 2024 | Concept-bottleneck generative PLM; intervene on concepts | 3× larger targeted concept change vs baselines | Sets **dose–response / monotone control** expectation | mech-db |
| Steered Generation via GD on Sparse Features (2502.18644) | arXiv 2025 | Optimize sparse features to steer | Alt steering mechanics | Contrast to fixed-coefficient amplification | web |
| SAE-RNA (2025); SAE Low-N Protein (2025); ProtSAE (2025) | arXiv 2025 | SAE interpretability for RNA/protein LMs | Structure/function latents recovered | Cross-biomolecule generality of the recipe | mech-db |
| DSSP (Kabsch & Sander); ESMFold | JMB / Science | Secondary-structure assignment; fast structure prediction | "H"=α-helix; ESMFold ~60× faster than AF2 | **Evaluation pipeline** for %-helix of translated ORFs | web |

## 2. Core Landscape Narrative

**SAE interpretability of biological sequence models is now established, and Evo 2 is its genomic flagship.** The Evo 2 paper (Nature 2026) trained BatchTopK SAEs on model activations and recovered a broad spectrum of interpretable latents — exon boundaries, promoter motifs (70% HOCOMOCO hits), prophage elements — and, most directly for us, **features whose activation marks α-helices, β-sheets, and tRNAs** in coding loci (E. coli thrT/tufB). The authors explicitly frame SAE features as usable for "annotation, discovery and **steering** of sequence generations," and already report distributions of secondary structure for Evo 2-generated proteins. Our task is therefore a faithful, quantitative instantiation of a capability the model's own authors demonstrated qualitatively: confirm α-helix-selective SAE latents exist in the **released Layer-26 Mixed SAE**, then amplify them during autoregressive generation and measure the resulting change in helical content, sweeping to find the optimal strength.

**The select-then-steer recipe is well-trodden in the sibling protein-LM literature.** "Interpreting and Steering Protein Language Models through SAEs" (2502.09135) is the clearest blueprint: it statistically associates each SAE latent with protein annotations (giving a selectivity score), then uses the top latents to guide generation. InterPLM shows SAE latents are far more concept-aligned than raw neurons (2,548 vs ≤46 per layer), which is why feature *selection* should be done in SAE space with a selectivity statistic (correlation / F1 / AUROC of latent activation vs. per-residue α-helix label), not on neurons. CB-pLM contributes the key causal expectation for the amplification step: intervening on an interpretable concept produces a **large, targeted, monotone** change in that property (3× baseline), i.e. we should expect a dose–response curve of %-helix vs. amplification coefficient with an interior or saturating optimum before sequence quality collapses.

**Steering genomic LMs specifically is very recent (2025) and sparse.** Decode-gLM is the nearest concurrent effort to interpret/audit/steer genomic LMs via latent intervention, but demonstrates on resistance genes / enhancers, not protein secondary structure, and not on Evo 2's released Layer-26 SAE. This leaves a clean, well-scoped niche: **α-helix-directed generation on Evo 2-7B via its own published SAE, with a rigorous phenotype readout.**

**The evaluation bridge — DNA→protein→structure — is standard but must be built carefully.** Because Evo 2 emits nucleotides, α-helical content is a *protein* property of the translated product. The accepted pipeline: extract the ORF from each generated sequence (longest ORF / start–stop in frame), translate, predict structure with a fast folder (ESMFold, ~60× faster than AlphaFold2 and suitable for the hundreds–thousands of short sequences a sweep produces), and assign secondary structure with **DSSP** (the standard; "H" = α-helix), yielding per-sequence %-helix. Confounds to control (from the field's own cautions): sequences with no valid ORF, very short/failed folds, low pLDDT (unreliable structure), and coil-vs-helix mislabeling — all of which the plan must filter or report.

## 3. Sub-direction-Specific Work

- **Evo2 SAE feature discovery** — Evo2 paper + "SAEs in Small Gene LMs" — features are real and transferable; *gap*: no quantitative, dose-controlled steering of a structural phenotype using the released SAE.
- **Select-then-steer in protein LMs** — 2502.09135, InterPLM, CB-pLM — mature recipe (selectivity stat → amplify → measure); *gap*: not ported to a DNA LM where the phenotype requires translation.
- **Steering genomic LMs** — Decode-gLM, Steered-Generation-via-GD — emerging; *gap*: targets non-structural properties, different model/SAE.
- **Structure readout** — ESMFold + DSSP — standard tooling; *gap*: assembling an ORF→fold→DSSP %-helix scorer robust to junk generations is the main engineering risk.

## 4. Structural Gaps

- **Gap G1 — Quantitative, dose-controlled α-helix steering of a DNA LM via its released SAE.** *Competitive set*: Evo2 (qualitative feature only), 2502.09135 (protein, not DNA), Decode-gLM (non-structural). *Why open*: nobody has swept amplification strength for a *structural* phenotype on Evo2-7B + Layer-26 SAE and reported the optimum. **This is exactly the task.**
- **Gap G2 — α-helix label provenance for feature selection on DNA.** *Competitive set*: InterPLM/2502.09135 label residues directly from protein annotations; on DNA we must derive per-position α-helix labels by translating known/predicted coding sequences and running DSSP, then aligning back to nucleotide positions. *Why open*: the DNA→residue label alignment is the non-obvious methodological piece.
- **Gap G3 — Specificity of steering.** *Competitive set*: CB-pLM emphasizes targeted change. *Why open*: amplifying helix features may trivially degrade coding validity (fewer valid ORFs) rather than genuinely raise helix *among valid proteins* — the optimum must be defined on a quality-controlled subset with off-target/quality controls (ORF-validity rate, β-sheet/coil, perplexity).

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist)_
