# Reference Paper Summaries

Two papers named in `task.md`, summarized for the methodology they contribute to this project's M0 (feature discovery) and mechanism (steering + verification) design.

---

## Paper A — Evo 2: Genome modelling and design across all domains of life

**Authors/venue**: Arc Institute, NVIDIA, et al. (Nature 2026; bioRxiv 2025.02.18.638918). SAE interpretability in collaboration with Goodfire.

### What they did
Evo 2 is a genomic foundation model (**StripedHyena2** architecture, **7B and 40B** parameters, **single-nucleotide** resolution, up to **1 Mb** context) trained on **~9.3 trillion nucleotides** spanning prokaryotic and eukaryotic genomes. Beyond prediction/design, they apply **sparse autoencoders (SAEs)** to Evo 2's residual-stream activations — trained with **no biological labels** — to decompose them into interpretable features.

### Key results relevant here
- SAE features recover **canonical gene structure** (CDS, UTRs, exon/intron boundaries), **regulatory motifs** (TF binding sites), **organism-level** signals (prophage, CRISPR), and — critically — **protein secondary structure** features that fire on **α-helix** and **β-sheet** coding regions, plus RNA stem-loops/tRNA. The α-helix association is shown as **activation maps over sequence** — i.e. **correlational** evidence that the feature *co-occurs* with helical coding regions.
- The released **Layer-26 mixed prokaryote/eukaryote SAE** (HF `Goodfire/Evo-2-Layer-26-Mixed`) is a **BatchTopK** SAE with **expansion factor 8** and **k = 64** active latents. For Evo2-7B (hidden dim 4096) this yields a **~32,768-feature dictionary**. This is the exact artifact at `/data1/share_model/evo2/evo2_sae_layer26_mixed/`.
- The paper notes SAE-feature **steering to guide DNA generation is promising but "still in its early stages."**

### What this project takes from it (and the gap it fills)
Evo 2 establishes the α-helix feature *correlationally* and leaves **causal, dose-responsive, specific** steering open. This project pins the **same model + same Layer-26 SAE** and converts the correlational feature into a **causally manipulable knob**: identify the α-helix feature set with rigorous natural-CDS+DSSP labels (M0), then **amplify** it during autoregressive generation and measure a **dose-response** rise in encoded-protein helical content (M2), with specificity controls (M3).

---

## Paper B — InterPLM: Discovering Interpretable Features in Protein Language Models via SAEs

**Authors/venue**: Simon, Zou, et al. (2024).

### What they did
Train **SAEs on ESM-2** residual embeddings and systematically extract interpretable latent features, then **align each feature to known biological concepts** and demonstrate downstream uses (annotation-filling, generation steering).

### Key results relevant here
- SAE features capture **far more biological annotations than raw neurons** — up to **~2,548 interpretable features per layer** mapping to up to **~143 known biological concepts**, vs **≤46** concept-selective individual neurons per layer.
- **Feature↔concept alignment metric (the key methodology this project adopts):** for each (feature, concept) pair, treat the feature's per-residue activation (thresholded) as a binary predictor of the concept label and compute **precision / recall / F1** over residues; a feature is called concept-selective when its F1 (with a selectivity margin) is high. Concept labels come from **UniProtKB/Swiss-Prot** annotations and structure-derived tracks; secondary-structure concepts are included.
- Features cluster by shared functional/structural role; LLM auto-descriptions (Claude-3.5 Sonnet) correlate with the concept labels; features can **steer** ESM generation.

### What this project takes from it
- The **per-residue F1 concept-alignment** procedure — applied at **codon resolution** in the DNA domain: label each codon of a natural CDS with the DSSP secondary-structure class of its aligned residue, then score each Layer-26 SAE feature's α-helix-vs-rest discrimination by **F1 / AUROC** with **confound control** and **FDR control**. This is the M0 selectivity test.
- The discipline of aligning model positions to **real** structural labels (not synthetic) — reinforcing `task.md`'s prohibition on reverse-translation and its requirement for natural CDS + real DSSP labels after proper CDS↔protein alignment.

---

## Consolidated design implications
1. **M0 selectivity metric** = per-codon F1/AUROC of feature activation vs real DSSP α-helix label on natural CDS, with confound + FDR control (InterPLM method, DNA-domain adaptation).
2. **Steering site** = the Layer-26 SAE feature space (BatchTopK, k=64, ~32,768 features); amplify the α-helix feature set's activation during autoregressive nucleotide generation.
3. **Verification** = translate generated DNA (valid-ORF filter) → predict structure (ESMFold/AlphaFold) → DSSP α-helix fraction; compare across doses and controls.
4. **Novel contribution over both papers** = a **cross-modal causal** result (genomic-LM feature → downstream *protein* structure) with dose-response + specificity, which Evo 2 left as "early stages" and InterPLM only did in-modality on protein LMs.
