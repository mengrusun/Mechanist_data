# Research Hypothesis: A Two-Stage Mechanism Inside ESMFold's Folding Trunk

## Motivation
Large protein structure-prediction models such as ESMFold produce accurate 3D structures end-to-end, but the internal computations of their folding trunk are largely opaque. The trunk consists of 48 iterative blocks that maintain two latent representations — a per-residue sequence representation `s` and a per-residue-pair representation `z` — coupled by two specific operations: the **seq2pair** pathway (an elementwise outer-product-style operation that writes information from `s` into `z`) and the **pair2seq** pathway (a learned attention bias derived from `z` that modulates sequence self-attention). Without a mechanistic decomposition of how these latents and pathways carry folding decisions, biologists cannot trust, audit, or controllably modify the model's structural predictions. A focused mechanistic study on a canonical structural motif (the β-hairpin) provides a tractable testbed for asking *how* — not just whether — the trunk decides which residues come into contact.

## Claim
You must validate the following three claims. There is to be no omission or delay in their experiments.
- **Folding decisions for β-hairpin structures are localized in the early blocks of the folding trunk.** Whether the trunk will fold a target region into a β-hairpin is committed to within early blocks, and the sequence representation `s` is the active locus of that decision during this window.

- **The early-block seq2pair pathway is the critical channel through which the β-hairpin folding decision is transferred from `s` into `z`.** In the early blocks, the **seq2pair** operation carries the decision to fold a target region into a β-hairpin from the sequence representation `s` into the pairwise representation `z`.

- **Charge is a linearly encoded chemical feature in early blocks of ESMFold and causally influences β-hairpin formation.** This charge feature exerts a causal effect on β-hairpin formation, consistent with the physical principle that opposite-charge residues on facing β-strands favor pairing (same-charge configurations correspondingly increase cross-strand distance).The evaluation of β-hairpin formation must be based on DSSP secondary-structure assignment on the predicted structure.



## Resources
Set DATA_DIR and MODEL_DIR once below, then use them throughout:

  DATA_DIR=<YOUR_DATA_DIR>
  MODEL_DIR=<YOUR_MODEL_DIR>

  You can find models and datasets in $DATA_DIR and $MODEL_DIR. You should use symbolic links if you want to use the models and datasets in the work_dir.
  If you can't find some models and datasets but you need them, you need to download them by yourself. All datasets should be placed under $DATA_DIR and all models under $MODEL_DIR. You can download from huggingface, github, modelscope.

- **Experiment stage**:
  - model: ESMFold — the full pipeline including the ESM-2 language-model backbone, the 48-block folding trunk, and the structure module (the sole model whose internals are being mechanistically analysed)
  - dataset: PDB (Protein Data Bank) structure files filtered by the **PISCES cull list** `cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055` — non-redundant chain whitelist (≤25% sequence identity, resolution 0.0–2.5 Å, R-factor ≤0.3, X-ray, chain length 40–10,000; 12,055 chains across 11,633 distinct PDB entries)
- **Verify stage — verify variants candidates (use as needed, not necessarily all)**:
  - models: (none — ESMFold is the sole mechanistic target)
  - datasets: **CATH classification** — hierarchical structural classification of PDB chains/domains (Class / Architecture / Topology / Homology levels), for stratified evaluation of the β-hairpin mechanistic findings

## hugging face token
<YOUR_HF_TOKEN>
## modelscope token
<YOUR_MODELSCOPE_TOKEN>
## Available API key
API_KEY = "<YOUR_API_KEY>"
BASE_URL = "<YOUR_BASE_URL>"
MODEL = "Access this API to retrieve the list of available models, select a suitable model from the list, and fill in the chosen MODEL name in task.md"

## Notice
- use conda env
- You have an 8-hour GPU budget. Do not pause experiments citing the GPU budget until actual GPU usage reaches this budget.
