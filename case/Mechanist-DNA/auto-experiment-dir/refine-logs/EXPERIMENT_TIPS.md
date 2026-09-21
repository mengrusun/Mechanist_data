# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning

## Matches

1. **steering-block-selection** — plan names a single steering site `blocks.26.post_norm`.
   - convention to adopt: the site is **not a free/copied index** — it is the SAE's native training site (expansion-8 k-64 BatchTopK SAE trained at `blocks.26`, `unit_sqrtd` input norm), validated in round-1 M(-1) by reproducing the Evo2-paper named Layer-26 features (β f/22326 rank-0, helix f/28741). HC1 pins it. Because the dictionary only exists at this one site, there is no block-count to sweep; the localization claim (C1/C3) is about SAE features *at this site*, so we intervene there and use feature-identity controls (matched-control + ≥30-dir random null) rather than block-count windows to keep it honest.

2. **steering-coefficient-tuning** — plan sweeps a steering strength (σ_proj-unit coefficient `c`, interior optimum `c*`).
   - convention to adopt: express the coefficient in **σ_proj units** (`σ_proj = std(hᵀ·unit_dir(S))` over a natural-CDS reference at the site) so doses are comparable — M1 computes σ_proj and persists the raw-α↔c map. Always include a `c=0` baseline; log a **capability/fluency metric** (valid-ORF rate + Evo2 perplexity/logit-drift) beside the target (pLDDT-weighted helix fraction) to detect off-distribution collapse. The **interior optimum `c*` = helix maximum within the capability-preserved region** (valid-ORF ≈ baseline), and we prefer the smallest sufficient dose. This is exactly plan P4/P6/P7 + M2's refined interior-bracketing grid. Compose order honored: site locked first (by construction), then coefficient swept on that locked site.

## No-match log
- **finetune-hyperparameter-sweep** — no fine-tuning of any kind (frozen Evo2-7B + frozen SAE; intervention is inference-time activation steering only). Not fired.
- **multiple-choice-evaluation** — endpoint is a continuous structural fraction (pLDDT-weighted α-helix fraction from DNA→protein→ESMFold/OmegaFold→DSSP), no A/B/A-D letter-parse of a generation. Not fired.
- **image** — genomic-sequence / protein-structure domain, no ImageNet/torchvision. Not fired.

## General Rule for Mechanism/Interpretability (always-on) — applied
- Feature set S is already localized (frozen round-1 discovery, re-confirmed by M0); we intervene on it directly (case: description/label already exists).
- General-ability guardrail measured **in parallel** with the target: valid-ORF rate, Evo2 perplexity/logit-drift, and pLDDT distribution are tracked at every dose/arm (M1 defines, M2/M3 report). The interior c* is defined precisely so capability ≈ baseline — a helix rise at a dose where generation has collapsed into gibberish/degenerate ORFs would be an artifact and is excluded by the capability-preserved-region rule (P7). Never report the helix metric alone.
