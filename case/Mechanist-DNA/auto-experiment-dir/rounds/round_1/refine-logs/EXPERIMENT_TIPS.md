# Experiment Tips Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
matched_tips:
  - steering-block-selection
  - steering-coefficient-tuning
  - general-rule-mechanism-interpretability

## Matches

1. **general-rule-mechanism-interpretability** — this is an SAE-feature-steering interpretability experiment (locate α-helix-selective Layer-26 SAE features, then intervene by amplifying them).
   - convention to adopt: measure a **general-ability / off-target metric in parallel** with the target metric at every dose — here: valid-ORF rate, Evo2 perplexity, pLDDT distribution, and **β-sheet fraction** alongside the target α-helix fraction. A helix rise that co-occurs with sequence/structure collapse (gibberish ORFs, pLDDT crash) is an artifact, not a knob. This is exactly M1's calibration + M3's quality guardrail.

2. **steering-block-selection** — steering site is declared (the Layer-26 SAE feature space).
   - convention to adopt: the site is **pinned by HC1** (only the Layer-26 mixed SAE exists on disk; no layer sweep is possible or permitted). The block-selection freedom is therefore spent on the **feature-set** choice within Layer-26, which M0 fixes rigorously (AUROC/F1 + selectivity margin + FDR). Honesty rule applied: M3 uses **matched-null feature controls** (random features matched on activation frequency/magnitude) and an **off-target β-sheet feature** as the "matched window elsewhere" null — a single feature can't adjudicate a specificity claim.

3. **steering-coefficient-tuning** — M2 sweeps a steering coefficient α ∈ {-2, 0, 1, 2, 4, 8, 16, 32}.
   - convention to adopt: (a) always include the **α=0 baseline** and a **negative-α sign check** (both already in the grid); (b) log a **fluency/general-ability metric** (valid-ORF rate, perplexity, pLDDT) at every dose so **collapse** is visible, not just the target effect; (c) prefer the **smallest sufficient α** — report **α\*** = max helix fraction *subject to* generation-quality tolerance, and identify the **degradation onset**; (d) express dose in interpretable per-feature units (α · s_f where s_f is the feature's active-activation scale), the SAE-feature analogue of β·σ_proj. M1 calibrates the measurement noise / samples-per-dose so the sweep detects Δhelix ≥ ~0.1 at power 0.8.

## Composition applied
Block-count/site is **fixed** (Layer-26 SAE, HC1) → the coefficient sweep (M2) is meaningful on that locked site. Coefficient tuning (tip 2) + parallel general-ability metric (general rule) are the load-bearing tips; block-selection (tip 3) mainly supplies the null-control discipline realized in M3.

## No-match log
- Tips 1 (ImageNet), 4 (fine-tuning sweep), 5 (MCQ letter-parse) do not fire — no vision preprocessing, no fine-tuning (the SAE and Evo2 are frozen; only inference-time steering), and the readout is a continuous DSSP α-helix fraction, not a letter-parse of a free-form generation.
